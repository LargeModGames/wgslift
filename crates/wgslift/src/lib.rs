//! wgslift translates Direct3D 9 era HLSL shaders to WGSL.
//!
//! The front end is the `wgslift-syntax` crate. This crate checks the types, lowers
//! the program to a typed IR and prints WGSL.

mod ir;
pub mod reflection;
mod sema;
#[cfg(test)]
mod tests;
mod types;
mod wgsl;

use std::fmt;

pub use reflection::Reflection;
pub use types::SamplerDim;
pub use wgslift_syntax as syntax;
use wgslift_syntax::{Diagnostic, IncludeResolver, Severity, SourceFiles};

/// The shader stage of the entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// A pixel shader: a WGSL `@fragment` function.
    Fragment,
}

/// What to do about WGSL's rule that derivatives and implicit-level sampling run
/// in uniform control flow. Old pixel shaders break the rule often.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uniformity {
    /// Print `diagnostic(off, derivative_uniformity);` at the top of the module.
    DirectiveOff,
    /// Print nothing: a validator that checks uniformity can reject the output.
    Keep,
}

/// Switches for the behaviour of older compilers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dialect {
    /// `pow`, `log`, `log2`, `log10`, `sqrt` and `rsqrt` use the absolute value of
    /// their input, as the Direct3D 9 instructions do.
    pub abs_domain_math: bool,
    /// Global variables that the shader writes become private variables.
    pub writable_globals: bool,
    /// A variable that a `for` initializer declares stays visible after the loop.
    pub for_init_outer_scope: bool,
    /// The uniformity policy.
    pub uniformity: Uniformity,
}

impl Dialect {
    /// Direct3D 9 shader models 2 and 3, as the FXC compiler of that time read them.
    pub fn d3d9_legacy() -> Dialect {
        Dialect {
            abs_domain_math: true,
            writable_globals: true,
            for_init_outer_scope: true,
            uniformity: Uniformity::DirectiveOff,
        }
    }
}

impl Default for Dialect {
    fn default() -> Dialect {
        Dialect::d3d9_legacy()
    }
}

/// Settings for one translation.
pub struct Options<'a> {
    /// The stage of the entry point.
    pub stage: Stage,
    /// The HLSL name of the entry-point function.
    pub entry_point: String,
    /// The dialect switches.
    pub dialect: Dialect,
    /// Macros defined before the source, as name and value.
    pub defines: Vec<(String, String)>,
    /// HLSL text that is preprocessed before the source, for macros the source
    /// uses. Its declarations are part of the program.
    pub prelude: Option<String>,
    /// Top-level declarations added only when the source does not declare the same
    /// name: (name, HLSL text of the declaration).
    pub predeclare: Vec<(String, String)>,
    /// Finds `#include` files. Without one, every `#include` is an error.
    pub includes: Option<&'a mut dyn IncludeResolver>,
    /// The most loop passes one invocation of the entry point can make, summed over
    /// all loops. When the budget is spent, every loop exits. Use it for untrusted
    /// shaders: a loop that never ends can hang the GPU.
    pub loop_limit: Option<u32>,
}

impl Default for Options<'_> {
    fn default() -> Self {
        Options {
            stage: Stage::Fragment,
            entry_point: "main".to_string(),
            dialect: Dialect::d3d9_legacy(),
            defines: Vec::new(),
            prelude: None,
            predeclare: Vec::new(),
            includes: None,
            loop_limit: None,
        }
    }
}

/// A successful translation.
#[derive(Debug, Clone)]
pub struct Output {
    /// The WGSL module.
    pub wgsl: String,
    /// The bindings and entry-point interface of the module.
    pub reflection: Reflection,
    /// Warnings.
    pub diagnostics: Vec<Diagnostic>,
    /// The source files, for the spans in `diagnostics`.
    pub files: SourceFiles,
}

/// A failed translation: at least one diagnostic is an error.
#[derive(Debug, Clone)]
pub struct Failed {
    /// Errors and warnings.
    pub diagnostics: Vec<Diagnostic>,
    /// The source files, for the spans in `diagnostics`.
    pub files: SourceFiles,
}

impl fmt::Display for Failed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for d in &self.diagnostics {
            writeln!(f, "{}", d.display(&self.files))?;
        }
        Ok(())
    }
}

impl std::error::Error for Failed {}

/// Translates the HLSL source `text` to WGSL. `name` is the file name in
/// diagnostics.
pub fn translate(name: &str, text: &str, options: Options<'_>) -> Result<Output, Failed> {
    let Options {
        stage,
        entry_point,
        dialect,
        defines,
        prelude,
        predeclare,
        includes,
        loop_limit,
    } = options;
    let parsed = wgslift_syntax::parse(
        name,
        text,
        wgslift_syntax::Options {
            defines,
            prelude,
            includes,
        },
    );
    let mut files = parsed.files;
    let mut diagnostics = parsed.diagnostics;
    let mut units = vec![parsed.unit];
    let declared: std::collections::HashSet<String> = sema::top_level_names(&units[0])
        .into_iter()
        .map(str::to_owned)
        .collect();
    for (decl_name, decl) in &predeclare {
        if declared.contains(decl_name.as_str()) {
            continue;
        }
        let extra = wgslift_syntax::parse(
            &format!("<predeclared {decl_name}>"),
            decl,
            Default::default(),
        );
        for d in extra.diagnostics {
            diagnostics.push(Diagnostic::error(
                d.span,
                format!("in the predeclared `{decl_name}`: {}", d.message),
            ));
        }
        units.push(extra.unit);
        // A predeclared unit has its own file table; its spans only matter for its
        // own errors, which are reported above.
        files.add(format!("<predeclared {decl_name}>"), decl.clone());
    }
    let failed = |diagnostics: Vec<Diagnostic>, files: SourceFiles| Failed { diagnostics, files };
    if diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return Err(failed(diagnostics, files));
    }
    let settings = sema::Settings {
        stage,
        entry_point: &entry_point,
        dialect: &dialect,
    };
    let Some(module) = sema::analyze(&units, &settings, &mut diagnostics) else {
        return Err(failed(diagnostics, files));
    };
    if diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return Err(failed(diagnostics, files));
    }
    let (wgsl, reflection) = wgsl::print(
        &module,
        &wgsl::PrintOptions {
            stage,
            uniformity: dialect.uniformity,
            loop_limit,
        },
    );
    Ok(Output {
        wgsl,
        reflection,
        diagnostics,
        files,
    })
}
