//! The front end of wgslift: an HLSL preprocessor, lexer and parser.
//!
//! It turns HLSL source text into a syntax tree with source spans. It does no type
//! checking and no translation; the `wgslift` crate does that on top of this tree.
//!
//! ```
//! let parsed = wgslift_syntax::parse("main.hlsl", "float4 tint : register(c0);", Default::default());
//! assert!(!parsed.has_errors());
//! assert_eq!(parsed.unit.items.len(), 1);
//! ```

pub mod ast;
pub mod diagnostic;
pub mod lexer;
pub mod parser;
pub mod pp;
pub mod span;

pub use ast::TranslationUnit;
pub use diagnostic::{Diagnostic, Severity};
pub use lexer::{Punct, Token, TokenKind};
pub use pp::{IncludeResolver, NoIncludes};
pub use span::{FileId, SourceFile, SourceFiles, Span};

/// Settings for one parse.
#[derive(Default)]
pub struct Options<'a> {
    /// Macros defined before the source, as name and value.
    pub defines: Vec<(String, String)>,
    /// Finds `#include` files. Without one, every `#include` is an error.
    pub includes: Option<&'a mut dyn IncludeResolver>,
}

/// The result of a parse.
#[derive(Debug, Clone)]
pub struct Parsed {
    /// The syntax tree. After errors it holds every item that parsed.
    pub unit: TranslationUnit,
    /// The main file, every included file and one `<define NAME>` file for each
    /// predefined macro, for diagnostics and spans.
    pub files: SourceFiles,
    /// Errors and warnings in source order of discovery.
    pub diagnostics: Vec<Diagnostic>,
}

impl Parsed {
    /// At least one diagnostic is an error.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    }
}

/// Preprocesses and parses the HLSL source `text`. `name` is the file name in
/// diagnostics.
pub fn parse(name: &str, text: &str, options: Options<'_>) -> Parsed {
    let mut files = SourceFiles::default();
    let main = files.add(name, text);
    let mut diagnostics = Vec::new();
    let mut no_includes = NoIncludes;
    let includes: &mut dyn IncludeResolver = match options.includes {
        Some(resolver) => resolver,
        None => &mut no_includes,
    };
    let tokens = pp::preprocess(
        &mut files,
        main,
        &options.defines,
        includes,
        &mut diagnostics,
    );
    let eof = Span::new(main, text.len(), text.len());
    let unit = parser::parse_tokens(tokens, eof, &mut diagnostics);
    Parsed {
        unit,
        files,
        diagnostics,
    }
}

#[cfg(test)]
mod tests;
