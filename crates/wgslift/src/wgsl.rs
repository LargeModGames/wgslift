//! The WGSL printer: typed IR to WGSL text and reflection.

mod body;
mod entry;
mod names;
mod uniforms;

use std::fmt::Write;

use crate::ir::{self, Space};
use crate::reflection::{Reflection, TextureBinding};
use crate::types::{SamplerDim, Type};
use crate::{Stage, Uniformity};

use body::Body;
use names::Namer;
use uniforms::{NameTable, Uniforms};

/// Settings for the printer.
pub(crate) struct PrintOptions {
    /// The stage of the entry point.
    pub stage: Stage,
    /// The uniformity policy.
    pub uniformity: Uniformity,
}

/// Prints `module` as WGSL and reports its bindings and interface.
pub(crate) fn print(module: &ir::Module, options: &PrintOptions) -> (String, Reflection) {
    let Stage::Fragment = options.stage;
    Printer::new(module).print(options.uniformity)
}

/// The WGSL names of everything at module scope.
struct Names {
    structs: Vec<String>,
    fields: Vec<Vec<String>>,
    /// Globals; for a sampler global, the texture name.
    globals: Vec<String>,
    /// For a sampler global, the sampler name.
    samplers: Vec<String>,
    functions: Vec<String>,
    /// The name of the generated `@fragment` function.
    wrapper: String,
    uniform_var: String,
    /// Every module-scope name, to seed the namers of function scopes.
    module: Namer,
}

/// A module with its names and uniform plan.
struct Printer<'m> {
    module: &'m ir::Module,
    names: Names,
    uniforms: Uniforms,
}

impl<'m> Printer<'m> {
    fn new(module: &'m ir::Module) -> Printer<'m> {
        let mut namer = Namer::default();
        let structs: Vec<String> = module.structs.iter().map(|s| namer.name(&s.name)).collect();
        let fields: Vec<Vec<String>> = module
            .structs
            .iter()
            .map(|s| {
                let mut members = Namer::members();
                s.fields.iter().map(|f| members.name(&f.name)).collect()
            })
            .collect();
        let mut globals = Vec::new();
        let mut samplers = Vec::new();
        for g in &module.globals {
            if g.space == Space::Resource {
                globals.push(namer.name(&format!("{}_texture", g.name)));
                samplers.push(namer.name(&format!("{}_sampler", g.name)));
            } else {
                globals.push(namer.name(&g.name));
                samplers.push(String::new());
            }
        }
        let wrapper = match module.functions.get(module.entry) {
            Some(f) => namer.name(&f.name),
            None => namer.name("main"),
        };
        let functions = module
            .functions
            .iter()
            .map(|f| namer.name(&f.name))
            .collect();
        let uniform_struct = namer.name("Uniforms");
        let uniform_var = namer.name("uniforms");
        let members: Vec<(usize, String)> = module
            .globals
            .iter()
            .enumerate()
            .filter(|(_, g)| g.space == Space::Uniform)
            .map(|(i, _)| (i, globals[i].clone()))
            .collect();
        let uniforms = Uniforms::plan(
            module,
            &members,
            &NameTable {
                structs: &structs,
                fields: &fields,
            },
            &mut namer,
            &uniform_struct,
            &uniform_var,
        );
        Printer {
            module,
            names: Names {
                structs,
                fields,
                globals,
                samplers,
                functions,
                wrapper,
                uniform_var,
                module: namer,
            },
            uniforms,
        }
    }

    fn table(&self) -> NameTable<'_> {
        NameTable {
            structs: &self.names.structs,
            fields: &self.names.fields,
        }
    }

    /// The WGSL spelling of a type.
    fn ty(&self, ty: &Type) -> String {
        self.table().ty(ty)
    }

    fn print(mut self, uniformity: Uniformity) -> (String, Reflection) {
        let mut out = String::new();
        let mut reflection = Reflection {
            uniform_buffer: self.uniforms.reflection.take(),
            ..Reflection::default()
        };
        if uniformity == Uniformity::DirectiveOff {
            out.push_str("diagnostic(off, derivative_uniformity);\n\n");
        }
        self.print_structs(&mut out);
        out.push_str(&self.uniforms.decls);
        self.print_textures(&mut out, &mut reflection);
        self.print_globals(&mut out);
        for f in 0..self.module.functions.len() {
            out.push_str(&Body::function(&self, f));
            out.push('\n');
        }
        if self.module.functions.get(self.module.entry).is_some() {
            let (text, inputs, outputs) = entry::wrapper(&self);
            out.push_str(&text);
            reflection.inputs = inputs;
            reflection.outputs = outputs;
        }
        (out, reflection)
    }

    fn print_structs(&self, out: &mut String) {
        for (i, s) in self.module.structs.iter().enumerate() {
            let _ = writeln!(out, "struct {} {{", self.names.structs[i]);
            if s.fields.is_empty() {
                out.push_str("    pad: u32,\n");
            }
            for (f, field) in s.fields.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "    {}: {},",
                    self.names.fields[i][f],
                    self.ty(&field.ty)
                );
            }
            out.push_str("}\n\n");
        }
    }

    fn print_textures(&self, out: &mut String, reflection: &mut Reflection) {
        let mut binding = u32::from(reflection.uniform_buffer.is_some());
        for (i, g) in self.module.globals.iter().enumerate() {
            if g.space != Space::Resource {
                continue;
            }
            let dim = match g.ty {
                Type::Sampler(SamplerDim::Any) => SamplerDim::D2,
                Type::Sampler(dim) => dim,
                _ => SamplerDim::D2,
            };
            let texture = match dim {
                SamplerDim::D1 => "texture_1d<f32>",
                SamplerDim::D3 => "texture_3d<f32>",
                SamplerDim::Cube => "texture_cube<f32>",
                SamplerDim::D2 | SamplerDim::Any => "texture_2d<f32>",
            };
            let _ = writeln!(
                out,
                "@group(0) @binding({binding}) var {}: {texture};",
                self.names.globals[i]
            );
            let _ = writeln!(
                out,
                "@group(0) @binding({}) var {}: sampler;\n",
                binding + 1,
                self.names.samplers[i]
            );
            reflection.textures.push(TextureBinding {
                name: g.name.clone(),
                dim,
                texture_binding: binding,
                sampler_binding: binding + 1,
            });
            binding += 2;
        }
    }

    fn print_globals(&self, out: &mut String) {
        let mut any = false;
        for (i, g) in self.module.globals.iter().enumerate() {
            let keyword = match g.space {
                Space::Private => "var<private>",
                Space::Const => "const",
                Space::Uniform | Space::Resource => continue,
            };
            any = true;
            let ty = self.ty(&g.ty);
            let init = g.init.as_ref().map(|e| Body::module_scope(self).top(e));
            let init = match (init, g.space) {
                (Some(init), _) => format!(" = {init}"),
                (None, Space::Const) => format!(" = {ty}()"),
                (None, _) => String::new(),
            };
            let _ = writeln!(out, "{keyword} {}: {ty}{init};", self.names.globals[i]);
        }
        if any {
            out.push('\n');
        }
    }
}
