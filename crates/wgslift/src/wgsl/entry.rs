//! The generated `@fragment` function around the entry point.

use std::fmt::Write;

use crate::ir::ParamDir;
use crate::reflection::IoBinding;
use crate::types::{Scalar, Type};

use super::Printer;
use super::body::Body;
use super::names::Namer;

/// One member of the input or output struct.
struct Slot {
    field: String,
    location: Option<u32>,
    builtin: Option<&'static str>,
    /// The WGSL type of the member.
    ty: String,
    /// Integer inputs need flat interpolation.
    flat: bool,
    name: String,
    semantic: String,
}

/// The members of one interface struct.
struct Interface {
    slots: Vec<Slot>,
    members: Namer,
}

impl Interface {
    fn new() -> Interface {
        Interface {
            slots: Vec::new(),
            members: Namer::members(),
        }
    }

    /// Adds a member and returns its field name.
    fn add(
        &mut self,
        name: &str,
        semantic: String,
        location: Option<u32>,
        builtin: Option<&'static str>,
        ty: String,
        flat: bool,
    ) -> String {
        if let Some(b) = builtin
            && let Some(slot) = self.slots.iter().find(|s| s.builtin == Some(b))
        {
            return slot.field.clone();
        }
        let field = self.members.name(name);
        self.slots.push(Slot {
            field: field.clone(),
            location,
            builtin,
            ty,
            flat,
            name: name.to_string(),
            semantic,
        });
        field
    }

    /// Gives each member without a location the lowest free one.
    fn assign_locations(&mut self) {
        for i in 0..self.slots.len() {
            if self.slots[i].location.is_some() || self.slots[i].builtin.is_some() {
                continue;
            }
            let free = (0..)
                .find(|l| !self.slots.iter().any(|s| s.location == Some(*l)))
                .unwrap_or(0);
            self.slots[i].location = Some(free);
        }
    }

    fn declare(&self, out: &mut String, name: &str) {
        let _ = writeln!(out, "struct {name} {{");
        for slot in &self.slots {
            let attr = match (slot.builtin, slot.location) {
                (Some(b), _) => format!("@builtin({b})"),
                (None, l) => format!("@location({})", l.unwrap_or(0)),
            };
            let flat = if slot.flat { " @interpolate(flat)" } else { "" };
            let _ = writeln!(out, "    {attr}{flat} {}: {},", slot.field, slot.ty);
        }
        out.push_str("}\n\n");
    }

    fn reflect(&self) -> Vec<IoBinding> {
        self.slots
            .iter()
            .map(|s| IoBinding {
                name: s.name.clone(),
                semantic: s.semantic.clone(),
                location: if s.builtin.is_some() {
                    None
                } else {
                    s.location
                },
                builtin: s.builtin.map(str::to_string),
                wgsl_type: s.ty.clone(),
            })
            .collect()
    }
}

/// A semantic split into its upper-case name and its index.
fn split_semantic(semantic: &str) -> (String, &str, u32) {
    let digits = semantic.len()
        - semantic
            .bytes()
            .rev()
            .take_while(u8::is_ascii_digit)
            .count();
    let (base, index) = semantic.split_at(digits);
    (base.to_ascii_uppercase(), base, index.parse().unwrap_or(0))
}

/// Builds the interface of the entry point.
struct Wrapper<'b, 'p, 'm> {
    p: &'p Printer<'m>,
    body: &'b mut Body<'p, 'm>,
    inputs: Interface,
    outputs: Interface,
    input_var: String,
    output_var: String,
    stores: Vec<String>,
}

impl Wrapper<'_, '_, '_> {
    /// A WGSL expression that reads an entry-point input of type `ty`.
    fn input(&mut self, name: &str, ty: &Type, semantic: Option<&str>) -> String {
        let p = self.p;
        let Some(semantic) = semantic else {
            if let Type::Struct(s) = ty {
                let parts: Vec<String> = p.module.structs[*s]
                    .fields
                    .iter()
                    .map(|f| self.input(&f.name, &f.ty, f.semantic.as_deref()))
                    .collect();
                return format!("{}({})", p.ty(ty), parts.join(", "));
            }
            return format!("{}()", p.ty(ty));
        };
        let (upper, base, index) = split_semantic(semantic);
        match upper.as_str() {
            "VPOS" | "SV_POSITION" | "POSITION" => {
                let field = self.inputs.add(
                    name,
                    semantic.to_string(),
                    None,
                    Some("position"),
                    "vec4<f32>".to_string(),
                    false,
                );
                self.position(ty, &field, upper == "VPOS")
            }
            "VFACE" | "SV_ISFRONTFACE" => {
                let field = self.inputs.add(
                    name,
                    semantic.to_string(),
                    None,
                    Some("front_facing"),
                    "bool".to_string(),
                    false,
                );
                let read = format!("{}.{field}", self.input_var);
                match ty {
                    Type::Scalar(Scalar::Bool) => read,
                    Type::Scalar(_) => format!("{}(select(-1.0f, 1.0f, {read}))", p.ty(ty)),
                    _ => format!("{}()", p.ty(ty)),
                }
            }
            _ => {
                let location = match upper.as_str() {
                    "TEXCOORD" => Some(index),
                    "COLOR" => Some(8 + index),
                    _ => None,
                };
                let slots = match ty {
                    Type::Scalar(_) | Type::Vector(..) => {
                        return self.located(name, semantic.to_string(), location, ty);
                    }
                    Type::Matrix { rows, cols } => {
                        (u32::from(*rows), Type::Vector(Scalar::Float, *cols))
                    }
                    Type::Array(elem, len)
                        if matches!(**elem, Type::Scalar(_) | Type::Vector(..)) =>
                    {
                        (*len, (**elem).clone())
                    }
                    _ => return format!("{}()", p.ty(ty)),
                };
                let parts: Vec<String> = (0..slots.0)
                    .map(|i| {
                        self.located(
                            &format!("{name}_{i}"),
                            format!("{base}{}", index + i),
                            location.map(|l| l + i),
                            &slots.1,
                        )
                    })
                    .collect();
                format!("{}({})", p.ty(ty), parts.join(", "))
            }
        }
    }

    /// Reads one scalar or vector input at a location.
    fn located(
        &mut self,
        name: &str,
        semantic: String,
        location: Option<u32>,
        ty: &Type,
    ) -> String {
        let scalar = ty.scalar().unwrap_or(Scalar::Float);
        let stored = match scalar {
            Scalar::Bool => ty.with_scalar(Scalar::Uint),
            _ => ty.clone(),
        };
        let stored_text = self.p.ty(&stored);
        let flat = scalar != Scalar::Float;
        let field = self
            .inputs
            .add(name, semantic, location, None, stored_text.clone(), flat);
        let read = format!("{}.{field}", self.input_var);
        match ty {
            Type::Scalar(Scalar::Bool) => format!("({read} != 0u)"),
            Type::Vector(Scalar::Bool, _) => format!("({read} != {stored_text}(0u))"),
            _ => read,
        }
    }

    /// Reads the fragment position as `ty`. D3D9 `VPOS` has pixel centres at
    /// whole numbers, half a pixel before WGSL `position`.
    fn position(&self, ty: &Type, field: &str, vpos: bool) -> String {
        let Some(size) = ty.size() else {
            return format!("{}()", self.p.ty(ty));
        };
        let swizzle = match size {
            1 => ".x",
            2 => ".xy",
            3 => ".xyz",
            _ => "",
        };
        let mut read = format!("{}.{field}{swizzle}", self.input_var);
        if vpos {
            let half = match size {
                1 => "0.5f".to_string(),
                n => format!("vec{n}<f32>(0.5f)"),
            };
            read = format!("({read} - {half})");
        }
        if ty.scalar() != Some(Scalar::Float) {
            read = format!("{}({read})", self.p.ty(ty));
        }
        read
    }

    /// Stores `value`, of type `ty`, into the outputs.
    fn output(&mut self, name: &str, ty: &Type, semantic: Option<&str>, value: &str) {
        let p = self.p;
        let Some(semantic) = semantic else {
            if let Type::Struct(s) = ty {
                for (f, field) in p.module.structs[*s].fields.iter().enumerate() {
                    let read = format!("{value}.{}", p.names.fields[*s][f]);
                    self.output(&field.name, &field.ty, field.semantic.as_deref(), &read);
                }
            }
            return;
        };
        let (upper, _, index) = split_semantic(semantic);
        if matches!(upper.as_str(), "DEPTH" | "SV_DEPTH") {
            let read = match ty {
                Type::Scalar(Scalar::Float) => value.to_string(),
                Type::Scalar(_) => format!("f32({value})"),
                Type::Vector(Scalar::Float, _) => format!("{value}.x"),
                Type::Vector(..) => format!("f32({value}.x)"),
                _ => return,
            };
            let field = self.outputs.add(
                name,
                semantic.to_string(),
                None,
                Some("frag_depth"),
                "f32".to_string(),
                false,
            );
            self.stores
                .push(format!("{}.{field} = {read};", self.output_var));
            return;
        }
        let location = match upper.as_str() {
            "COLOR" | "SV_TARGET" => Some(index),
            _ => None,
        };
        let (stored, read) = match ty {
            Type::Scalar(Scalar::Bool) | Type::Vector(Scalar::Bool, _) => {
                let stored = ty.with_scalar(Scalar::Uint);
                let read = format!("{}({value})", p.ty(&stored));
                (stored, read)
            }
            Type::Scalar(_) | Type::Vector(..) => (ty.clone(), value.to_string()),
            _ => return,
        };
        let field = self.outputs.add(
            name,
            semantic.to_string(),
            location,
            None,
            p.ty(&stored),
            false,
        );
        self.stores
            .push(format!("{}.{field} = {read};", self.output_var));
    }
}

/// Prints the `@fragment` wrapper of the entry point, with its interface.
pub(super) fn wrapper(p: &Printer<'_>) -> (String, Vec<IoBinding>, Vec<IoBinding>) {
    let module = p.module;
    let func = &module.functions[module.entry];
    let mut body = Body::detached(p);
    let input_struct = body.fresh("FragmentInput");
    let output_struct = body.fresh("FragmentOutput");
    let input_var = body.fresh("input");
    let output_var = body.fresh("output");
    let mut lines = Vec::new();
    for (g, global) in module.globals.iter().enumerate() {
        if let Some(init) = &global.entry_init {
            let value = body.top(init);
            lines.extend(body.take_lets());
            lines.push(format!("{} = {value};", p.names.globals[g]));
        }
    }
    let mut w = Wrapper {
        p,
        body: &mut body,
        inputs: Interface::new(),
        outputs: Interface::new(),
        input_var,
        output_var,
        stores: Vec::new(),
    };
    let mut args = Vec::new();
    let mut after = Vec::new();
    for param in &func.params {
        let local = &func.locals[param.local];
        let semantic = param.semantic.as_deref();
        match param.dir {
            ParamDir::In => args.push(w.input(&local.name, &local.ty, semantic)),
            ParamDir::Out | ParamDir::InOut => {
                let var = w.body.fresh(&local.name);
                let ty = p.ty(&local.ty);
                if param.dir == ParamDir::InOut {
                    let value = w.input(&local.name, &local.ty, semantic);
                    lines.push(format!("var {var}: {ty} = {value};"));
                } else {
                    lines.push(format!("var {var}: {ty};"));
                }
                args.push(format!("&{var}"));
                after.push((
                    local.name.clone(),
                    local.ty.clone(),
                    param.semantic.clone(),
                    var,
                ));
            }
        }
    }
    let call = format!("{}({})", p.names.functions[module.entry], args.join(", "));
    if func.ret == Type::Void {
        lines.push(format!("{call};"));
    } else {
        let result = w.body.fresh("result");
        lines.push(format!("let {result} = {call};"));
        w.output("result", &func.ret, func.ret_semantic.as_deref(), &result);
    }
    for (name, ty, semantic, var) in &after {
        w.output(name, ty, semantic.as_deref(), var);
    }
    w.inputs.assign_locations();
    w.outputs.assign_locations();
    let mut out = String::new();
    let has_inputs = !w.inputs.slots.is_empty();
    let has_outputs = !w.outputs.slots.is_empty();
    if has_inputs {
        w.inputs.declare(&mut out, &input_struct);
    }
    if has_outputs {
        w.outputs.declare(&mut out, &output_struct);
    }
    let params = if has_inputs {
        format!("{}: {input_struct}", w.input_var)
    } else {
        String::new()
    };
    let ret = if has_outputs {
        format!(" -> {output_struct}")
    } else {
        String::new()
    };
    let _ = writeln!(out, "@fragment\nfn {}({params}){ret} {{", p.names.wrapper);
    for line in &lines {
        let _ = writeln!(out, "    {line}");
    }
    if has_outputs {
        let _ = writeln!(out, "    var {}: {output_struct};", w.output_var);
        for store in &w.stores {
            let _ = writeln!(out, "    {store}");
        }
        let _ = writeln!(out, "    return {};", w.output_var);
    }
    out.push_str("}\n");
    (out, w.inputs.reflect(), w.outputs.reflect())
}
