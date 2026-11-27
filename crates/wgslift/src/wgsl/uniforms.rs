//! The uniform buffer: how each uniform is stored and read back.
//!
//! A uniform keeps its IR type when that type is host-shareable with a valid
//! uniform layout. A bool travels as a `u32`, an array whose stride is not a
//! multiple of 16 wraps its elements in a padded struct, and a struct that needs
//! either gets a mirror struct. A read converts the stored value back.

use std::collections::HashMap;
use std::fmt::Write;

use crate::ir::Module;
use crate::reflection::{UniformBuffer, UniformMember};
use crate::types::{Scalar, Type, round_up};

use super::names::Namer;

/// How one value is stored in the uniform buffer.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Repr {
    /// A scalar, vector or matrix stored as it is.
    Value(Type),
    /// A bool or bool vector of this many lanes, stored as `u32` lanes.
    Bool(u8),
    /// An array; `wrap` names the padded struct around each element.
    Array {
        /// The element.
        elem: Box<Repr>,
        /// The length.
        len: u32,
        /// The padded element struct, when the stride needs it.
        wrap: Option<String>,
    },
    /// A struct, by its index in the module.
    Struct(usize),
}

/// The stored form of one struct.
#[derive(Debug, Clone)]
struct StructRepr {
    /// The mirror struct, when the struct cannot be stored as it is.
    mirror: Option<String>,
    /// The fields.
    fields: Vec<Repr>,
    /// The offset and size of each field.
    places: Vec<(u32, u32)>,
    /// Size and alignment.
    layout: (u32, u32),
}

/// One member of a struct in the uniform buffer, with its layout attributes.
struct Member {
    repr: Repr,
    align_attr: Option<u32>,
    size_attr: Option<u32>,
    offset: u32,
    size: u32,
}

/// The uniform buffer of a module.
#[derive(Debug, Default)]
pub(super) struct Uniforms {
    /// The WGSL declarations: helper structs, the buffer struct and its variable.
    pub decls: String,
    /// The representation of each uniform global.
    pub globals: HashMap<usize, Repr>,
    /// The reflection, when there are uniforms.
    pub reflection: Option<UniformBuffer>,
    structs: HashMap<usize, StructRepr>,
    wrappers: HashMap<String, String>,
}

/// What the uniform planner needs to know about names.
pub(super) struct NameTable<'a> {
    /// WGSL struct names by index.
    pub structs: &'a [String],
    /// WGSL field names by struct index.
    pub fields: &'a [Vec<String>],
}

impl NameTable<'_> {
    /// The WGSL spelling of a type.
    pub(super) fn ty(&self, ty: &Type) -> String {
        ty.wgsl(&|i| self.structs[i].clone()).to_string()
    }
}

impl Uniforms {
    /// Plans the buffer for the uniform globals `members` (index and WGSL name).
    pub(super) fn plan(
        module: &Module,
        members: &[(usize, String)],
        names: &NameTable<'_>,
        namer: &mut Namer,
        struct_name: &str,
        var_name: &str,
    ) -> Uniforms {
        let mut plan = Uniforms::default();
        if members.is_empty() {
            return plan;
        }
        let reprs: Vec<Repr> = members
            .iter()
            .map(|(g, _)| plan.repr(module, &module.globals[*g].ty, names, namer))
            .collect();
        let laid = plan.lay_out(reprs);
        let mut text = format!("struct {struct_name} {{\n");
        let mut reflected = Vec::new();
        for ((g, name), m) in members.iter().zip(&laid.0) {
            plan.member_line(&mut text, name, m, names);
            reflected.push(plan.reflect(
                module,
                &module.globals[*g].name,
                (m.offset, m.size),
                &m.repr,
                names,
            ));
            plan.globals.insert(*g, m.repr.clone());
        }
        text.push_str("}\n\n");
        let _ = writeln!(
            text,
            "@group(0) @binding(0) var<uniform> {var_name}: {struct_name};\n"
        );
        plan.decls.push_str(&text);
        plan.reflection = Some(UniformBuffer {
            binding: 0,
            size: round_up(laid.1.0.max(1), 16),
            members: reflected,
        });
        plan
    }

    fn member_line(&self, text: &mut String, name: &str, m: &Member, names: &NameTable<'_>) {
        text.push_str("    ");
        if let Some(a) = m.align_attr {
            let _ = write!(text, "@align({a}) ");
        }
        if let Some(s) = m.size_attr {
            let _ = write!(text, "@size({s}) ");
        }
        let _ = writeln!(text, "{name}: {},", self.stored(&m.repr, names));
    }

    fn repr(
        &mut self,
        module: &Module,
        ty: &Type,
        names: &NameTable<'_>,
        namer: &mut Namer,
    ) -> Repr {
        match ty {
            Type::Scalar(Scalar::Bool) => Repr::Bool(1),
            Type::Vector(Scalar::Bool, n) => Repr::Bool(*n),
            Type::Array(elem, len) => {
                let elem = self.repr(module, elem, names, namer);
                let (size, align) = self.layout(&elem);
                let wrap = (!round_up(size, align).is_multiple_of(16)).then(|| {
                    let stored = self.stored(&elem, names);
                    if let Some(name) = self.wrappers.get(&stored) {
                        return name.clone();
                    }
                    let name = namer.name("UniformElement");
                    let _ = write!(
                        self.decls,
                        "struct {name} {{\n    @size({}) v: {stored},\n}}\n\n",
                        round_up(size, 16)
                    );
                    self.wrappers.insert(stored, name.clone());
                    name
                });
                Repr::Array {
                    elem: Box::new(elem),
                    len: *len,
                    wrap,
                }
            }
            Type::Struct(index) => {
                if !self.structs.contains_key(index) {
                    let fields: Vec<Repr> = module.structs[*index]
                        .fields
                        .iter()
                        .map(|f| self.repr(module, &f.ty, names, namer))
                        .collect();
                    let (members, layout) = self.lay_out(fields);
                    let plain = members.iter().all(|m| {
                        m.align_attr.is_none() && m.size_attr.is_none() && !self.converts(&m.repr)
                    });
                    let mirror = (!plain).then(|| {
                        let name = namer.name(&format!("{}_uniform", names.structs[*index]));
                        let mut text = format!("struct {name} {{\n");
                        for (field, m) in names.fields[*index].iter().zip(&members) {
                            self.member_line(&mut text, field, m, names);
                        }
                        text.push_str("}\n\n");
                        self.decls.push_str(&text);
                        name
                    });
                    let places = members.iter().map(|m| (m.offset, m.size)).collect();
                    let fields = members.into_iter().map(|m| m.repr).collect();
                    self.structs.insert(
                        *index,
                        StructRepr {
                            mirror,
                            fields,
                            places,
                            layout,
                        },
                    );
                }
                Repr::Struct(*index)
            }
            other => Repr::Value(other.clone()),
        }
    }

    /// Lays out struct members, adding the attributes the uniform rules need.
    /// Returns the members and the struct's size and alignment.
    fn lay_out(&self, reprs: Vec<Repr>) -> (Vec<Member>, (u32, u32)) {
        let mut offset = 0;
        let mut max_align = 4;
        let mut members = Vec::new();
        for repr in reprs {
            let (size, natural) = self.layout(&repr);
            let aggregate = matches!(repr, Repr::Array { .. } | Repr::Struct(_));
            let align_attr = (aggregate && natural < 16).then_some(16);
            let size_attr = (aggregate && !size.is_multiple_of(16)).then(|| round_up(size, 16));
            let align = align_attr.unwrap_or(natural);
            let stored_size = size_attr.unwrap_or(size);
            offset = round_up(offset, align);
            members.push(Member {
                repr,
                align_attr,
                size_attr,
                offset,
                size: stored_size,
            });
            offset += stored_size;
            max_align = max_align.max(align);
        }
        (members, (round_up(offset, max_align), max_align))
    }

    /// Size and natural alignment of a stored value.
    pub(super) fn layout(&self, repr: &Repr) -> (u32, u32) {
        match repr {
            Repr::Value(ty) => ty.uniform_layout(&|_| None).unwrap_or((4, 4)),
            Repr::Bool(n) => Type::vector(Scalar::Uint, *n)
                .uniform_layout(&|_| None)
                .unwrap_or((4, 4)),
            Repr::Array { elem, len, wrap } => {
                let (_, align) = self.layout(elem);
                (self.stride(elem, wrap.is_some()) * len, align)
            }
            Repr::Struct(index) => self.structs[index].layout,
        }
    }

    /// The bytes from one array element to the next.
    fn stride(&self, elem: &Repr, wrapped: bool) -> u32 {
        let (size, align) = self.layout(elem);
        if wrapped {
            round_up(size, 16)
        } else {
            round_up(size, align)
        }
    }

    /// The reflection of a stored value at `place` (offset and size).
    fn reflect(
        &self,
        module: &Module,
        name: &str,
        place: (u32, u32),
        repr: &Repr,
        names: &NameTable<'_>,
    ) -> UniformMember {
        let (stride, members) = match repr {
            Repr::Array { elem, wrap, .. } => {
                let size = self.layout(elem).0;
                let element = self.reflect(module, "", (0, size), elem, names);
                (Some(self.stride(elem, wrap.is_some())), vec![element])
            }
            Repr::Struct(index) => {
                let stored = &self.structs[index];
                let fields = module.structs[*index].fields.iter();
                let members = fields
                    .zip(&stored.fields)
                    .zip(&stored.places)
                    .map(|((f, r), place)| self.reflect(module, &f.name, *place, r, names))
                    .collect();
                (None, members)
            }
            Repr::Value(_) | Repr::Bool(_) => (None, Vec::new()),
        };
        UniformMember {
            name: name.to_string(),
            offset: place.0,
            size: place.1,
            wgsl_type: self.stored(repr, names),
            stride,
            members,
        }
    }

    /// True when a read must convert the stored value.
    pub(super) fn converts(&self, repr: &Repr) -> bool {
        match repr {
            Repr::Value(_) => false,
            Repr::Bool(_) => true,
            Repr::Array { elem, wrap, .. } => wrap.is_some() || self.converts(elem),
            Repr::Struct(index) => self.structs[index].mirror.is_some(),
        }
    }

    /// The stored WGSL type.
    pub(super) fn stored(&self, repr: &Repr, names: &NameTable<'_>) -> String {
        match repr {
            Repr::Value(ty) => names.ty(ty),
            Repr::Bool(1) => "u32".to_string(),
            Repr::Bool(n) => format!("vec{n}<u32>"),
            Repr::Array { elem, len, wrap } => match wrap {
                Some(w) => format!("array<{w}, {len}>"),
                None => format!("array<{}, {len}>", self.stored(elem, names)),
            },
            Repr::Struct(index) => self.structs[index]
                .mirror
                .clone()
                .unwrap_or_else(|| names.structs[*index].clone()),
        }
    }

    /// The representation of field `field` of a stored struct.
    pub(super) fn field(&self, index: usize, field: usize) -> Repr {
        self.structs[&index].fields[field].clone()
    }

    /// An expression that reads the stored value `text` as its IR type, and
    /// whether it is a primary expression.
    pub(super) fn read(&self, repr: &Repr, text: &str, names: &NameTable<'_>) -> (String, bool) {
        match repr {
            Repr::Value(_) => (text.to_string(), true),
            Repr::Bool(1) => (format!("({text} != 0u)"), true),
            Repr::Bool(n) => (format!("({text} != vec{n}<u32>(0u))"), true),
            Repr::Array { elem, len, wrap } => {
                if !self.converts(repr) {
                    return (text.to_string(), true);
                }
                let suffix = if wrap.is_some() { ".v" } else { "" };
                let parts: Vec<String> = (0..*len)
                    .map(|i| self.read(elem, &format!("{text}[{i}]{suffix}"), names).0)
                    .collect();
                let ty = self.value_type(repr, names);
                (format!("{ty}({})", parts.join(", ")), true)
            }
            Repr::Struct(index) => {
                if !self.converts(repr) {
                    return (text.to_string(), true);
                }
                let parts: Vec<String> = names.fields[*index]
                    .iter()
                    .zip(&self.structs[index].fields)
                    .map(|(f, r)| self.read(r, &format!("{text}.{f}"), names).0)
                    .collect();
                (
                    format!("{}({})", names.structs[*index], parts.join(", ")),
                    true,
                )
            }
        }
    }

    /// The WGSL spelling of the IR type that `repr` stores.
    fn value_type(&self, repr: &Repr, names: &NameTable<'_>) -> String {
        match repr {
            Repr::Value(ty) => names.ty(ty),
            Repr::Bool(n) => names.ty(&Type::vector(Scalar::Bool, *n)),
            Repr::Array { elem, len, .. } => {
                format!("array<{}, {len}>", self.value_type(elem, names))
            }
            Repr::Struct(index) => names.structs[*index].clone(),
        }
    }
}
