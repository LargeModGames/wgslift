//! The types of translated programs: HLSL names in, WGSL spelling and layout out.

use std::fmt;

/// A scalar type. HLSL `half`, `double` and the `min16` types become `Float`:
/// WGSL has 32-bit floats only, without extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scalar {
    /// `bool`
    Bool,
    /// `int`
    Int,
    /// `uint` or `dword`
    Uint,
    /// `float`, `half` or `double`
    Float,
}

impl Scalar {
    /// The WGSL name.
    pub fn wgsl(self) -> &'static str {
        match self {
            Scalar::Bool => "bool",
            Scalar::Int => "i32",
            Scalar::Uint => "u32",
            Scalar::Float => "f32",
        }
    }
}

/// The dimension of a sampler. A plain `sampler` gets its dimension from its use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplerDim {
    /// `sampler1D`
    D1,
    /// `sampler2D`
    D2,
    /// `sampler3D`
    D3,
    /// `samplerCUBE`
    Cube,
    /// `sampler`: not known until a `tex*` call uses it.
    Any,
}

/// A type. Vectors have 2 to 4 components: HLSL `float1` is the scalar `float`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// `void`
    Void,
    /// A scalar.
    Scalar(Scalar),
    /// A vector of 2, 3 or 4 components.
    Vector(Scalar, u8),
    /// An HLSL `floatRxC`: `rows` rows of `cols` components. WGSL stores each HLSL
    /// row as one column, so it is the WGSL type `matRxC`.
    Matrix {
        /// The number of HLSL rows, 2 to 4.
        rows: u8,
        /// The number of HLSL columns, 2 to 4.
        cols: u8,
    },
    /// A fixed-size array.
    Array(Box<Type>, u32),
    /// A struct, by its index in the module.
    Struct(usize),
    /// A texture and sampler pair.
    Sampler(SamplerDim),
}

/// Where an HLSL scalar type name came from, when it is not exact in WGSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// The type is exact.
    Exact,
    /// `half` or a `min16` type, which is 32-bit in WGSL. This is safe.
    Raised,
    /// `double`, which is 32-bit in WGSL. Precision is lost.
    Lowered,
}

impl Type {
    /// `float`
    pub const FLOAT: Type = Type::Scalar(Scalar::Float);
    /// `int`
    pub const INT: Type = Type::Scalar(Scalar::Int);
    /// `bool`
    pub const BOOL: Type = Type::Scalar(Scalar::Bool);

    /// A vector, or the scalar for one component.
    pub fn vector(scalar: Scalar, size: u8) -> Type {
        if size <= 1 {
            Type::Scalar(scalar)
        } else {
            Type::Vector(scalar, size)
        }
    }

    /// The type of an HLSL built-in type name such as `float3`, `half2x2` or
    /// `sampler2D`. Struct names and `vector<...>` are not handled here.
    pub fn from_hlsl_name(name: &str) -> Option<(Type, Precision)> {
        let sampler = match name {
            "void" => return Some((Type::Void, Precision::Exact)),
            "sampler" | "texture" => Some(SamplerDim::Any),
            "sampler1D" | "texture1D" => Some(SamplerDim::D1),
            "sampler2D" | "texture2D" => Some(SamplerDim::D2),
            "sampler3D" | "texture3D" => Some(SamplerDim::D3),
            "samplerCUBE" | "textureCUBE" => Some(SamplerDim::Cube),
            _ => None,
        };
        if let Some(dim) = sampler {
            return Some((Type::Sampler(dim), Precision::Exact));
        }
        const SCALARS: [(&str, Scalar, Precision); 11] = [
            ("bool", Scalar::Bool, Precision::Exact),
            ("int", Scalar::Int, Precision::Exact),
            ("uint", Scalar::Uint, Precision::Exact),
            ("dword", Scalar::Uint, Precision::Exact),
            ("float", Scalar::Float, Precision::Exact),
            ("half", Scalar::Float, Precision::Raised),
            ("double", Scalar::Float, Precision::Lowered),
            ("min16float", Scalar::Float, Precision::Raised),
            ("min10float", Scalar::Float, Precision::Raised),
            ("min16int", Scalar::Int, Precision::Raised),
            ("min16uint", Scalar::Uint, Precision::Raised),
        ];
        let dim = |c: u8| (b'1'..=b'4').contains(&c).then_some(c - b'0');
        SCALARS.iter().find_map(|&(prefix, scalar, precision)| {
            let rest = name.strip_prefix(prefix)?;
            let ty = match rest.as_bytes() {
                [] => Type::Scalar(scalar),
                [n] => Type::vector(scalar, dim(*n)?),
                [r, b'x', c] => {
                    let (rows, cols) = (dim(*r)?, dim(*c)?);
                    match (rows, cols, scalar) {
                        // A one-row or one-column matrix is a vector in practice.
                        (1, n, _) | (n, 1, _) => Type::vector(scalar, n),
                        (_, _, Scalar::Float) => Type::Matrix { rows, cols },
                        _ => return None,
                    }
                }
                _ => return None,
            };
            Some((ty, precision))
        })
    }

    /// The scalar type of the components, for scalars, vectors and matrices.
    pub fn scalar(&self) -> Option<Scalar> {
        match self {
            Type::Scalar(s) | Type::Vector(s, _) => Some(*s),
            Type::Matrix { .. } => Some(Scalar::Float),
            _ => None,
        }
    }

    /// The number of components of a scalar (1) or a vector.
    pub fn size(&self) -> Option<u8> {
        match self {
            Type::Scalar(_) => Some(1),
            Type::Vector(_, n) => Some(*n),
            _ => None,
        }
    }

    /// The same shape with another component type. A matrix stays float.
    pub fn with_scalar(&self, scalar: Scalar) -> Type {
        match self {
            Type::Scalar(_) => Type::Scalar(scalar),
            Type::Vector(_, n) => Type::Vector(scalar, *n),
            other => other.clone(),
        }
    }

    /// A scalar, vector or matrix.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Type::Scalar(_) | Type::Vector(..) | Type::Matrix { .. }
        )
    }

    /// The type in WGSL. Structs need their names, so they come from `struct_name`.
    pub fn wgsl<'a>(&'a self, struct_name: &'a dyn Fn(usize) -> String) -> impl fmt::Display + 'a {
        WgslType {
            ty: self,
            struct_name,
        }
    }

    /// Size and alignment in the WGSL uniform address space, in bytes.
    /// `None` for a type that cannot be in a uniform buffer.
    pub fn uniform_layout(
        &self,
        struct_layout: &dyn Fn(usize) -> Option<(u32, u32)>,
    ) -> Option<(u32, u32)> {
        match self {
            // A bool is not host-shareable; it travels as a u32.
            Type::Scalar(_) => Some((4, 4)),
            Type::Vector(_, 2) => Some((8, 8)),
            Type::Vector(_, 3) => Some((12, 16)),
            Type::Vector(_, _) => Some((16, 16)),
            Type::Matrix { rows, cols } => {
                let (_, column_align) =
                    Type::Vector(Scalar::Float, *cols).uniform_layout(struct_layout)?;
                Some((u32::from(*rows) * column_align, column_align))
            }
            Type::Array(element, len) => {
                let (size, align) = element.uniform_layout(struct_layout)?;
                // Uniform arrays need a stride that is a multiple of 16.
                let stride = round_up(size, align.max(16));
                (stride == round_up(size, align)).then_some((stride * len, align.max(16)))
            }
            Type::Struct(index) => struct_layout(*index),
            Type::Void | Type::Sampler(_) => None,
        }
    }
}

/// `value` rounded up to a multiple of `align`.
pub fn round_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

struct WgslType<'a> {
    ty: &'a Type,
    struct_name: &'a dyn Fn(usize) -> String,
}

impl fmt::Display for WgslType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ty {
            Type::Void => f.write_str("void"),
            Type::Scalar(s) => f.write_str(s.wgsl()),
            Type::Vector(s, n) => write!(f, "vec{n}<{}>", s.wgsl()),
            Type::Matrix { rows, cols } => write!(f, "mat{rows}x{cols}<f32>"),
            Type::Array(element, len) => {
                write!(f, "array<{}, {len}>", element.wgsl(self.struct_name))
            }
            Type::Struct(index) => f.write_str(&(self.struct_name)(*index)),
            Type::Sampler(_) => f.write_str("sampler"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(text: &str) -> Option<Type> {
        Type::from_hlsl_name(text).map(|(t, _)| t)
    }

    #[test]
    fn hlsl_names_map_to_types() {
        assert_eq!(name("float"), Some(Type::FLOAT));
        assert_eq!(name("float1"), Some(Type::FLOAT));
        assert_eq!(name("half3"), Some(Type::Vector(Scalar::Float, 3)));
        assert_eq!(name("int2"), Some(Type::Vector(Scalar::Int, 2)));
        assert_eq!(name("dword"), Some(Type::Scalar(Scalar::Uint)));
        assert_eq!(name("bool3"), Some(Type::Vector(Scalar::Bool, 3)));
        assert_eq!(name("float2x3"), Some(Type::Matrix { rows: 2, cols: 3 }));
        assert_eq!(name("float1x3"), Some(Type::Vector(Scalar::Float, 3)));
        assert_eq!(name("sampler"), Some(Type::Sampler(SamplerDim::Any)));
        assert_eq!(name("samplerCUBE"), Some(Type::Sampler(SamplerDim::Cube)));
        for bad in ["float5", "float2x", "int2x2", "floaty", "Light"] {
            assert_eq!(name(bad), None, "{bad}");
        }
        assert_eq!(
            Type::from_hlsl_name("double3").unwrap().1,
            Precision::Lowered
        );
        assert_eq!(Type::from_hlsl_name("half").unwrap().1, Precision::Raised);
    }

    #[test]
    fn wgsl_spelling() {
        let none = |_: usize| String::new();
        let show = |t: Type| t.wgsl(&none).to_string();
        assert_eq!(show(Type::FLOAT), "f32");
        assert_eq!(show(Type::Vector(Scalar::Int, 3)), "vec3<i32>");
        assert_eq!(show(Type::Matrix { rows: 2, cols: 3 }), "mat2x3<f32>");
        assert_eq!(
            show(Type::Array(Box::new(Type::Vector(Scalar::Float, 2)), 4)),
            "array<vec2<f32>, 4>"
        );
    }

    #[test]
    fn uniform_layout_follows_wgsl() {
        let none = |_: usize| None;
        let layout = |t: Type| t.uniform_layout(&none);
        assert_eq!(layout(Type::FLOAT), Some((4, 4)));
        assert_eq!(layout(Type::Vector(Scalar::Float, 3)), Some((12, 16)));
        assert_eq!(layout(Type::Matrix { rows: 3, cols: 3 }), Some((48, 16)));
        assert_eq!(layout(Type::Matrix { rows: 2, cols: 2 }), Some((16, 8)));
        assert_eq!(
            layout(Type::Array(Box::new(Type::Vector(Scalar::Float, 4)), 3)),
            Some((48, 16))
        );
        // An array of floats would have a 4-byte stride, which uniform buffers refuse.
        assert_eq!(layout(Type::Array(Box::new(Type::FLOAT), 4)), None);
        assert_eq!(layout(Type::Sampler(SamplerDim::D2)), None);
    }
}
