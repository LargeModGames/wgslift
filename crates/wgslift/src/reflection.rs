//! What a host needs to run the translated module: bindings and interface.
//!
//! Every binding is in `@group(0)`. The uniform buffer, when there is one, is
//! binding 0; each texture and sampler pair takes the next two bindings.

use crate::types::SamplerDim;

/// The bindings and entry-point interface of a translated module.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reflection {
    /// The uniform buffer, when the shader has uniforms.
    pub uniform_buffer: Option<UniformBuffer>,
    /// The texture and sampler pairs, in binding order.
    pub textures: Vec<TextureBinding>,
    /// The entry-point inputs.
    pub inputs: Vec<IoBinding>,
    /// The entry-point outputs.
    pub outputs: Vec<IoBinding>,
}

/// The uniform buffer: one WGSL struct that holds every uniform.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformBuffer {
    /// The binding number in group 0.
    pub binding: u32,
    /// The size in bytes, a multiple of 16.
    pub size: u32,
    /// The members in offset order.
    pub members: Vec<UniformMember>,
}

/// One uniform in the uniform buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformMember {
    /// The HLSL name.
    pub name: String,
    /// The offset in bytes.
    pub offset: u32,
    /// The size in bytes.
    pub size: u32,
    /// The WGSL type, for example `vec4<f32>`.
    pub wgsl_type: String,
}

/// One HLSL sampler, which becomes a WGSL texture and a WGSL sampler.
#[derive(Debug, Clone, PartialEq)]
pub struct TextureBinding {
    /// The HLSL name of the sampler.
    pub name: String,
    /// The texture dimension.
    pub dim: SamplerDim,
    /// The binding number of the texture in group 0.
    pub texture_binding: u32,
    /// The binding number of the sampler in group 0.
    pub sampler_binding: u32,
}

/// One input or output of the entry point.
#[derive(Debug, Clone, PartialEq)]
pub struct IoBinding {
    /// The HLSL name of the parameter or field.
    pub name: String,
    /// The HLSL semantic, for example `TEXCOORD0` or `COLOR0`.
    pub semantic: String,
    /// The WGSL `@location`, for a user-defined value.
    pub location: Option<u32>,
    /// The WGSL `@builtin`, for example `position`.
    pub builtin: Option<String>,
    /// The WGSL type.
    pub wgsl_type: String,
}
