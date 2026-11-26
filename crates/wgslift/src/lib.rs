//! wgslift translates Direct3D 9 era HLSL shaders to WGSL.
//!
//! The front end is the `wgslift-syntax` crate. This crate checks the types, lowers
//! the program and prints WGSL.

pub mod types;
