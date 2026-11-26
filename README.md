# wgslift

A translator from HLSL to WGSL, written in Rust. It needs no C++ toolchain.

wgslift lifts HLSL shaders, starting with the Direct3D 9 era (shader models 2 and 3),
into WGSL for `wgpu` and WebGPU.

## Status

Early development. Nothing is published yet.

The first milestone is the front end (`wgslift-syntax`): the preprocessor, the lexer and
the parser. Translation to WGSL comes next.

wgslift is not affiliated with Microsoft. HLSL is used here only to name the language
that wgslift reads.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
