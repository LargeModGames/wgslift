# wgslift

A translator from HLSL to WGSL, written in Rust. It needs no C++ toolchain.

wgslift lifts HLSL shaders, starting with the Direct3D 9 era (shader models 2 and 3),
into WGSL for `wgpu` and WebGPU.

## Status

Early development: the API can change between releases.

- `wgslift-syntax`: the preprocessor, the lexer and the parser.
- `wgslift`: type checking, the HLSL intrinsics, and a WGSL writer with a reflection
  of the uniform buffer, the textures and the entry point.

Direct3D 9 pixel shaders translate today. Of the 15,895 shaders in a large MilkDrop
preset collection, 15,892 translate to WGSL that both naga and Tint accept. The 3
that fail are not valid HLSL. Vertex shaders and later shader models come next.

```toml
wgslift = "0.1"
```

```rust
let options = wgslift::Options {
    entry_point: "main".to_string(),
    ..Default::default()
};
let source = "sampler2D tex;\nfloat4 main(float2 uv : TEXCOORD0) : COLOR0 { return tex2D(tex, uv); }";
match wgslift::translate("main.hlsl", source, options) {
    Ok(output) => println!("{}", output.wgsl),
    Err(failed) => {
        for diagnostic in &failed.diagnostics {
            eprintln!("{}", diagnostic.display(&failed.files));
        }
    }
}
```

`Dialect::d3d9_legacy()` (the default) follows the Direct3D 9 compiler: for example,
`pow`, `log` and `sqrt` use the absolute value of their input, 0 times infinity is 0,
and a shader may write to a uniform.

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
