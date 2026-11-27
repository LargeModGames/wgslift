# Roadmap

The goal is to translate any HLSL shader that WebGPU can run.

High-level requirements yet to be implemented:

- [x] Direct3D 9 pixel shaders
- [ ] Direct3D 9 vertex shaders
- [ ] `.fx` Effects files
- [ ] A command-line tool
- [ ] Shader model 4 and 5: cbuffers, texture objects, structured buffers
- [ ] Compute shaders
- [ ] Shader model 6: HLSL 2021 templates, wave intrinsics, 16-bit floats
- [ ] Bevy asset loader
- [ ] Browser playground

This table shows what WebGPU can run, what is implemented already, and why.

| Feature | Implemented yet? | Notes |
| --- | --- | --- |
| Pixel shaders | Shader model 2 and 3 | 15,892 of 15,895 MilkDrop shaders translate |
| Vertex shaders | No | |
| Compute shaders | No | |
| Geometry, hull and domain shaders | No | WebGPU has no such stages yet |
| Mesh shaders and ray tracing | No | WebGPU has neither yet |
| 64-bit integers and doubles | No | WGSL has neither yet |
