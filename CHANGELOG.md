# Changelog

All notable changes to this project are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-27

### Added

- `wgslift-syntax`: an HLSL front end with source spans on every node.
  - A preprocessor: object-like and function-like macros with `#` and `##`,
    `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`, `#undef`, `#error`, and
    `#include` through an `IncludeResolver`. `#line` and `#pragma` are ignored.
  - `#if` follows C: `&&`, `||` and `?:` short-circuit, and a `u` suffix makes
    unsigned arithmetic.
  - A parser for functions, global and local declarations, structs, typedefs,
    `cbuffer`/`tbuffer` blocks, semantics, `register` (with a profile or a register
    space, one for each profile) and `packoffset`, annotations, `sampler_state` and
    D3D10 `SamplerState s { ... }` blocks, geometry, hull and domain shader
    parameters, attributes, all statements and all expressions.
  - Error recovery: after an error the parser goes on at the next statement, item,
    member, sampler state or `case`.
  - Limits for hostile input: nesting (`MAX_DEPTH`), expression tree depth
    (`MAX_TREE_DEPTH`), macro expansion work (`MAX_EXPANSION_WORK`), `#if` nesting,
    include depth (`MAX_INCLUDE_DEPTH`) and included files (`MAX_INCLUDES`).
- The `milkdrop_corpus` example parses every shader in a folder of MilkDrop presets.
- `wgslift`: a translator from Direct3D 9 pixel shaders (shader models 2 and 3) to
  WGSL.
  - `translate` with `Options`: the entry point, macros, a prelude, predeclared
    names, an `IncludeResolver`, and the `Dialect` of older compilers
    (`Dialect::d3d9_legacy`, the default: the absolute value in `pow`, `log` and
    `sqrt`, writable uniforms, `for` variables visible after the loop, and the
    `Uniformity` policy for derivatives in non-uniform control flow).
  - Type checking with the HLSL conversion rules, overload resolution, the HLSL
    intrinsics, `out` and `inout` arguments (swizzles included), static locals set
    once when the shader starts, and constant folding of values that leave the
    finite range.
  - `Output` with the WGSL text and a `Reflection`: the WGSL entry-point name, the
    uniform buffer layout (offsets, array strides and struct fields), the texture
    and sampler bindings, and the entry inputs and outputs. `Failed` holds the
    diagnostics with their source spans.
  - `Options::loop_limit`: a budget of loop passes per invocation, summed over all
    loops, for untrusted shaders.
  - Limits for hostile input: array and struct sizes, constant integer overflow,
    and deep expressions, which are split into `let` and `const` temporaries.
- The `milkdrop_translate` example translates every shader in a folder of MilkDrop
  presets and checks the WGSL with naga and, with `--tint`, with Tint.

[Unreleased]: https://github.com/LargeModGames/wgslift/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/LargeModGames/wgslift/releases/tag/v0.1.0
