# Changelog

All notable changes to this project are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
