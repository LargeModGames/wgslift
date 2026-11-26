# Contributing to wgslift

Thanks for your interest in wgslift! You do not need to write code to help: bug reports with shaders that fail, docs and testing on your setup all count.

- [Ways to help](#ways-to-help)
- [Report a bug](#report-a-bug)
- [Set up](#set-up)
- [Where things live](#where-things-live)
- [Before you open a PR](#before-you-open-a-pr)
- [Test shaders and other translators](#test-shaders-and-other-translators)
- [After you open a PR](#after-you-open-a-pr)
- [Using AI Tools](#using-ai-tools)
- [Questions](#questions)

## Ways to help

- **Report a bug** or ask for a feature in the [issues](https://github.com/LargeModGames/wgslift/issues).
- **Send a shader that fails.** A small shader that shows the problem is the most useful bug report.
- **Test on your setup.** Try the output in your `wgpu` program or browser and tell us what breaks.
- **Write code.** Open an issue first for a new feature or a large refactor.

## Report a bug

Tell us the wgslift version and paste the error. Add the shader when you may share it, or a smaller shader that fails the same way.

## Set up

1. Install Rust with [rustup](https://rustup.rs). The minimum version is the `rust-version` in `Cargo.toml`.
2. Fork the repo, clone your fork, and make a branch from `main`.

## Where things live

| Path | What it is |
|---|---|
| `crates/wgslift-syntax/` | The front end: preprocessor, lexer, parser and syntax tree |
| `crates/wgslift-syntax/examples/milkdrop_corpus.rs` | Parses every shader in a folder of MilkDrop presets that you have on your computer |
| `.github/workflows/` | CI |

## Before you open a PR

Run the local gate. CI runs the same three commands:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Also run any cargo command after you edit a `Cargo.toml`, so `Cargo.lock` updates. CI builds with `--locked` and fails on a stale lock.

CI then runs clippy and the tests on Linux, Windows and macOS, and checks that the minimum Rust version still builds ([`ci.yml`](.github/workflows/ci.yml)).

Add or update tests when you change behavior, and add a line under `Unreleased` in [`CHANGELOG.md`](CHANGELOG.md) for a change users notice.

## Test shaders and other translators

- Commit only shaders that you wrote yourself or that have a licence that permits it, with that licence noted. Never commit a shader from a preset pack or a game without one.
- wgslift is its own implementation. Do not copy code from other shader translators (DXC, glslang, Slang, SPIRV-Cross, HLSLcc, hlslparser, vkd3d and others). Work from Microsoft's HLSL documentation, the WGSL specification and how shaders behave.

## After you open a PR

- **Title:** `type(scope): subject`, with a type in use here (`feat`, `fix`, `perf`, `refactor`, `docs`, `chore`, `ci`, `test`), for example `fix(parser): accept a trailing comma in an initializer`.
- **Description:** what changed and why, and the commands you ran under Testing.
- **`Fixes #N`:** only when the PR fixes the whole issue, because the issue closes on merge. For part of an issue, write `Part of #N`.
- A maintainer reviews and squash-merges the PR.

### Using AI Tools

AI tools are welcome. How you wrote the code matters less than whether you checked it. You are the author of your PR, whatever wrote the first draft:

- Read your whole diff before you open the PR, and be ready to explain any line of it.
- Run the checks yourself and paste their output under Testing. Do not claim a check you did not run.
- Change only what the issue is about. No unrelated rewrites or deletions.
- Write the description yourself: what changed and why. A copy of the issue text or generic filler tells a reviewer nothing.

PRs that ignore the issue, delete unrelated content, or claim checks that were never run are closed without a detailed review.

## Questions

Ask in [Discussions](https://github.com/LargeModGames/wgslift/discussions), or open an [issue](https://github.com/LargeModGames/wgslift/issues) for a bug. We are happy to help.
