//! Parses every warp and comp shader in a folder of MilkDrop presets and reports
//! the errors, grouped by message.
//!
//! ```text
//! cargo run --release --example milkdrop_corpus -- <preset folder> [examples per error]
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use wgslift_syntax::lexer::lex;
use wgslift_syntax::{FileId, Options, Punct, parse};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(root) = args.next() else {
        eprintln!("usage: milkdrop_corpus <preset folder> [examples per error]");
        std::process::exit(2);
    };
    let examples: usize = args.next().and_then(|n| n.parse().ok()).unwrap_or(1);
    let mut presets = Vec::new();
    collect(Path::new(&root), &mut presets);
    presets.sort();

    let started = Instant::now();
    let (mut shaders, mut failed) = (0usize, 0usize);
    let (mut max_parens, mut max_braces) = (0usize, 0usize);
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in &presets {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let text = String::from_utf8_lossy(&bytes);
        for (kind, shader) in shaders_of(&text) {
            shaders += 1;
            let source =
                until_last_brace(&shader).replace("shader_body", "void shader_body_main()");
            let (parens, braces) = nesting(&source);
            max_parens = max_parens.max(parens);
            max_braces = max_braces.max(braces);
            let name = format!("{} [{kind}]", path.display());
            let parsed = parse(&name, &source, Options::default());
            let Some(first) = parsed.diagnostics.first() else {
                continue;
            };
            failed += 1;
            let group = groups.entry(group_of(&first.message)).or_default();
            if group.len() < examples {
                let line = parsed
                    .files
                    .get(first.span.file)
                    .map(|f| {
                        let (line, _) = f.line_column(first.span.start);
                        f.text
                            .lines()
                            .nth(line - 1)
                            .unwrap_or("")
                            .trim()
                            .to_string()
                    })
                    .unwrap_or_default();
                group.push(format!("{}\n        {line}", first.display(&parsed.files)));
            } else {
                group.push(String::new());
            }
        }
    }

    let mut ranked: Vec<_> = groups.into_iter().collect();
    ranked.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
    for (message, hits) in &ranked {
        println!("{:>6}  {message}", hits.len());
        for example in hits.iter().filter(|e| !e.is_empty()) {
            println!("        {example}");
        }
    }
    println!(
        "{} presets, {shaders} shaders, {} parsed, {failed} failed ({:.2}% parsed) in {:.2?}",
        presets.len(),
        shaders - failed,
        100.0 * (shaders - failed) as f64 / shaders.max(1) as f64,
        started.elapsed()
    );
    println!("deepest nesting: {max_parens} parentheses, {max_braces} braces");
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("milk"))
        {
            out.push(path);
        }
    }
}

/// The `warp` and `comp` shaders of a preset: `warp_N=` and `comp_N=` lines in
/// order, each without its leading backtick.
fn shaders_of(text: &str) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    for kind in ["warp", "comp"] {
        let prefix = format!("{kind}_");
        let mut lines: Vec<(u32, &str)> = text
            .lines()
            .filter_map(|line| {
                let (key, value) = line.split_once('=')?;
                let n = key.trim().strip_prefix(&prefix)?.parse().ok()?;
                Some((n, value.strip_prefix('`').unwrap_or(value)))
            })
            .collect();
        if lines.is_empty() {
            continue;
        }
        lines.sort_by_key(|(n, _)| *n);
        let body: Vec<&str> = lines.into_iter().map(|(_, l)| l).collect();
        out.push((kind, body.join("\n")));
    }
    out
}

/// The shader up to its last `}`. Some presets that MilkDrop 2 runs have a
/// signature after the shader body, so MilkDrop ignores text after it.
fn until_last_brace(text: &str) -> &str {
    let mut ignored = Vec::new();
    lex(FileId(0), text, &mut ignored)
        .iter()
        .rev()
        .find(|t| t.is(Punct::RBrace))
        .map_or(text, |t| &text[..t.span.end as usize])
}

/// The deepest nesting of parentheses and of braces in the text.
fn nesting(text: &str) -> (usize, usize) {
    let mut ignored = Vec::new();
    let (mut parens, mut braces, mut max_parens, mut max_braces) = (0usize, 0usize, 0, 0);
    for t in lex(FileId(0), text, &mut ignored) {
        if t.is(Punct::LParen) {
            parens += 1;
            max_parens = max_parens.max(parens);
        } else if t.is(Punct::RParen) {
            parens = parens.saturating_sub(1);
        } else if t.is(Punct::LBrace) {
            braces += 1;
            max_braces = max_braces.max(braces);
        } else if t.is(Punct::RBrace) {
            braces = braces.saturating_sub(1);
        }
    }
    (max_parens, max_braces)
}

/// The message with the found token removed, so that similar errors group.
fn group_of(message: &str) -> String {
    match message.split_once(", found") {
        Some((head, _)) => format!("{head}, found ..."),
        None => message.to_string(),
    }
}
