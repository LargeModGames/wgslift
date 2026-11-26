//! Translates every warp and comp shader in a folder of MilkDrop presets, checks
//! the WGSL with naga and optionally with Tint, and reports the failures grouped by
//! message.
//!
//! ```text
//! cargo run --release -p wgslift --example milkdrop_translate -- <preset folder>
//!     [--tint <path to tint.exe>] [--show N] [--dump <dir>]
//! ```

mod milkdrop;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use milkdrop::{ENTRY_POINT, PRELUDE, predeclared, shaders_of, until_last_brace, wrap};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use wgslift::syntax::Severity;
use wgslift::{Options, translate};

const USAGE: &str =
    "usage: milkdrop_translate <preset folder> [--tint <path>] [--show N] [--dump <dir>]";

struct Args {
    root: PathBuf,
    tint: Option<PathBuf>,
    show: usize,
    dump: Option<PathBuf>,
}

fn main() {
    let Some(args) = parse_args(std::env::args().skip(1)) else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    // naga's parser and the translator recurse; deep shaders need a big stack.
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(move || run(&args))
        .expect("spawn the worker thread")
        .join()
        .expect("the worker thread panicked");
}

fn parse_args(mut it: impl Iterator<Item = String>) -> Option<Args> {
    let mut args = Args {
        root: PathBuf::new(),
        tint: None,
        show: 3,
        dump: None,
    };
    let mut root = None;
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--tint" => args.tint = Some(it.next()?.into()),
            "--show" => args.show = it.next()?.parse().ok()?,
            "--dump" => args.dump = Some(it.next()?.into()),
            _ if root.is_none() && !arg.starts_with("--") => root = Some(arg),
            _ => return None,
        }
    }
    args.root = root?.into();
    Some(args)
}

/// Failures with the same message, and up to `--show` examples of them.
#[derive(Default)]
struct Groups {
    groups: BTreeMap<String, (usize, Vec<String>)>,
}

impl Groups {
    fn add(&mut self, key: String, show: usize, example: impl FnOnce() -> String) {
        let (count, examples) = self.groups.entry(key).or_default();
        *count += 1;
        if examples.len() < show {
            examples.push(example());
        }
    }

    fn print(&self, title: &str, top: usize) {
        if self.groups.is_empty() {
            return;
        }
        let mut ranked: Vec<_> = self.groups.iter().collect();
        ranked.sort_by_key(|(_, (count, _))| std::cmp::Reverse(*count));
        println!("== {title}: {} groups", ranked.len());
        for (message, (count, examples)) in ranked.into_iter().take(top) {
            println!("{count:>6}  {message}");
            for example in examples {
                for line in example.lines() {
                    println!("          {line}");
                }
            }
        }
    }
}

fn run(args: &Args) {
    let started = Instant::now();
    let mut presets = Vec::new();
    collect(&args.root, &mut presets);
    presets.sort();
    let predeclare = predeclared();
    if let Some(dir) = &args.dump {
        std::fs::create_dir_all(dir).expect("create the dump folder");
    }
    let tint_file =
        std::env::temp_dir().join(format!("wgslift-corpus-{}.wgsl", std::process::id()));

    let (mut shaders, mut translated, mut naga_valid, mut tint_valid) = (0, 0, 0, 0);
    let (mut translate_errors, mut naga_errors, mut tint_errors) =
        (Groups::default(), Groups::default(), Groups::default());
    for path in &presets {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let text = String::from_utf8_lossy(&bytes);
        let relative = path.strip_prefix(&args.root).unwrap_or(path);
        for (kind, shader) in shaders_of(&text) {
            shaders += 1;
            let name = format!("{} [{}]", relative.display(), kind.key());
            let source = match wrap(kind, until_last_brace(&shader)) {
                Ok(source) => source,
                Err(message) => {
                    translate_errors.add(message.to_string(), args.show, || name.clone());
                    continue;
                }
            };
            let options = Options {
                entry_point: ENTRY_POINT.to_string(),
                prelude: Some(PRELUDE.to_string()),
                predeclare: predeclare.clone(),
                ..Default::default()
            };
            let wgsl = match translate(&name, &source, options) {
                Ok(output) => output.wgsl,
                Err(failed) => {
                    let first = failed
                        .diagnostics
                        .iter()
                        .find(|d| d.severity == Severity::Error)
                        .or(failed.diagnostics.first());
                    let key = first
                        .map_or("translation failed without a diagnostic".to_string(), |d| {
                            group_of(&d.message)
                        });
                    translate_errors.add(key, args.show, || {
                        first.map_or(name.clone(), |d| d.display(&failed.files).to_string())
                    });
                    continue;
                }
            };
            translated += 1;
            if let Some(dir) = &args.dump {
                let file = dir.join(format!("{}.{}.wgsl", flat_name(relative), kind.key()));
                if let Err(e) = std::fs::write(&file, &wgsl) {
                    eprintln!("cannot write {}: {e}", file.display());
                }
            }
            match validate_with_naga(&wgsl) {
                Ok(()) => naga_valid += 1,
                Err(report) => naga_errors.add(first_line(&report), args.show, || {
                    format!("{name}\n{report}")
                }),
            }
            if let Some(tint) = &args.tint {
                match validate_with_tint(tint, &tint_file, &wgsl) {
                    Ok(()) => tint_valid += 1,
                    Err(report) => tint_errors.add(tint_group(&report), args.show, || {
                        format!("{name}\n{report}")
                    }),
                }
            }
        }
    }
    let _ = std::fs::remove_file(&tint_file);

    const TOP: usize = 15;
    translate_errors.print("translation errors", TOP);
    naga_errors.print("naga errors", TOP);
    tint_errors.print("tint errors", TOP);
    let tint = if args.tint.is_some() {
        tint_valid.to_string()
    } else {
        "-".to_string()
    };
    println!(
        "{} presets, {shaders} shaders, {translated} translated, {naga_valid} naga-valid, \
         {tint} tint-valid in {:.2?}",
        presets.len(),
        started.elapsed()
    );
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

/// Parses and validates the module the way wgpu would with default features.
fn validate_with_naga(wgsl: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| e.emit_to_string(wgsl))?;
    Validator::new(ValidationFlags::all(), Capabilities::default())
        .validate(&module)
        .map_err(|e| e.emit_to_string(wgsl))?;
    Ok(())
}

/// Runs `tint --format wgsl -o NUL <file>` and fails on a non-zero exit or an
/// `error:` line.
fn validate_with_tint(tint: &Path, file: &Path, wgsl: &str) -> Result<(), String> {
    std::fs::write(file, wgsl).map_err(|e| format!("cannot write {}: {e}", file.display()))?;
    let null = if cfg!(windows) { "NUL" } else { "/dev/null" };
    let output = Command::new(tint)
        .args(["--format", "wgsl", "-o", null])
        .arg(file)
        .output()
        .map_err(|e| format!("cannot run {}: {e}", tint.display()))?;
    let report = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    if output.status.success() && !report.contains("error:") {
        Ok(())
    } else {
        Err(report.replace(&file.display().to_string(), "<shader>"))
    }
}

/// The message with the found token removed, so that similar errors group.
fn group_of(message: &str) -> String {
    match message.split_once(", found") {
        Some((head, _)) => format!("{head}, found ..."),
        None => message.to_string(),
    }
}

fn first_line(report: &str) -> String {
    report
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("empty report")
        .trim()
        .to_string()
}

/// The first line of a Tint report from `error:` on, without the file and line.
fn tint_group(report: &str) -> String {
    let line = report
        .lines()
        .find(|l| l.contains("error:"))
        .unwrap_or_else(|| report.lines().next().unwrap_or("empty report"));
    line.find("error:")
        .map_or(line, |at| &line[at..])
        .trim()
        .to_string()
}

/// A file name for a preset path: the folders joined with `__`.
fn flat_name(relative: &Path) -> String {
    relative
        .with_extension("")
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("__")
        .replace(
            |c: char| !(c.is_alphanumeric() || " -_.()!=".contains(c)),
            "_",
        )
}
