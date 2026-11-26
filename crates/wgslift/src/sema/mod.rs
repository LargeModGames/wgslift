//! Semantic analysis: names, types and implicit conversions, from the syntax tree
//! to the typed IR.

mod intrinsics;
mod lowerer;

use std::collections::HashSet;

use wgslift_syntax::Diagnostic;
use wgslift_syntax::ast::{Item, TranslationUnit};

use crate::{Dialect, Stage, ir};

/// What the analysis needs from the options.
pub(crate) struct Settings<'a> {
    /// The stage of the entry point.
    pub stage: Stage,
    /// The HLSL name of the entry point.
    pub entry_point: &'a str,
    /// The dialect switches.
    pub dialect: &'a Dialect,
}

/// The names that a unit declares at the top level.
pub(crate) fn top_level_names(unit: &TranslationUnit) -> HashSet<&str> {
    let mut names = HashSet::new();
    for item in &unit.items {
        match item {
            Item::Variable(v) => names.extend(v.declarators.iter().map(|d| d.name.name.as_str())),
            Item::Function(f) => {
                names.insert(f.name.name.as_str());
            }
            Item::Struct(s) => {
                names.insert(s.name.name.as_str());
            }
            Item::Typedef(t) => names.extend(t.names.iter().map(|(n, _)| n.name.as_str())),
            Item::Buffer(b) => names.extend(
                b.members
                    .iter()
                    .flat_map(|m| m.declarators.iter().map(|d| d.name.name.as_str())),
            ),
        }
    }
    names
}

/// Checks `units` (the source first, then predeclared units) and lowers them to
/// the IR. Returns `None` after errors.
pub(crate) fn analyze(
    units: &[TranslationUnit],
    settings: &Settings<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<ir::Module> {
    let _ = (units, settings, diagnostics);
    None
}
