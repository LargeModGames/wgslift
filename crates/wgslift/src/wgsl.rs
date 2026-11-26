//! The WGSL printer: typed IR to WGSL text and reflection.

mod names;
mod uniforms;

use crate::reflection::Reflection;
use crate::{Stage, Uniformity, ir};

/// Settings for the printer.
pub(crate) struct PrintOptions {
    /// The stage of the entry point.
    pub stage: Stage,
    /// The uniformity policy.
    pub uniformity: Uniformity,
}

/// Prints `module` as WGSL and reports its bindings and interface.
pub(crate) fn print(module: &ir::Module, options: &PrintOptions) -> (String, Reflection) {
    let _ = (module, options);
    (String::new(), Reflection::default())
}
