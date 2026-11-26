//! The HLSL intrinsic functions, lowered to WGSL built-ins.

use wgslift_syntax::Span;

use super::lowerer::{IntrinsicArg, Lowered, Lowerer};

/// Lowers a call of `name` with `args`, or answers that it is not an intrinsic.
pub(crate) fn lower(
    cx: &mut dyn Lowerer,
    name: &str,
    args: Vec<IntrinsicArg>,
    span: Span,
) -> Lowered {
    let _ = (cx, name, args, span);
    Lowered::NotIntrinsic
}
