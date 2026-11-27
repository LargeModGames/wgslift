//! The interface between semantic analysis and the intrinsic table.

use wgslift_syntax::Span;

use crate::Dialect;
use crate::ir::{Expr, Place, Stmt};
use crate::types::{SamplerDim, Type};

/// One argument of an intrinsic call, already lowered to the IR.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct IntrinsicArg {
    /// The value. A sampler argument is `ExprKind::Var(Root::Global(index))` with
    /// a `Type::Sampler` type.
    pub value: Expr,
    /// The place, when the argument can be assigned: the `out` arguments of
    /// `modf` and `sincos` need it.
    pub place: Option<Place>,
    /// Where the argument is in the source.
    pub span: Span,
}

/// The result of lowering a call through the intrinsic table.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Lowered {
    /// The name is not an intrinsic with this number of arguments.
    NotIntrinsic,
    /// The call, lowered.
    Done(Expr),
    /// The call is an intrinsic, but wrong; an error is reported.
    Failed,
}

/// What the intrinsic table needs from semantic analysis.
pub(crate) trait Lowerer {
    /// The dialect switches.
    fn dialect(&self) -> &Dialect;

    /// Converts a value to `to` as HLSL does implicitly: a scalar splats, a longer
    /// vector is truncated with a warning, and the component type converts.
    /// Reports an error and returns a value of type `to` when no conversion exists.
    fn coerce(&mut self, value: Expr, to: &Type, span: Span) -> Expr;

    /// Reports an error.
    fn error(&mut self, span: Span, message: String);

    /// Reports a warning.
    fn warn(&mut self, span: Span, message: String);

    /// Stores `value` in a new local before the current statement and returns a
    /// read of that local. Use it for a value that is used more than once.
    fn temp(&mut self, value: Expr) -> Expr;

    /// Adds a statement before the current statement, for example a `discard`.
    /// Outside a function it reports an error at `span`.
    fn emit(&mut self, stmt: Stmt, span: Span);

    /// Records that sampler global `global` is used with dimension `dim`. A plain
    /// `sampler` takes the dimension of its first use. Returns false after an
    /// error when the dimension conflicts.
    fn use_sampler(&mut self, global: usize, dim: SamplerDim, span: Span) -> bool;
}
