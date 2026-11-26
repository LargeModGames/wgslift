//! Expressions: types, HLSL's implicit conversions, places and side effects.
//!
//! A side effect inside an expression (an assignment, `++`, a call with `out`
//! arguments) becomes a statement before the current statement. HLSL evaluates
//! every operand of `&&`, `||` and `?:`, so this keeps the meaning.

use wgslift_syntax::Span;
use wgslift_syntax::ast::{self, BinaryOp as AstBinary, ExprKind as K, UnaryOp as AstUnary};

use super::lowerer::{IntrinsicArg, Lowered, Lowerer};
use super::{Analyzer, R, Reported, intrinsic_out_args, intrinsics, zero};
use crate::Dialect;
use crate::ir::{
    Access, Arg, BinaryOp, Expr, ExprKind, Literal, ParamDir, Place, Root, Space, Stmt, UnaryOp,
};
use crate::types::{SamplerDim, Scalar, Type};

/// An assignment target with its type. A swizzle of more than one component is
/// stored one component at a time.
#[derive(Debug, Clone)]
pub(crate) struct Target {
    pub place: Place,
    /// The type of the place, before the swizzle.
    pub ty: Type,
    /// The components of a vector place that the target names.
    pub swizzle: Option<Vec<u8>>,
}

impl Target {
    /// The type of the target, after the swizzle.
    pub fn value_type(&self) -> Type {
        match (&self.swizzle, &self.ty) {
            (Some(s), ty) => Type::vector(ty.scalar().unwrap_or(Scalar::Float), s.len() as u8),
            (None, ty) => ty.clone(),
        }
    }
}

/// The value of a void call: it has no use.
pub(crate) fn void() -> Expr {
    zero(&Type::Void)
}

/// HLSL's name of a type, for messages.
pub(crate) fn type_name(ty: &Type) -> String {
    let scalar = |s: Scalar| match s {
        Scalar::Bool => "bool",
        Scalar::Int => "int",
        Scalar::Uint => "uint",
        Scalar::Float => "float",
    };
    match ty {
        Type::Void => "void".into(),
        Type::Scalar(s) => scalar(*s).into(),
        Type::Vector(s, n) => format!("{}{n}", scalar(*s)),
        Type::Matrix { rows, cols } => format!("float{rows}x{cols}"),
        Type::Array(t, n) => format!("{}[{n}]", type_name(t)),
        Type::Struct(_) => "a struct".into(),
        Type::Sampler(_) => "sampler".into(),
    }
}

/// The components that a swizzle names: `xyzw` or `rgba`, not mixed.
fn parse_swizzle(name: &str) -> Option<Vec<u8>> {
    if name.is_empty() || name.len() > 4 {
        return None;
    }
    for set in ["xyzw", "rgba"] {
        let comps: Option<Vec<u8>> = name.chars().map(|c| set.find(c).map(|i| i as u8)).collect();
        if comps.is_some() {
            return comps;
        }
    }
    None
}

/// The elements a matrix member such as `_m01` or `_12_21` names, as (row, col).
fn parse_matrix_member(name: &str) -> Option<Vec<(u8, u8)>> {
    let mut out = Vec::new();
    let mut rest = name;
    while !rest.is_empty() {
        let (zero_based, tail) = if let Some(t) = rest.strip_prefix("_m") {
            (true, t)
        } else {
            (false, rest.strip_prefix('_')?)
        };
        let b = tail.as_bytes();
        if b.len() < 2 || !b[0].is_ascii_digit() || !b[1].is_ascii_digit() {
            return None;
        }
        let (r, c) = (b[0] - b'0', b[1] - b'0');
        let (r, c) = if zero_based {
            (r, c)
        } else {
            (r.checked_sub(1)?, c.checked_sub(1)?)
        };
        out.push((r, c));
        rest = &tail[2..];
    }
    (!out.is_empty() && out.len() <= 4).then_some(out)
}

/// The number of scalar components of a numeric type.
fn components(ty: &Type) -> Option<u32> {
    match ty {
        Type::Scalar(_) => Some(1),
        Type::Vector(_, n) => Some(u32::from(*n)),
        Type::Matrix { rows, cols } => Some(u32::from(rows * cols)),
        _ => None,
    }
}

/// The shape of a numeric type, without its component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Scalar,
    Vector(u8),
    Matrix(u8, u8),
}

fn shape(ty: &Type) -> Option<Shape> {
    match ty {
        Type::Scalar(_) => Some(Shape::Scalar),
        Type::Vector(_, n) => Some(Shape::Vector(*n)),
        Type::Matrix { rows, cols } => Some(Shape::Matrix(*rows, *cols)),
        _ => None,
    }
}

fn with_shape(scalar: Scalar, shape: Shape) -> Type {
    match shape {
        Shape::Scalar => Type::Scalar(scalar),
        Shape::Vector(n) => Type::Vector(scalar, n),
        Shape::Matrix(rows, cols) => Type::Matrix { rows, cols },
    }
}

/// The common shape of two operands: a scalar takes the other shape, and two
/// vectors or two matrices truncate to the smaller one.
fn common_shape(a: Shape, b: Shape) -> Option<Shape> {
    Some(match (a, b) {
        (Shape::Scalar, s) | (s, Shape::Scalar) => s,
        (Shape::Vector(n), Shape::Vector(m)) => Shape::Vector(n.min(m)),
        (Shape::Matrix(r1, c1), Shape::Matrix(r2, c2)) => Shape::Matrix(r1.min(r2), c1.min(c2)),
        _ => return None,
    })
}

/// The type both operands of an arithmetic operator convert to. A bool takes part
/// as an int.
fn arith_type(a: &Type, b: &Type) -> Option<Type> {
    let shape = common_shape(shape(a)?, shape(b)?)?;
    let promote = |s: Scalar| if s == Scalar::Bool { Scalar::Int } else { s };
    let scalar = promote(a.scalar()?).max(promote(b.scalar()?));
    if matches!(shape, Shape::Matrix(..)) && scalar != Scalar::Float {
        return Some(with_shape(Scalar::Float, shape));
    }
    Some(with_shape(scalar, shape))
}

impl Analyzer<'_> {
    // Building blocks.

    pub fn emit_stmt(&mut self, stmt: Stmt, span: Span) {
        if self.in_function() {
            self.state().pre.push(stmt);
        } else {
            self.error(span, "this is not allowed in the value of a global");
        }
    }

    /// Keeps `value` for more than one use: a simple value as it is, anything else
    /// in a new local.
    pub fn reuse(&mut self, value: Expr) -> Expr {
        let simple = matches!(value.kind, ExprKind::Literal(_) | ExprKind::Var(_));
        if simple || !self.in_function() {
            return value;
        }
        let ty = value.ty.clone();
        let id = self.new_local("", ty.clone());
        self.state().pre.push(Stmt::Local {
            id,
            init: Some(value),
        });
        Expr {
            kind: ExprKind::Var(Root::Local(id)),
            ty,
        }
    }

    fn scalar_convert(&mut self, v: Expr, to: Scalar) -> Expr {
        if v.ty == Type::Scalar(to) {
            return v;
        }
        if let ExprKind::Literal(lit) = v.kind {
            let as_f64 = match lit {
                Literal::Bool(b) => f64::from(u8::from(b)),
                Literal::Int(i) => f64::from(i),
                Literal::Uint(u) => f64::from(u),
                Literal::Float(f) => f64::from(f),
            };
            let lit = match to {
                Scalar::Bool => Literal::Bool(as_f64 != 0.0),
                Scalar::Int => Literal::Int(as_f64 as i32),
                Scalar::Uint => Literal::Uint(as_f64 as u32),
                Scalar::Float => Literal::Float(as_f64 as f32),
            };
            return Expr {
                kind: ExprKind::Literal(lit),
                ty: Type::Scalar(to),
            };
        }
        Expr {
            kind: ExprKind::Convert(Box::new(v)),
            ty: Type::Scalar(to),
        }
    }

    /// Converts `v` to `to` as HLSL does. An implicit narrowing warns; an explicit
    /// one (a cast) does not. A conversion that does not exist is an error.
    pub fn convert(&mut self, v: Expr, to: &Type, span: Span, explicit: bool) -> Expr {
        if v.ty == *to {
            return v;
        }
        let warn_truncation = |a: &mut Self, from: &Type| {
            if !explicit {
                a.warning(
                    span,
                    format!(
                        "implicit truncation from {} to {}",
                        type_name(from),
                        type_name(to)
                    ),
                );
            }
        };
        match (v.ty.clone(), to) {
            (Type::Scalar(_), Type::Scalar(t)) => self.scalar_convert(v, *t),
            (Type::Scalar(_), Type::Vector(t, _)) => {
                let s = self.scalar_convert(v, *t);
                Expr {
                    kind: ExprKind::Splat(Box::new(s)),
                    ty: to.clone(),
                }
            }
            (Type::Scalar(_), Type::Matrix { rows, cols }) => {
                let s = self.scalar_convert(v, Scalar::Float);
                let s = self.reuse(s);
                Expr {
                    kind: ExprKind::Construct(vec![s; usize::from(rows * cols)]),
                    ty: to.clone(),
                }
            }
            (Type::Vector(s, _), Type::Scalar(t)) => {
                warn_truncation(self, &v.ty);
                let x = Expr {
                    kind: ExprKind::Swizzle(Box::new(v), vec![0]),
                    ty: Type::Scalar(s),
                };
                self.scalar_convert(x, *t)
            }
            (Type::Vector(s, n), Type::Vector(t, m)) if *m <= n => {
                let mut v = v;
                if *m < n {
                    warn_truncation(self, &v.ty);
                    v = Expr {
                        kind: ExprKind::Swizzle(Box::new(v), (0..*m).collect()),
                        ty: Type::Vector(s, *m),
                    };
                }
                if s != *t {
                    v = Expr {
                        kind: ExprKind::Convert(Box::new(v)),
                        ty: to.clone(),
                    };
                }
                v
            }
            (Type::Matrix { rows: r1, cols: c1 }, Type::Matrix { rows: r2, cols: c2 })
                if *r2 <= r1 && *c2 <= c1 =>
            {
                warn_truncation(self, &v.ty);
                let m = self.reuse(v);
                let rows = (0..*r2)
                    .map(|i| {
                        let row = Expr {
                            kind: ExprKind::Index(
                                Box::new(m.clone()),
                                Box::new(Expr::int(i32::from(i))),
                            ),
                            ty: Type::Vector(Scalar::Float, c1),
                        };
                        if *c2 == c1 {
                            row
                        } else {
                            Expr {
                                kind: ExprKind::Swizzle(Box::new(row), (0..*c2).collect()),
                                ty: Type::Vector(Scalar::Float, *c2),
                            }
                        }
                    })
                    .collect();
                Expr {
                    kind: ExprKind::Construct(rows),
                    ty: to.clone(),
                }
            }
            (from, to_ty)
                if explicit
                    && components(&from).is_some()
                    && components(&from) == components(to_ty)
                    && matches!(to_ty, Type::Vector(..) | Type::Matrix { .. }) =>
            {
                let scalars = self.flatten(v, span);
                let target_scalar = to_ty.scalar().unwrap_or(Scalar::Float);
                let scalars = scalars
                    .into_iter()
                    .map(|s| self.scalar_convert(s, target_scalar))
                    .collect();
                Expr {
                    kind: ExprKind::Construct(scalars),
                    ty: to_ty.clone(),
                }
            }
            (Type::Matrix { cols, .. }, Type::Scalar(t)) => {
                // HLSL truncates a matrix to its first element, `_m00`.
                warn_truncation(self, &v.ty);
                let row = Expr {
                    kind: ExprKind::Index(Box::new(v), Box::new(Expr::int(0))),
                    ty: Type::Vector(Scalar::Float, cols),
                };
                let first = Expr {
                    kind: ExprKind::Swizzle(Box::new(row), vec![0]),
                    ty: Type::FLOAT,
                };
                self.scalar_convert(first, *t)
            }
            (Type::Scalar(_), Type::Struct(_) | Type::Array(..)) if explicit => zero(to),
            _ => {
                self.error(
                    span,
                    format!("cannot convert {} to {}", type_name(&v.ty), type_name(to)),
                );
                zero(to)
            }
        }
    }

    /// The scalar components of a scalar, vector or matrix value, in HLSL order.
    pub fn flatten(&mut self, v: Expr, span: Span) -> Vec<Expr> {
        match v.ty.clone() {
            Type::Scalar(_) => vec![v],
            Type::Vector(s, n) => {
                let v = self.reuse(v);
                (0..n)
                    .map(|i| Expr {
                        kind: ExprKind::Swizzle(Box::new(v.clone()), vec![i]),
                        ty: Type::Scalar(s),
                    })
                    .collect()
            }
            Type::Matrix { rows, cols } => {
                let m = self.reuse(v);
                let mut out = Vec::new();
                for r in 0..rows {
                    let row = Expr {
                        kind: ExprKind::Index(
                            Box::new(m.clone()),
                            Box::new(Expr::int(i32::from(r))),
                        ),
                        ty: Type::Vector(Scalar::Float, cols),
                    };
                    for c in 0..cols {
                        out.push(Expr {
                            kind: ExprKind::Swizzle(Box::new(row.clone()), vec![c]),
                            ty: Type::FLOAT,
                        });
                    }
                }
                out
            }
            other => {
                self.error(span, format!("{} has no components", type_name(&other)));
                Vec::new()
            }
        }
    }

    /// The value as a bool of the same shape: a number is true when it is not 0.
    pub fn truth(&mut self, v: Expr, span: Span) -> Expr {
        let Some(sh @ (Shape::Scalar | Shape::Vector(_))) = shape(&v.ty) else {
            self.error(
                span,
                format!("expected a condition, found {}", type_name(&v.ty)),
            );
            return Expr::bool(false);
        };
        if v.ty.scalar() == Some(Scalar::Bool) {
            return v;
        }
        let zero_value = {
            let z = self.scalar_convert(Expr::int(0), v.ty.scalar().unwrap_or(Scalar::Float));
            self.convert(z, &v.ty, span, true)
        };
        Expr {
            kind: ExprKind::Binary(BinaryOp::Ne, Box::new(v), Box::new(zero_value)),
            ty: with_shape(Scalar::Bool, sh),
        }
    }

    /// A scalar bool condition, for `if` and loops.
    pub fn condition(&mut self, e: &ast::Expr) -> R<Expr> {
        let v = self.expr(e)?;
        let v = match v.ty {
            Type::Vector(..) => self.convert(
                v.clone(),
                &Type::Scalar(v.ty.scalar().unwrap_or(Scalar::Float)),
                e.span,
                false,
            ),
            _ => v,
        };
        Ok(self.truth(v, e.span))
    }

    // Expressions.

    pub fn expr(&mut self, e: &ast::Expr) -> R<Expr> {
        match &e.kind {
            K::Literal(lit) => self.literal(lit, e.span),
            K::Ident(name) => self.ident(name, e.span),
            K::Unary(op, x) => self.unary(*op, x, e.span, true),
            K::Binary(op, a, b) => {
                let (a, b) = (self.expr(a)?, self.expr(b)?);
                self.binary(*op, a, b, e.span)
            }
            K::Assign(op, target, value) => self.assign(*op, target, value, e.span, true),
            K::Ternary(c, a, b) => self.ternary(c, a, b, e.span),
            K::Comma(list) => {
                let (last, rest) = list.split_last().expect("a comma has two parts");
                for x in rest {
                    self.effect(x)?;
                }
                self.expr(last)
            }
            K::Call(callee, args) => self.call(callee, args, e.span),
            K::Construct(t, args) => {
                let ty = self.resolve_type(t)?;
                let args = args.iter().map(|a| self.expr(a)).collect::<R<Vec<_>>>()?;
                self.construct(ty, args, e.span)
            }
            K::Cast(t, x) => {
                let ty = self.resolve_type(t)?;
                let v = self.expr(x)?;
                Ok(self.convert(v, &ty, e.span, true))
            }
            K::Member(x, name) => {
                let base = self.expr(x)?;
                self.member(base, name)
            }
            K::Index(x, i) => {
                let base = self.expr(x)?;
                let index = self.index_value(i)?;
                self.index(base, index, e.span)
            }
        }
    }

    /// Evaluates an expression for its effects only, as an expression statement.
    pub fn effect(&mut self, e: &ast::Expr) -> R<()> {
        match &e.kind {
            K::Assign(op, target, value) => {
                self.assign(*op, target, value, e.span, false)?;
            }
            K::Unary(
                op @ (AstUnary::PreInc | AstUnary::PreDec | AstUnary::PostInc | AstUnary::PostDec),
                x,
            ) => {
                self.unary(*op, x, e.span, false)?;
            }
            K::Comma(list) => {
                for x in list {
                    self.effect(x)?;
                }
            }
            _ => {
                let v = self.expr(e)?;
                if v.ty != Type::Void || !matches!(&v.kind, ExprKind::Construct(l) if l.is_empty())
                {
                    self.keep_effects(v, e.span);
                }
            }
        }
        Ok(())
    }

    /// Keeps the calls in an unused value.
    fn keep_effects(&mut self, v: Expr, span: Span) {
        let mut has_call = false;
        super::visit_expr(&v, &mut |x| {
            has_call |= matches!(x.kind, ExprKind::Call(..))
        });
        if matches!(v.kind, ExprKind::Call(..)) {
            self.emit_stmt(Stmt::Eval(v), span);
        } else if has_call {
            self.reuse(v);
        }
    }

    fn literal(&mut self, lit: &ast::Literal, span: Span) -> R<Expr> {
        let (kind, ty) = match lit {
            ast::Literal::Bool(b) => (Literal::Bool(*b), Type::BOOL),
            ast::Literal::Float(text) => {
                let digits = text.trim_end_matches(['f', 'F', 'h', 'H', 'l', 'L']);
                let value: f64 = digits
                    .parse()
                    .map_err(|_| self.error(span, format!("a bad number `{text}`")))?;
                (Literal::Float(value as f32), Type::FLOAT)
            }
            ast::Literal::Int(text) => {
                let digits = text.trim_end_matches(['u', 'U', 'l', 'L']);
                let unsigned = text[digits.len()..].contains(['u', 'U']);
                let value = if let Some(hex) = digits
                    .strip_prefix("0x")
                    .or_else(|| digits.strip_prefix("0X"))
                {
                    u64::from_str_radix(hex, 16).ok()
                } else if digits.len() > 1 && digits.starts_with('0') {
                    u64::from_str_radix(&digits[1..], 8).ok()
                } else {
                    digits.parse().ok()
                };
                let value =
                    value.ok_or_else(|| self.error(span, format!("a bad integer `{text}`")))?;
                if unsigned || value > i32::MAX as u64 {
                    let v = u32::try_from(value).map_err(|_| {
                        self.error(span, format!("`{text}` does not fit in 32 bits"))
                    })?;
                    (Literal::Uint(v), Type::Scalar(Scalar::Uint))
                } else {
                    (Literal::Int(value as i32), Type::INT)
                }
            }
            ast::Literal::Str(_) => return Err(self.error(span, "strings are not supported")),
        };
        Ok(Expr {
            kind: ExprKind::Literal(kind),
            ty,
        })
    }

    fn ident(&mut self, name: &str, span: Span) -> R<Expr> {
        if let Some(id) = self.lookup_local(name) {
            let ty = self.func.as_ref().expect("in a function").locals[id]
                .ty
                .clone();
            return Ok(Expr {
                kind: ExprKind::Var(Root::Local(id)),
                ty,
            });
        }
        if let Some(&g) = self.globals.get(name) {
            return Ok(Expr {
                kind: ExprKind::Var(Root::Global(g)),
                ty: self.module.globals[g].ty.clone(),
            });
        }
        if self.functions.contains_key(name) {
            return Err(self.error(span, format!("`{name}` is a function, not a value")));
        }
        Err(self.error(span, format!("unknown name `{name}`")))
    }

    fn unary(&mut self, op: AstUnary, x: &ast::Expr, span: Span, want_value: bool) -> R<Expr> {
        match op {
            AstUnary::PreInc | AstUnary::PreDec | AstUnary::PostInc | AstUnary::PostDec => {
                let target = self.place(x)?;
                let old = self.read(&target);
                let ty = old.ty.clone();
                if components(&ty).is_none() || ty.scalar() == Some(Scalar::Bool) {
                    return Err(self.error(span, format!("cannot increment {}", type_name(&ty))));
                }
                let post = matches!(op, AstUnary::PostInc | AstUnary::PostDec);
                let old = if post && want_value {
                    self.reuse(old)
                } else {
                    old
                };
                let one = self.convert(Expr::int(1), &ty, span, true);
                let bop = if matches!(op, AstUnary::PreInc | AstUnary::PostInc) {
                    BinaryOp::Add
                } else {
                    BinaryOp::Sub
                };
                let new = Expr {
                    kind: ExprKind::Binary(bop, Box::new(old.clone()), Box::new(one)),
                    ty,
                };
                self.store(&target, new, span);
                Ok(if post { old } else { self.read(&target) })
            }
            AstUnary::Plus => {
                let v = self.expr(x)?;
                Ok(match v.ty.scalar() {
                    Some(Scalar::Bool) => {
                        let ty = v.ty.with_scalar(Scalar::Int);
                        self.convert(v, &ty, span, true)
                    }
                    _ => v,
                })
            }
            AstUnary::Neg => {
                let mut v = self.expr(x)?;
                if components(&v.ty).is_none() {
                    return Err(self.error(span, format!("cannot negate {}", type_name(&v.ty))));
                }
                if v.ty.scalar() == Some(Scalar::Bool) {
                    let ty = v.ty.with_scalar(Scalar::Int);
                    v = self.convert(v, &ty, span, true);
                }
                if matches!(v.ty, Type::Matrix { .. }) {
                    let ty = v.ty.clone();
                    return Ok(Expr {
                        kind: ExprKind::Binary(
                            BinaryOp::Mul,
                            Box::new(v),
                            Box::new(Expr::float(-1.0)),
                        ),
                        ty,
                    });
                }
                let ty = v.ty.clone();
                Ok(Expr {
                    kind: ExprKind::Unary(UnaryOp::Neg, Box::new(v)),
                    ty,
                })
            }
            AstUnary::Not => {
                let v = self.expr(x)?;
                let b = self.truth(v, span);
                let ty = b.ty.clone();
                Ok(Expr {
                    kind: ExprKind::Unary(UnaryOp::Not, Box::new(b)),
                    ty,
                })
            }
            AstUnary::BitNot => {
                let v = self.expr(x)?;
                if !matches!(v.ty.scalar(), Some(Scalar::Int | Scalar::Uint)) {
                    return Err(self.error(
                        span,
                        format!("`~` needs an integer, found {}", type_name(&v.ty)),
                    ));
                }
                let ty = v.ty.clone();
                Ok(Expr {
                    kind: ExprKind::Unary(UnaryOp::BitNot, Box::new(v)),
                    ty,
                })
            }
        }
    }

    pub fn binary(&mut self, op: AstBinary, a: Expr, b: Expr, span: Span) -> R<Expr> {
        let mismatch = |s: &mut Self| {
            s.error(
                span,
                format!(
                    "cannot use `{op:?}` on {} and {}",
                    type_name(&a.ty),
                    type_name(&b.ty)
                ),
            )
        };
        let Some(ty) = arith_type(&a.ty, &b.ty) else {
            return Err(mismatch(self));
        };
        let sh = shape(&ty).expect("numeric");
        let bop = match op {
            AstBinary::Add => BinaryOp::Add,
            AstBinary::Sub => BinaryOp::Sub,
            AstBinary::Mul => BinaryOp::Mul,
            AstBinary::Div => BinaryOp::Div,
            AstBinary::Rem => BinaryOp::Rem,
            AstBinary::Lt => BinaryOp::Lt,
            AstBinary::Gt => BinaryOp::Gt,
            AstBinary::Le => BinaryOp::Le,
            AstBinary::Ge => BinaryOp::Ge,
            AstBinary::Eq => BinaryOp::Eq,
            AstBinary::Ne => BinaryOp::Ne,
            AstBinary::And | AstBinary::Or => {
                let bool_ty = with_shape(Scalar::Bool, sh);
                if matches!(sh, Shape::Matrix(..)) {
                    return Err(mismatch(self));
                }
                let a = {
                    let v = self.convert(a, &with_shape(a_scalar(&ty), sh), span, false);
                    self.truth(v, span)
                };
                let b = {
                    let v = self.convert(b, &with_shape(a_scalar(&ty), sh), span, false);
                    self.truth(v, span)
                };
                let bop = if op == AstBinary::And {
                    BinaryOp::And
                } else {
                    BinaryOp::Or
                };
                return Ok(Expr {
                    kind: ExprKind::Binary(bop, Box::new(a), Box::new(b)),
                    ty: bool_ty,
                });
            }
            AstBinary::BitAnd
            | AstBinary::BitOr
            | AstBinary::BitXor
            | AstBinary::Shl
            | AstBinary::Shr => {
                if !matches!(ty.scalar(), Some(Scalar::Int | Scalar::Uint))
                    || matches!(sh, Shape::Matrix(..))
                {
                    return Err(mismatch(self));
                }
                let bop = match op {
                    AstBinary::BitAnd => BinaryOp::And,
                    AstBinary::BitOr => BinaryOp::Or,
                    AstBinary::BitXor => BinaryOp::Xor,
                    AstBinary::Shl => BinaryOp::Shl,
                    _ => BinaryOp::Shr,
                };
                let a = self.convert(a, &ty, span, false);
                // WGSL shifts by an unsigned amount.
                let b = self.convert(b, &ty.with_scalar(Scalar::Uint), span, false);
                return Ok(Expr {
                    kind: ExprKind::Binary(bop, Box::new(a), Box::new(b)),
                    ty,
                });
            }
        };
        let compare = matches!(
            bop,
            BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge | BinaryOp::Eq | BinaryOp::Ne
        );
        if let Shape::Matrix(rows, cols) = sh {
            // WGSL has no element-wise matrix product, division, remainder or
            // comparison: build the result one row at a time.
            let is_scale = matches!(bop, BinaryOp::Mul)
                && (shape(&a.ty) == Some(Shape::Scalar) || shape(&b.ty) == Some(Shape::Scalar));
            if !compare && (matches!(bop, BinaryOp::Add | BinaryOp::Sub) || is_scale) {
                // WGSL scales a matrix by a scalar, but adds only matrices.
                let keep_scalar = |e: &Expr| is_scale && shape(&e.ty) == Some(Shape::Scalar);
                let a = if keep_scalar(&a) {
                    self.convert(a, &Type::FLOAT, span, false)
                } else {
                    self.convert(a, &ty, span, false)
                };
                let b = if keep_scalar(&b) {
                    self.convert(b, &Type::FLOAT, span, false)
                } else {
                    self.convert(b, &ty, span, false)
                };
                return Ok(Expr {
                    kind: ExprKind::Binary(bop, Box::new(a), Box::new(b)),
                    ty,
                });
            }
            if compare {
                return Err(self.error(span, "matrix comparisons are not supported"));
            }
            let a = self.convert(a, &ty, span, false);
            let b = self.convert(b, &ty, span, false);
            let (a, b) = (self.reuse(a), self.reuse(b));
            let row_ty = Type::Vector(Scalar::Float, cols);
            let rows = (0..rows)
                .map(|r| {
                    let row = |m: &Expr| Expr {
                        kind: ExprKind::Index(
                            Box::new(m.clone()),
                            Box::new(Expr::int(i32::from(r))),
                        ),
                        ty: row_ty.clone(),
                    };
                    Expr {
                        kind: ExprKind::Binary(bop, Box::new(row(&a)), Box::new(row(&b))),
                        ty: row_ty.clone(),
                    }
                })
                .collect();
            return Ok(Expr {
                kind: ExprKind::Construct(rows),
                ty,
            });
        }
        let a = self.convert(a, &ty, span, false);
        let b = self.convert(b, &ty, span, false);
        let result = if compare {
            with_shape(Scalar::Bool, sh)
        } else {
            ty
        };
        Ok(Expr {
            kind: ExprKind::Binary(bop, Box::new(a), Box::new(b)),
            ty: result,
        })
    }

    fn assign(
        &mut self,
        op: Option<AstBinary>,
        target: &ast::Expr,
        value: &ast::Expr,
        span: Span,
        want_value: bool,
    ) -> R<Expr> {
        let place = self.place(target)?;
        let mut v = self.expr(value)?;
        if let Some(op) = op {
            let current = self.read(&place);
            v = self.binary(op, current, v, span)?;
        }
        let ty = place.value_type();
        let v = self.convert(v, &ty, span, false);
        self.store(&place, v, span);
        Ok(if want_value {
            self.read(&place)
        } else {
            void()
        })
    }

    fn ternary(&mut self, c: &ast::Expr, a: &ast::Expr, b: &ast::Expr, span: Span) -> R<Expr> {
        let cond = self.expr(c)?;
        let (a, b) = (self.expr(a)?, self.expr(b)?);
        let ty = if a.ty == b.ty {
            a.ty.clone()
        } else {
            arith_type(&a.ty, &b.ty).ok_or_else(|| {
                self.error(
                    span,
                    format!(
                        "the two values of `?:` have the types {} and {}",
                        type_name(&a.ty),
                        type_name(&b.ty)
                    ),
                )
            })?
        };
        // A vector condition selects per component.
        let ty = match (shape(&cond.ty), shape(&ty)) {
            (Some(Shape::Vector(n)), Some(Shape::Scalar)) => {
                Type::vector(ty.scalar().unwrap_or(Scalar::Float), n)
            }
            (Some(Shape::Vector(n)), Some(Shape::Vector(m))) if n != m => {
                Type::vector(ty.scalar().unwrap_or(Scalar::Float), n.min(m))
            }
            _ => ty,
        };
        let cond_ty = match shape(&ty) {
            Some(Shape::Vector(n)) if matches!(shape(&cond.ty), Some(Shape::Vector(_))) => {
                Type::Vector(cond.ty.scalar().unwrap_or(Scalar::Bool), n)
            }
            _ => Type::Scalar(cond.ty.scalar().unwrap_or(Scalar::Bool)),
        };
        let cond = self.convert(cond, &cond_ty, span, false);
        let cond = self.truth(cond, span);
        let accept = self.convert(a, &ty, span, false);
        let reject = self.convert(b, &ty, span, false);
        Ok(Expr {
            kind: ExprKind::Select {
                cond: Box::new(cond),
                accept: Box::new(accept),
                reject: Box::new(reject),
            },
            ty,
        })
    }

    pub fn construct(&mut self, ty: Type, args: Vec<Expr>, span: Span) -> R<Expr> {
        let Some(needed) = components(&ty) else {
            return Err(self.error(span, format!("{} has no constructor", type_name(&ty))));
        };
        let total: u32 = args.iter().map(|a| components(&a.ty).unwrap_or(0)).sum();
        if args.iter().any(|a| components(&a.ty).is_none()) {
            return Err(self.error(span, "a constructor takes numbers only"));
        }
        let scalar = ty.scalar().unwrap_or(Scalar::Float);
        // One value: a scalar repeats, and anything else converts.
        if let [single] = args.as_slice() {
            if single.ty.size() == Some(1) && needed > 1 {
                let s = self.convert(single.clone(), &Type::Scalar(scalar), span, true);
                return Ok(self.convert(s, &ty, span, true));
            }
            if total == needed || matches!(ty, Type::Scalar(_)) {
                return Ok(self.convert(single.clone(), &ty, span, true));
            }
        }
        if total != needed {
            return Err(self.error(
                span,
                format!("{} needs {needed} values, found {total}", type_name(&ty)),
            ));
        }
        let (split_all, part_scalar) = match ty {
            // A vector takes scalars and vectors; a matrix argument splits.
            Type::Vector(..) => (false, scalar),
            // A matrix takes all scalars or all rows; anything else splits.
            Type::Matrix { cols, .. } => {
                let rows_given = args
                    .iter()
                    .all(|a| a.ty == Type::Vector(Scalar::Float, cols));
                let all_scalars = args.iter().all(|a| a.ty.size() == Some(1));
                (!(rows_given || all_scalars), Scalar::Float)
            }
            _ => unreachable!("single values handled above"),
        };
        let mut parts = Vec::new();
        for a in args {
            let pieces = if split_all || matches!(a.ty, Type::Matrix { .. }) {
                self.flatten(a, span)
            } else {
                vec![a]
            };
            for piece in pieces {
                let t = piece.ty.with_scalar(part_scalar);
                parts.push(self.convert(piece, &t, span, true));
            }
        }
        Ok(Expr {
            kind: ExprKind::Construct(parts),
            ty,
        })
    }

    fn member(&mut self, base: Expr, name: &ast::Ident) -> R<Expr> {
        match base.ty.clone() {
            Type::Struct(s) => {
                let fields = &self.module.structs[s].fields;
                let Some(i) = fields.iter().position(|f| f.name == name.name) else {
                    return Err(self.error(
                        name.span,
                        format!("the struct has no field `{}`", name.name),
                    ));
                };
                let ty = fields[i].ty.clone();
                Ok(Expr {
                    kind: ExprKind::Field(Box::new(base), i),
                    ty,
                })
            }
            Type::Scalar(s) | Type::Vector(s, _) => {
                let size = base.ty.size().expect("scalar or vector");
                let comps = parse_swizzle(&name.name).filter(|c| c.iter().all(|&i| i < size));
                let Some(comps) = comps else {
                    return Err(self.error(
                        name.span,
                        format!(
                            "`.{}` is not a swizzle of {}",
                            name.name,
                            type_name(&base.ty)
                        ),
                    ));
                };
                let ty = Type::vector(s, comps.len() as u8);
                if size == 1 {
                    // `x.xxx` repeats a scalar.
                    let b = base;
                    return Ok(if comps.len() == 1 {
                        b
                    } else {
                        self.convert(b, &ty, name.span, true)
                    });
                }
                Ok(Expr {
                    kind: ExprKind::Swizzle(Box::new(base), comps),
                    ty,
                })
            }
            Type::Matrix { rows, cols } => {
                let elements = parse_matrix_member(&name.name)
                    .filter(|e| e.iter().all(|&(r, c)| r < rows && c < cols));
                let Some(elements) = elements else {
                    return Err(self.error(
                        name.span,
                        format!("`.{}` is not a matrix element", name.name),
                    ));
                };
                let m = self.reuse(base);
                let parts: Vec<Expr> = elements
                    .iter()
                    .map(|&(r, c)| {
                        let row = Expr {
                            kind: ExprKind::Index(
                                Box::new(m.clone()),
                                Box::new(Expr::int(i32::from(r))),
                            ),
                            ty: Type::Vector(Scalar::Float, cols),
                        };
                        Expr {
                            kind: ExprKind::Swizzle(Box::new(row), vec![c]),
                            ty: Type::FLOAT,
                        }
                    })
                    .collect();
                Ok(match parts.len() {
                    1 => parts.into_iter().next().expect("one"),
                    n => Expr {
                        kind: ExprKind::Construct(parts),
                        ty: Type::Vector(Scalar::Float, n as u8),
                    },
                })
            }
            other => Err(self.error(name.span, format!("{} has no members", type_name(&other)))),
        }
    }

    fn index_value(&mut self, i: &ast::Expr) -> R<Expr> {
        let v = self.expr(i)?;
        if v.ty.size() != Some(1) {
            return Err(self.error(
                i.span,
                format!("an index must be a number, found {}", type_name(&v.ty)),
            ));
        }
        let v = match v.ty {
            Type::Scalar(Scalar::Int | Scalar::Uint) => v,
            _ => self.convert(v, &Type::INT, i.span, true),
        };
        Ok(v)
    }

    fn index(&mut self, base: Expr, index: Expr, span: Span) -> R<Expr> {
        let ty = match &base.ty {
            Type::Array(element, _) => (**element).clone(),
            Type::Vector(s, _) => Type::Scalar(*s),
            Type::Matrix { cols, .. } => Type::Vector(Scalar::Float, *cols),
            other => return Err(self.error(span, format!("cannot index {}", type_name(other)))),
        };
        Ok(Expr {
            kind: ExprKind::Index(Box::new(base), Box::new(index)),
            ty,
        })
    }

    // Places.

    /// The assignment target that `e` names. Its writes count for the root.
    pub fn place(&mut self, e: &ast::Expr) -> R<Target> {
        let target = match &e.kind {
            K::Ident(name) => {
                if let Some(id) = self.lookup_local(name) {
                    let state = self.state();
                    state.locals[id].written = true;
                    Target {
                        place: Place {
                            root: Root::Local(id),
                            path: Vec::new(),
                        },
                        ty: state.locals[id].ty.clone(),
                        swizzle: None,
                    }
                } else if let Some(&g) = self.globals.get(name) {
                    let global = &self.module.globals[g];
                    if global.space != Space::Private {
                        return Err(self.error(e.span, format!("`{name}` cannot be written")));
                    }
                    Target {
                        place: Place {
                            root: Root::Global(g),
                            path: Vec::new(),
                        },
                        ty: global.ty.clone(),
                        swizzle: None,
                    }
                } else {
                    return Err(self.error(e.span, format!("unknown name `{name}`")));
                }
            }
            K::Member(x, name) => {
                let mut t = self.place(x)?;
                if let Some(sw) = &t.swizzle {
                    let comps = parse_swizzle(&name.name)
                        .filter(|c| c.iter().all(|&i| usize::from(i) < sw.len()));
                    let Some(comps) = comps else {
                        return Err(self
                            .error(name.span, format!("`.{}` is not a swizzle here", name.name)));
                    };
                    t.swizzle = Some(comps.iter().map(|&i| sw[usize::from(i)]).collect());
                    return Ok(t);
                }
                match t.ty.clone() {
                    Type::Struct(s) => {
                        let fields = &self.module.structs[s].fields;
                        let Some(i) = fields.iter().position(|f| f.name == name.name) else {
                            return Err(self.error(
                                name.span,
                                format!("the struct has no field `{}`", name.name),
                            ));
                        };
                        t.ty = fields[i].ty.clone();
                        t.place.path.push(Access::Field(i));
                    }
                    Type::Vector(_, n) => {
                        let comps = parse_swizzle(&name.name).filter(|c| c.iter().all(|&i| i < n));
                        let Some(comps) = comps else {
                            return Err(self.error(
                                name.span,
                                format!(
                                    "`.{}` is not a swizzle of {}",
                                    name.name,
                                    type_name(&t.ty)
                                ),
                            ));
                        };
                        let mut unique = comps.clone();
                        unique.sort_unstable();
                        unique.dedup();
                        if unique.len() != comps.len() {
                            return Err(self.error(
                                name.span,
                                "a swizzle that is written cannot repeat a component",
                            ));
                        }
                        t.swizzle = Some(comps);
                    }
                    Type::Scalar(_) if matches!(name.name.as_str(), "x" | "r") => {}
                    Type::Matrix { rows, cols } => {
                        let element = parse_matrix_member(&name.name)
                            .filter(|e| e.len() == 1 && e[0].0 < rows && e[0].1 < cols);
                        let Some(&[(r, c)]) = element.as_deref() else {
                            return Err(self.error(
                                name.span,
                                "only one matrix element can be written at a time",
                            ));
                        };
                        t.place.path.push(Access::Index(Expr::int(i32::from(r))));
                        t.place.path.push(Access::Component(c));
                        t.ty = Type::FLOAT;
                    }
                    other => {
                        return Err(
                            self.error(name.span, format!("{} has no members", type_name(&other)))
                        );
                    }
                }
                t
            }
            K::Index(x, i) => {
                let mut t = self.place(x)?;
                if t.swizzle.is_some() {
                    return Err(self.error(e.span, "cannot index a swizzle"));
                }
                let index = self.index_value(i)?;
                let index = self.reuse(index);
                let (ty, access) = match t.ty.clone() {
                    Type::Array(element, _) => (*element, Access::Index(index)),
                    Type::Vector(s, _) => (Type::Scalar(s), Access::Index(index)),
                    Type::Matrix { cols, .. } => {
                        (Type::Vector(Scalar::Float, cols), Access::Index(index))
                    }
                    other => {
                        return Err(
                            self.error(e.span, format!("cannot index {}", type_name(&other)))
                        );
                    }
                };
                t.ty = ty;
                t.place.path.push(access);
                t
            }
            _ => return Err(self.error(e.span, "this cannot be assigned")),
        };
        Ok(target)
    }

    /// The current value of a target.
    pub fn read(&mut self, t: &Target) -> Expr {
        let root_ty = match t.place.root {
            Root::Local(id) => self.state().locals[id].ty.clone(),
            Root::Global(g) => self.module.globals[g].ty.clone(),
        };
        let mut value = Expr {
            kind: ExprKind::Var(t.place.root),
            ty: root_ty,
        };
        for access in &t.place.path {
            value = match access {
                Access::Field(i) => {
                    let Type::Struct(s) = value.ty else {
                        unreachable!("a field of a struct")
                    };
                    let ty = self.module.structs[s].fields[*i].ty.clone();
                    Expr {
                        kind: ExprKind::Field(Box::new(value), *i),
                        ty,
                    }
                }
                Access::Index(i) => {
                    let ty = match &value.ty {
                        Type::Array(element, _) => (**element).clone(),
                        Type::Vector(s, _) => Type::Scalar(*s),
                        Type::Matrix { cols, .. } => Type::Vector(Scalar::Float, *cols),
                        _ => unreachable!("checked in place()"),
                    };
                    Expr {
                        kind: ExprKind::Index(Box::new(value), Box::new(i.clone())),
                        ty,
                    }
                }
                Access::Component(c) => {
                    let ty = Type::Scalar(value.ty.scalar().unwrap_or(Scalar::Float));
                    Expr {
                        kind: ExprKind::Swizzle(Box::new(value), vec![*c]),
                        ty,
                    }
                }
            };
        }
        if let Some(sw) = &t.swizzle {
            let ty = Type::vector(value.ty.scalar().unwrap_or(Scalar::Float), sw.len() as u8);
            value = Expr {
                kind: ExprKind::Swizzle(Box::new(value), sw.clone()),
                ty,
            };
        }
        value
    }

    /// Stores a value of the target's type. A swizzle stores each component.
    pub fn store(&mut self, t: &Target, value: Expr, span: Span) {
        match &t.swizzle {
            None => self.emit_stmt(Stmt::Assign(t.place.clone(), value), span),
            Some(sw) if sw.len() == 1 => {
                let mut place = t.place.clone();
                place.path.push(Access::Component(sw[0]));
                self.emit_stmt(Stmt::Assign(place, value), span);
            }
            Some(sw) => {
                let value = self.reuse(value);
                let scalar = value.ty.scalar().unwrap_or(Scalar::Float);
                for (k, &c) in sw.iter().enumerate() {
                    let mut place = t.place.clone();
                    place.path.push(Access::Component(c));
                    let part = Expr {
                        kind: ExprKind::Swizzle(Box::new(value.clone()), vec![k as u8]),
                        ty: Type::Scalar(scalar),
                    };
                    self.emit_stmt(Stmt::Assign(place, part), span);
                }
            }
        }
    }

    // Calls.

    fn call(&mut self, callee: &ast::Expr, args: &[ast::Expr], span: Span) -> R<Expr> {
        let K::Ident(name) = &callee.kind else {
            return Err(self.error(callee.span, "methods and indirect calls are not supported"));
        };
        if self.lookup_local(name).is_some() || self.globals.contains_key(name) {
            return Err(self.error(
                callee.span,
                format!("`{name}` is a variable, not a function"),
            ));
        }
        if let Some(overloads) = self.functions.get(name).cloned() {
            return self.user_call(name, &overloads, args, span);
        }
        let outs = intrinsic_out_args(name, args.len()).unwrap_or(&[]);
        let mut lowered = Vec::new();
        for (i, a) in args.iter().enumerate() {
            let (value, place) = if outs.contains(&i) {
                let t = self.place(a)?;
                let value = self.read(&t);
                let place = t.swizzle.is_none().then(|| t.place.clone());
                (value, place)
            } else {
                (self.expr(a)?, None)
            };
            lowered.push(IntrinsicArg {
                value,
                place,
                span: a.span,
            });
        }
        match intrinsics::lower(self, name, lowered, span) {
            Lowered::Done(e) => Ok(e),
            Lowered::Failed => Err(Reported),
            Lowered::NotIntrinsic => Err(self.error(
                callee.span,
                format!("unknown function `{name}` with {} arguments", args.len()),
            )),
        }
    }

    fn user_call(
        &mut self,
        name: &str,
        overloads: &[usize],
        args: &[ast::Expr],
        span: Span,
    ) -> R<Expr> {
        let fits = |sig: &super::Signature| {
            let required = sig.params.iter().filter(|p| p.default.is_none()).count();
            (required..=sig.params.len()).contains(&args.len())
        };
        let candidates: Vec<usize> = overloads
            .iter()
            .copied()
            .filter(|&f| fits(&self.signatures[f]))
            .collect();
        let index = match candidates.as_slice() {
            [] => {
                return Err(self.error(span, format!("no `{name}` takes {} arguments", args.len())));
            }
            [one] => *one,
            many => {
                // Pick the overload whose parameter types match the most arguments.
                let mut types = Vec::new();
                for a in args {
                    types.push(self.expr(a)?.ty);
                }
                *many
                    .iter()
                    .max_by_key(|&&f| {
                        self.signatures[f]
                            .params
                            .iter()
                            .zip(&types)
                            .filter(|(p, t)| p.ty == **t)
                            .count()
                    })
                    .expect("not empty")
            }
        };
        if !self.has_body(index) {
            return Err(self.error(span, format!("`{name}` is declared but has no body")));
        }
        let sig = self.signatures[index].clone();
        let mut call_args = Vec::new();
        let mut write_back = Vec::new();
        for (i, p) in sig.params.iter().enumerate() {
            match (p.dir, args.get(i)) {
                (ParamDir::In, Some(a)) => {
                    let v = self.expr(a)?;
                    call_args.push(Arg::Value(self.convert(v, &p.ty, a.span, false)));
                }
                (ParamDir::In, None) => {
                    let default = p.default.clone().expect("checked by fits()");
                    let v = self.expr(&default)?;
                    call_args.push(Arg::Value(self.convert(v, &p.ty, span, false)));
                }
                (dir, Some(a)) => {
                    let target = self.place(a)?;
                    let init = if dir == ParamDir::InOut {
                        let current = self.read(&target);
                        Some(self.convert(current, &p.ty, a.span, false))
                    } else {
                        None
                    };
                    let tmp = self.new_local("", p.ty.clone());
                    self.state().locals[tmp].written = true;
                    self.emit_stmt(Stmt::Local { id: tmp, init }, a.span);
                    call_args.push(Arg::Ref(tmp));
                    write_back.push((target, tmp, a.span));
                }
                (_, None) => return Err(self.error(span, "an `out` parameter needs an argument")),
            }
        }
        let call = Expr {
            kind: ExprKind::Call(index, call_args),
            ty: sig.ret.clone(),
        };
        if write_back.is_empty() {
            return Ok(call);
        }
        let result = if sig.ret == Type::Void {
            self.emit_stmt(Stmt::Eval(call), span);
            void()
        } else {
            let id = self.new_local("", sig.ret.clone());
            self.emit_stmt(
                Stmt::Local {
                    id,
                    init: Some(call),
                },
                span,
            );
            Expr {
                kind: ExprKind::Var(Root::Local(id)),
                ty: sig.ret.clone(),
            }
        };
        for (target, tmp, arg_span) in write_back {
            let ty = self.state().locals[tmp].ty.clone();
            let v = Expr {
                kind: ExprKind::Var(Root::Local(tmp)),
                ty,
            };
            let v = self.convert(v, &target.value_type(), arg_span, false);
            self.store(&target, v, arg_span);
        }
        Ok(result)
    }

    fn has_body(&self, index: usize) -> bool {
        self.signatures[index].has_body
    }
}

/// The component type of an arithmetic type.
fn a_scalar(ty: &Type) -> Scalar {
    ty.scalar().unwrap_or(Scalar::Float)
}

impl Lowerer for Analyzer<'_> {
    fn dialect(&self) -> &Dialect {
        self.dialect
    }

    fn coerce(&mut self, value: Expr, to: &Type, span: Span) -> Expr {
        self.convert(value, to, span, false)
    }

    fn error(&mut self, span: Span, message: String) {
        Analyzer::error(self, span, message);
    }

    fn warn(&mut self, span: Span, message: String) {
        self.warning(span, message);
    }

    fn temp(&mut self, value: Expr) -> Expr {
        if !self.in_function() {
            return value;
        }
        let ty = value.ty.clone();
        let id = self.new_local("", ty.clone());
        self.state().pre.push(Stmt::Local {
            id,
            init: Some(value),
        });
        Expr {
            kind: ExprKind::Var(Root::Local(id)),
            ty,
        }
    }

    fn emit(&mut self, stmt: Stmt) {
        if self.in_function() {
            self.state().pre.push(stmt);
        } else {
            self.errors = true;
        }
    }

    fn use_sampler(&mut self, global: usize, dim: SamplerDim, span: Span) -> bool {
        let Some(g) = self.module.globals.get_mut(global) else {
            return false;
        };
        match g.ty {
            Type::Sampler(SamplerDim::Any) => {
                g.ty = Type::Sampler(dim);
                true
            }
            Type::Sampler(current) if current == dim => true,
            Type::Sampler(current) => {
                let message = format!(
                    "`{}` is used as a {current:?} and a {dim:?} sampler",
                    g.name
                );
                Analyzer::error(self, span, message);
                false
            }
            _ => {
                Analyzer::error(self, span, "expected a sampler");
                false
            }
        }
    }
}
