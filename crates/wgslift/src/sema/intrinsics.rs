//! The HLSL intrinsic functions, lowered to WGSL built-ins.
//!
//! WGSL built-ins take scalars and vectors only, so a component-wise intrinsic on
//! a matrix runs once per WGSL column, which is one HLSL row.

use wgslift_syntax::Span;

use super::lowerer::{IntrinsicArg, Lowered, Lowerer};
use crate::ir::{BinaryOp, Expr, ExprKind, Literal, Place, Root, Sample, Stmt};
use crate::types::{SamplerDim, Scalar, Type};

/// Lowers a call of `name` with `args`, or answers that it is not an intrinsic.
pub(crate) fn lower(
    cx: &mut dyn Lowerer,
    name: &str,
    args: Vec<IntrinsicArg>,
    span: Span,
) -> Lowered {
    if let Some((dim, form)) = texture_function(name, args.len()) {
        return sample(cx, name, dim, form, args);
    }
    let result = if let Some((wgsl, elem, abs_input)) = componentwise(name, args.len()) {
        lower_componentwise(cx, name, wgsl, elem, abs_input, args, span)
    } else {
        match (name, args.len()) {
            ("mul", 2) => mul(cx, args, span),
            ("pow", 2) => pow(cx, args, span),
            ("fmod", 2) => fmod(cx, args, span),
            ("log10", 1) => log10(cx, args, span),
            ("sign", 1) => sign(cx, args, span),
            ("dot", 2) => dot(cx, args, span),
            // A scalar normalizes to x / |x|, which is its sign.
            ("normalize", 1) if args[0].value.ty.size() == Some(1) => {
                let x = cx.coerce(args[0].value.clone(), &Type::FLOAT, span);
                Some(Expr {
                    kind: ExprKind::Builtin("sign", vec![x]),
                    ty: Type::FLOAT,
                })
            }
            ("length", 1) | ("distance", 2) | ("normalize", 1) | ("reflect", 2) => {
                geometric(cx, name, args, span)
            }
            ("cross", 2) => Some(cross(cx, args)),
            ("determinant", 1) | ("transpose", 1) => matrix_function(cx, name, args),
            ("all", 1) | ("any", 1) => all_any(cx, name, args),
            ("isnan", 1) | ("isinf", 1) => isnan_isinf(cx, name, args, span),
            ("lit", 3) => Some(lit(cx, args)),
            ("modf", 2) => modf(cx, args, span),
            ("sincos", 3) => sincos(cx, args, span),
            ("clip", 1) => clip(cx, args, span),
            _ => return Lowered::NotIntrinsic,
        }
    };
    match result {
        Some(expr) if name == "clamp" => Lowered::Done(clamp_as_min_max(expr)),
        Some(expr) => Lowered::Done(expr),
        None => Lowered::Failed,
    }
}

/// HLSL's `clamp(x, low, high)` is `min(max(x, low), high)`, also when `low` is
/// above `high`. WGSL rejects a constant `clamp` with `low` above `high`.
fn clamp_as_min_max(e: Expr) -> Expr {
    match e.kind {
        ExprKind::Builtin("clamp", args) if args.len() == 3 => {
            let [x, low, high]: [Expr; 3] = args.try_into().expect("three arguments");
            let max = Expr {
                kind: ExprKind::Builtin("max", vec![x, low]),
                ty: e.ty.clone(),
            };
            Expr {
                kind: ExprKind::Builtin("min", vec![max, high]),
                ty: e.ty,
            }
        }
        ExprKind::Construct(parts) => Expr {
            kind: ExprKind::Construct(parts.into_iter().map(clamp_as_min_max).collect()),
            ty: e.ty,
        },
        kind => Expr { kind, ty: e.ty },
    }
}

/// The component types an intrinsic takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Elem {
    /// Float only: other inputs convert to float.
    Float,
    /// Int, uint or float; bool becomes int.
    Number,
    /// Int or float; bool and uint become int.
    Signed,
}

/// The shape of an argument or a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// A vector of this size; 1 is a scalar.
    Vector(u8),
    /// An HLSL matrix of rows and columns.
    Matrix(u8, u8),
}

impl Shape {
    fn of(ty: &Type) -> Option<Shape> {
        match ty {
            Type::Scalar(_) => Some(Shape::Vector(1)),
            Type::Vector(_, n) => Some(Shape::Vector(*n)),
            Type::Matrix { rows, cols } => Some(Shape::Matrix(*rows, *cols)),
            _ => None,
        }
    }

    fn ty(self, scalar: Scalar) -> Type {
        match self {
            Shape::Vector(n) => Type::vector(scalar, n),
            Shape::Matrix(rows, cols) => Type::Matrix { rows, cols },
        }
    }
}

/// The intrinsics that map one to one to a component-wise WGSL built-in: the
/// WGSL name, the component types, and whether the first input takes `abs()`
/// under `abs_domain_math`.
fn componentwise(name: &str, arity: usize) -> Option<(&'static str, Elem, bool)> {
    use Elem::{Float, Number};
    Some(match (name, arity) {
        ("abs", 1) => ("abs", Number, false),
        ("saturate", 1) => ("saturate", Float, false),
        ("frac", 1) => ("fract", Float, false),
        ("rsqrt", 1) => ("inverseSqrt", Float, true),
        ("sqrt", 1) => ("sqrt", Float, true),
        ("log", 1) => ("log", Float, true),
        ("log2", 1) => ("log2", Float, true),
        ("exp", 1) => ("exp", Float, false),
        ("exp2", 1) => ("exp2", Float, false),
        ("ddx", 1) => ("dpdx", Float, false),
        ("ddy", 1) => ("dpdy", Float, false),
        ("fwidth", 1) => ("fwidth", Float, false),
        ("sin", 1) => ("sin", Float, false),
        ("cos", 1) => ("cos", Float, false),
        ("tan", 1) => ("tan", Float, false),
        ("asin", 1) => ("asin", Float, false),
        ("acos", 1) => ("acos", Float, false),
        ("atan", 1) => ("atan", Float, false),
        ("floor", 1) => ("floor", Float, false),
        ("ceil", 1) => ("ceil", Float, false),
        // WGSL rounds halfway cases to even, as the HLSL page says. fxc for
        // ps_2_0 and ps_3_0 has no round instruction and may round .5 up.
        ("round", 1) => ("round", Float, false),
        ("trunc", 1) => ("trunc", Float, false),
        ("degrees", 1) => ("degrees", Float, false),
        ("radians", 1) => ("radians", Float, false),
        ("atan2", 2) => ("atan2", Float, false),
        ("step", 2) => ("step", Float, false),
        ("min", 2) => ("min", Number, false),
        ("max", 2) => ("max", Number, false),
        ("lerp", 3) => ("mix", Float, false),
        ("smoothstep", 3) => ("smoothstep", Float, false),
        ("clamp", 3) => ("clamp", Number, false),
        _ => return None,
    })
}

fn lower_componentwise(
    cx: &mut dyn Lowerer,
    name: &str,
    wgsl: &'static str,
    elem: Elem,
    abs_input: bool,
    args: Vec<IntrinsicArg>,
    span: Span,
) -> Option<Expr> {
    let (shape, scalar) = common(cx, name, &args, elem, span)?;
    let values = fit(cx, args, shape, scalar);
    let abs = abs_input && cx.dialect().abs_domain_math;
    Some(apply(cx, shape, scalar, values, &mut |_, mut parts, ty| {
        if abs {
            abs_first(&mut parts);
        }
        call(wgsl, parts, ty.clone())
    }))
}

/// The common shape and component type of `args`, as HLSL converts them: scalars
/// splat, vectors truncate to the shortest, and a matrix makes the result float.
fn common(
    cx: &mut dyn Lowerer,
    name: &str,
    args: &[IntrinsicArg],
    elem: Elem,
    span: Span,
) -> Option<(Shape, Scalar)> {
    let mut size = None::<u8>;
    let mut matrix = None::<(u8, u8)>;
    for a in args {
        match Shape::of(&a.value.ty) {
            Some(Shape::Vector(1)) => {}
            Some(Shape::Vector(n)) => size = Some(size.map_or(n, |s| s.min(n))),
            Some(Shape::Matrix(r, c)) => {
                matrix = Some(matrix.map_or((r, c), |(mr, mc)| (mr.min(r), mc.min(c))));
            }
            None => {
                cx.error(
                    a.span,
                    format!("`{name}` takes scalars, vectors and matrices"),
                );
                return None;
            }
        }
    }
    let shape = match (matrix, size) {
        (Some(_), Some(_)) => {
            cx.error(span, format!("`{name}` cannot mix vectors and matrices"));
            return None;
        }
        (Some((rows, cols)), None) => return Some((Shape::Matrix(rows, cols), Scalar::Float)),
        (None, size) => Shape::Vector(size.unwrap_or(1)),
    };
    let widest = args.iter().filter_map(|a| a.value.ty.scalar()).max();
    let scalar = match (elem, widest) {
        (Elem::Float, _) | (_, None) | (Elem::Signed, Some(Scalar::Float)) => Scalar::Float,
        (Elem::Number, Some(s)) => s.max(Scalar::Int),
        (Elem::Signed, Some(_)) => Scalar::Int,
    };
    Some((shape, scalar))
}

/// Coerces every argument to `shape`. Under a matrix shape a scalar stays a float
/// scalar, and `apply` splats it per column.
fn fit(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>, shape: Shape, scalar: Scalar) -> Vec<Expr> {
    args.into_iter()
        .map(|a| {
            let to = match shape {
                Shape::Matrix(..) if a.value.ty.size() == Some(1) => Type::FLOAT,
                _ => shape.ty(scalar),
            };
            cx.coerce(a.value, &to, a.span)
        })
        .collect()
}

/// Builds one result from parts that all have the given type.
type Build<'a> = dyn FnMut(&mut dyn Lowerer, Vec<Expr>, &Type) -> Expr + 'a;

/// Builds the result with `f`, once for a scalar or vector shape and once per
/// WGSL column for a matrix shape. The parts that `f` gets all have its type.
fn apply(
    cx: &mut dyn Lowerer,
    shape: Shape,
    scalar: Scalar,
    args: Vec<Expr>,
    f: &mut Build<'_>,
) -> Expr {
    let (rows, cols) = match shape {
        Shape::Vector(n) => return f(cx, args, &Type::vector(scalar, n)),
        Shape::Matrix(rows, cols) => (rows, cols),
    };
    let args: Vec<Expr> = args.into_iter().map(|a| share(cx, a)).collect();
    let column_ty = Type::Vector(Scalar::Float, cols);
    let mut columns = Vec::new();
    for i in 0..rows {
        let parts = args
            .iter()
            .map(|a| match a.ty {
                Type::Matrix { .. } => column(a.clone(), i, cols),
                _ => splat(a.clone(), &column_ty),
            })
            .collect();
        columns.push(f(cx, parts, &column_ty));
    }
    Expr {
        kind: ExprKind::Construct(columns),
        ty: Type::Matrix { rows, cols },
    }
}

/// Checks that the arguments are scalars or vectors, then coerces them to their
/// common shape. Answers the size, the component type and the values.
fn vectors(
    cx: &mut dyn Lowerer,
    name: &str,
    args: Vec<IntrinsicArg>,
    elem: Elem,
    allow_scalar: bool,
    span: Span,
) -> Option<(u8, Scalar, Vec<Expr>)> {
    let (shape, scalar) = common(cx, name, &args, elem, span)?;
    let size = match shape {
        Shape::Vector(n) if n > 1 || allow_scalar => n,
        Shape::Vector(_) => {
            cx.error(span, format!("`{name}` takes a vector"));
            return None;
        }
        Shape::Matrix(..) => {
            cx.error(span, format!("`{name}` does not take a matrix"));
            return None;
        }
    };
    let values = fit(cx, args, shape, scalar);
    Some((size, scalar, values))
}

/// `mul` after the matrix model: HLSL rows are WGSL columns, so the WGSL operands
/// of a matrix product swap.
fn mul(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>, span: Span) -> Option<Expr> {
    let shapes = [Shape::of(&args[0].value.ty), Shape::of(&args[1].value.ty)];
    let (a, b) = match shapes {
        [Some(Shape::Vector(1)), Some(_)] | [Some(_), Some(Shape::Vector(1))] => {
            let (shape, scalar) = common(cx, "mul", &args, Elem::Number, span)?;
            let values = fit(cx, args, shape, scalar);
            return Some(apply(cx, shape, scalar, values, &mut |_, mut p, ty| {
                let y = p.pop().unwrap();
                binary(BinaryOp::Mul, p.pop().unwrap(), y, ty.clone())
            }));
        }
        [Some(a), Some(b)] => (a, b),
        _ => {
            let bad = if shapes[0].is_none() {
                &args[0]
            } else {
                &args[1]
            };
            cx.error(bad.span, "`mul` takes scalars, vectors and matrices".into());
            return None;
        }
    };
    if let (Shape::Vector(_), Shape::Vector(_)) = (a, b) {
        return dot(cx, args, span);
    }
    let mut args = args.into_iter();
    let (x, y) = (args.next().unwrap(), args.next().unwrap());
    let float = |n| Type::Vector(Scalar::Float, n);
    let matrix = |rows, cols| Type::Matrix { rows, cols };
    Some(match (a, b) {
        (Shape::Vector(n), Shape::Matrix(rows, cols)) => {
            let k = n.min(rows);
            let v = cx.coerce(x.value, &float(k), x.span);
            let m = cx.coerce(y.value, &matrix(k, cols), y.span);
            binary(BinaryOp::MatMul, m, v, float(cols))
        }
        (Shape::Matrix(rows, cols), Shape::Vector(n)) => {
            let k = n.min(cols);
            let m = cx.coerce(x.value, &matrix(rows, k), x.span);
            let v = cx.coerce(y.value, &float(k), y.span);
            binary(BinaryOp::MatMul, v, m, float(rows))
        }
        (Shape::Matrix(rows, k1), Shape::Matrix(k2, cols)) => {
            let k = k1.min(k2);
            let l = cx.coerce(x.value, &matrix(rows, k), x.span);
            let r = cx.coerce(y.value, &matrix(k, cols), y.span);
            binary(BinaryOp::MatMul, r, l, matrix(rows, cols))
        }
        (Shape::Vector(_), Shape::Vector(_)) => unreachable!("handled as a dot product"),
    })
}

fn dot(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>, span: Span) -> Option<Expr> {
    let (size, scalar, mut values) = vectors(cx, "dot", args, Elem::Number, true, span)?;
    let ty = Type::Scalar(scalar);
    let b = values.pop().unwrap();
    let a = values.pop().unwrap();
    Some(if size == 1 {
        binary(BinaryOp::Mul, a, b, ty)
    } else {
        call("dot", vec![a, b], ty)
    })
}

/// `length` and `distance` take scalars too; `normalize` and `reflect` take
/// vectors only.
fn geometric(
    cx: &mut dyn Lowerer,
    name: &str,
    args: Vec<IntrinsicArg>,
    span: Span,
) -> Option<Expr> {
    let (wgsl, to_scalar) = match name {
        "length" => ("length", true),
        "distance" => ("distance", true),
        "normalize" => ("normalize", false),
        _ => ("reflect", false),
    };
    let (size, scalar, values) = vectors(cx, name, args, Elem::Float, to_scalar, span)?;
    let ty = if to_scalar {
        Type::FLOAT
    } else {
        Type::vector(scalar, size)
    };
    Some(call(wgsl, values, ty))
}

fn cross(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>) -> Expr {
    let float3 = Type::Vector(Scalar::Float, 3);
    let values = args
        .into_iter()
        .map(|a| cx.coerce(a.value, &float3, a.span))
        .collect();
    call("cross", values, float3)
}

/// `pow`. A literal whole exponent from 1 to 4 becomes products, which keep the
/// sign of the base as fxc does. Otherwise the base takes `abs()` under
/// `abs_domain_math`, like the D3D9 `pow` instruction.
fn pow(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>, span: Span) -> Option<Expr> {
    let power = small_power(&args[1].value);
    let (shape, scalar) = common(cx, "pow", &args, Elem::Float, span)?;
    let mut values = fit(cx, args, shape, scalar);
    if let Some(n) = power {
        values.pop();
        return Some(apply(cx, shape, scalar, values, &mut |cx, mut p, ty| {
            let base = share(cx, p.pop().unwrap());
            (1..n).fold(base.clone(), |acc, _| {
                binary(BinaryOp::Mul, acc, base.clone(), ty.clone())
            })
        }));
    }
    let abs = cx.dialect().abs_domain_math;
    Some(apply(cx, shape, scalar, values, &mut |_, mut p, ty| {
        if abs {
            abs_first(&mut p);
        }
        call("pow", p, ty.clone())
    }))
}

/// The exponent when it is a literal whole number from 1 to 4.
fn small_power(e: &Expr) -> Option<u32> {
    let n = match e.kind {
        ExprKind::Literal(Literal::Int(n)) => i64::from(n),
        ExprKind::Literal(Literal::Uint(n)) => i64::from(n),
        ExprKind::Literal(Literal::Float(f)) if f.fract() == 0.0 && f.abs() <= 4.0 => f as i64,
        _ => return None,
    };
    (1..=4).contains(&n).then_some(n as u32)
}

/// `fmod` is the WGSL `%` on floats: `x - y * trunc(x / y)`, with the sign of x.
fn fmod(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>, span: Span) -> Option<Expr> {
    let (shape, scalar) = common(cx, "fmod", &args, Elem::Float, span)?;
    let values = fit(cx, args, shape, scalar);
    Some(apply(cx, shape, scalar, values, &mut |_, mut p, ty| {
        let y = p.pop().unwrap();
        binary(BinaryOp::Rem, p.pop().unwrap(), y, ty.clone())
    }))
}

/// WGSL has no `log10`: it is `log2(x) * log10(2)`.
fn log10(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>, span: Span) -> Option<Expr> {
    let (shape, scalar) = common(cx, "log10", &args, Elem::Float, span)?;
    let values = fit(cx, args, shape, scalar);
    let abs = cx.dialect().abs_domain_math;
    Some(apply(cx, shape, scalar, values, &mut |_, mut p, ty| {
        if abs {
            abs_first(&mut p);
        }
        let factor = splat(Expr::float(std::f32::consts::LOG10_2), ty);
        binary(
            BinaryOp::Mul,
            call("log2", p, ty.clone()),
            factor,
            ty.clone(),
        )
    }))
}

/// HLSL `sign` returns int components for every input; WGSL keeps the input type.
fn sign(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>, span: Span) -> Option<Expr> {
    let (size, scalar, values) = vectors(cx, "sign", args, Elem::Signed, true, span)?;
    let ty = Type::vector(scalar, size);
    let result = call("sign", values, ty.clone());
    Some(if scalar == Scalar::Int {
        result
    } else {
        Expr {
            kind: ExprKind::Convert(Box::new(result)),
            ty: ty.with_scalar(Scalar::Int),
        }
    })
}

fn matrix_function(cx: &mut dyn Lowerer, name: &str, args: Vec<IntrinsicArg>) -> Option<Expr> {
    let a = args.into_iter().next().unwrap();
    let (wgsl, ty) = match (name, &a.value.ty) {
        ("determinant", Type::Matrix { rows, cols }) if rows == cols => {
            ("determinant", Type::FLOAT)
        }
        ("transpose", Type::Matrix { rows, cols }) => (
            "transpose",
            Type::Matrix {
                rows: *cols,
                cols: *rows,
            },
        ),
        ("determinant", _) => {
            cx.error(a.span, "`determinant` takes a square matrix".into());
            return None;
        }
        _ => {
            cx.error(a.span, format!("`{name}` takes a matrix"));
            return None;
        }
    };
    Some(call(wgsl, vec![a.value], ty))
}

/// `all` and `any` take numbers in HLSL and bools in WGSL, so a number is first
/// compared with zero.
fn all_any(cx: &mut dyn Lowerer, name: &str, args: Vec<IntrinsicArg>) -> Option<Expr> {
    let a = args.into_iter().next().unwrap();
    if !a.value.ty.is_numeric() {
        cx.error(
            a.span,
            format!("`{name}` takes scalars, vectors and matrices"),
        );
        return None;
    }
    let (wgsl, join) = if name == "all" {
        ("all", BinaryOp::And)
    } else {
        ("any", BinaryOp::Or)
    };
    Some(reduce(cx, wgsl, join, BinaryOp::Ne, a.value))
}

/// Tests every component of `value` against zero with `cmp` (a bool is its own
/// test) and joins the answers with the WGSL `all` or `any`. Matrix columns are
/// joined with `join`.
fn reduce(
    cx: &mut dyn Lowerer,
    wgsl: &'static str,
    join: BinaryOp,
    cmp: BinaryOp,
    value: Expr,
) -> Expr {
    if let Type::Matrix { rows, cols } = value.ty {
        let m = share(cx, value);
        let mut result = None;
        for i in 0..rows {
            let test = reduce(cx, wgsl, join, cmp, column(m.clone(), i, cols));
            result = Some(match result {
                None => test,
                Some(acc) => binary(join, acc, test, Type::BOOL),
            });
        }
        return result.unwrap();
    }
    let ty = value.ty.clone();
    let scalar = ty.scalar().unwrap_or(Scalar::Float);
    let test = if scalar == Scalar::Bool {
        value
    } else {
        let zero = splat(zero(scalar), &ty);
        binary(cmp, value, zero, ty.with_scalar(Scalar::Bool))
    };
    if matches!(ty, Type::Vector(..)) {
        call(wgsl, vec![test], Type::BOOL)
    } else {
        test
    }
}

/// WGSL has no `isnan` or `isinf`: `x != x` and `abs(x) > f32::MAX`. Under the
/// WGSL finite-math assumption both may always be false.
fn isnan_isinf(
    cx: &mut dyn Lowerer,
    name: &str,
    args: Vec<IntrinsicArg>,
    span: Span,
) -> Option<Expr> {
    let (size, _, mut values) = vectors(cx, name, args, Elem::Float, true, span)?;
    cx.warn(
        span,
        format!("`{name}` may always be false: WGSL may assume that no value is NaN or infinite"),
    );
    let ty = Type::vector(Scalar::Float, size);
    let bools = Type::vector(Scalar::Bool, size);
    let x = values.pop().unwrap();
    Some(if name == "isnan" {
        let x = share(cx, x);
        binary(BinaryOp::Ne, x.clone(), x, bools)
    } else {
        let max = splat(Expr::float(f32::MAX), &ty);
        binary(BinaryOp::Gt, call("abs", vec![x], ty), max, bools)
    })
}

/// `lit(n_dot_l, n_dot_h, m)`: (1, diffuse, specular, 1) from the HLSL page.
fn lit(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>) -> Expr {
    let mut values: Vec<Expr> = args
        .into_iter()
        .map(|a| cx.coerce(a.value, &Type::FLOAT, a.span))
        .collect();
    let m = values.pop().unwrap();
    let n_dot_h = share(cx, values.pop().unwrap());
    let n_dot_l = share(cx, values.pop().unwrap());
    let below = |e: &Expr| binary(BinaryOp::Lt, e.clone(), Expr::float(0.0), Type::BOOL);
    let diffuse = select(below(&n_dot_l), Expr::float(0.0), n_dot_l.clone());
    let either = binary(BinaryOp::Or, below(&n_dot_l), below(&n_dot_h), Type::BOOL);
    let specular = select(
        either,
        Expr::float(0.0),
        call("pow", vec![n_dot_h, m], Type::FLOAT),
    );
    Expr {
        kind: ExprKind::Construct(vec![Expr::float(1.0), diffuse, specular, Expr::float(1.0)]),
        ty: Type::Vector(Scalar::Float, 4),
    }
}

/// The place of an `out` argument, or an error.
fn out_place(cx: &mut dyn Lowerer, name: &str, arg: &IntrinsicArg) -> Option<Place> {
    if arg.place.is_none() {
        cx.error(
            arg.span,
            format!("this `out` argument of `{name}` must be assignable"),
        );
    }
    arg.place.clone()
}

/// The float value of the first argument of `modf` or `sincos`.
fn float_input(cx: &mut dyn Lowerer, name: &str, arg: IntrinsicArg, span: Span) -> Option<Expr> {
    let (_, _, mut values) = vectors(cx, name, vec![arg], Elem::Float, true, span)?;
    values.pop()
}

/// `modf(x, out ip)`. The IR has no field of a built-in struct, so the parts are
/// `trunc(x)` and `x - trunc(x)`, as WGSL `modf` defines them.
fn modf(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>, span: Span) -> Option<Expr> {
    let mut args = args.into_iter();
    let (x, ip) = (args.next().unwrap(), args.next().unwrap());
    let place = out_place(cx, "modf", &ip)?;
    let x = float_input(cx, "modf", x, span)?;
    let x = cx.temp(x);
    let whole = cx.temp(call("trunc", vec![x.clone()], x.ty.clone()));
    let stored = cx.coerce(whole.clone(), &ip.value.ty, ip.span);
    cx.emit(Stmt::Assign(place, stored));
    let ty = x.ty.clone();
    Some(binary(BinaryOp::Sub, x, whole, ty))
}

/// `sincos(x, out s, out c)` assigns `sin(x)` and `cos(x)`.
fn sincos(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>, span: Span) -> Option<Expr> {
    let mut args = args.into_iter();
    let (x, s, c) = (
        args.next().unwrap(),
        args.next().unwrap(),
        args.next().unwrap(),
    );
    let sin_place = out_place(cx, "sincos", &s);
    let cos_place = out_place(cx, "sincos", &c);
    let (sin_place, cos_place) = (sin_place?, cos_place?);
    let x = float_input(cx, "sincos", x, span)?;
    let x = cx.temp(x);
    for (wgsl, place, out) in [("sin", sin_place, s), ("cos", cos_place, c)] {
        let value = call(wgsl, vec![x.clone()], x.ty.clone());
        let value = cx.coerce(value, &out.value.ty, out.span);
        cx.emit(Stmt::Assign(place, value));
    }
    Some(void())
}

/// `clip(x)` discards the fragment when any component of x is below zero.
fn clip(cx: &mut dyn Lowerer, args: Vec<IntrinsicArg>, span: Span) -> Option<Expr> {
    let (shape, scalar) = common(cx, "clip", &args, Elem::Float, span)?;
    let x = fit(cx, args, shape, scalar).pop().unwrap();
    let cond = reduce(cx, "any", BinaryOp::Or, BinaryOp::Lt, x);
    cx.emit(Stmt::If(cond, vec![Stmt::Discard], Vec::new()));
    Some(void())
}

/// How a `tex*` function samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// `textureSample` with the coordinates.
    Plain,
    /// `textureSampleGrad` with two more arguments.
    Grad,
    /// `textureSampleLevel` with the level in `t.w`.
    Level,
    /// `textureSampleBias` with the bias in `t.w`.
    Bias,
    /// `textureSample` with the coordinates divided by `t.w`.
    Proj,
}

fn texture_function(name: &str, arity: usize) -> Option<(SamplerDim, Form)> {
    use SamplerDim::{Cube, D1, D2, D3};
    Some(match (name, arity) {
        ("tex1D", 2) => (D1, Form::Plain),
        ("tex2D", 2) => (D2, Form::Plain),
        ("tex3D", 2) => (D3, Form::Plain),
        ("texCUBE", 2) => (Cube, Form::Plain),
        ("tex2D", 4) | ("tex2Dgrad", 4) => (D2, Form::Grad),
        ("tex2Dlod", 2) => (D2, Form::Level),
        ("tex3Dlod", 2) => (D3, Form::Level),
        ("texCUBElod", 2) => (Cube, Form::Level),
        ("tex2Dbias", 2) => (D2, Form::Bias),
        ("tex3Dbias", 2) => (D3, Form::Bias),
        ("texCUBEbias", 2) => (Cube, Form::Bias),
        ("tex2Dproj", 2) => (D2, Form::Proj),
        _ => return None,
    })
}

fn sample(
    cx: &mut dyn Lowerer,
    name: &str,
    dim: SamplerDim,
    form: Form,
    args: Vec<IntrinsicArg>,
) -> Lowered {
    let mut args = args.into_iter();
    let s = args.next().unwrap();
    let sampler = match (&s.value.kind, &s.value.ty) {
        (ExprKind::Var(Root::Global(index)), Type::Sampler(_)) => *index,
        _ => {
            cx.error(
                s.span,
                format!("the first argument of `{name}` must be a sampler"),
            );
            return Lowered::Failed;
        }
    };
    if !cx.use_sampler(sampler, dim, s.span) {
        return Lowered::Failed;
    }
    let size = match dim {
        SamplerDim::D1 => 1,
        SamplerDim::D2 => 2,
        _ => 3,
    };
    let coords_ty = Type::vector(Scalar::Float, size);
    let mut next = |cx: &mut dyn Lowerer, ty: &Type| {
        let a = args.next().unwrap();
        cx.coerce(a.value, ty, a.span)
    };
    let (coords, kind) = match form {
        Form::Plain => (next(cx, &coords_ty), Sample::Plain),
        Form::Grad => {
            let coords = next(cx, &coords_ty);
            let dx = next(cx, &coords_ty);
            let dy = next(cx, &coords_ty);
            (coords, Sample::Grad(Box::new(dx), Box::new(dy)))
        }
        Form::Level | Form::Bias | Form::Proj => {
            let t = next(cx, &Type::Vector(Scalar::Float, 4));
            let t = share(cx, t);
            let xyz = swizzle(t.clone(), (0..size).collect(), coords_ty.clone());
            let w = swizzle(t, vec![3], Type::FLOAT);
            match form {
                Form::Level => (xyz, Sample::Level(Box::new(w))),
                Form::Bias => (xyz, Sample::Bias(Box::new(w))),
                _ => {
                    let w = splat(w, &coords_ty);
                    (binary(BinaryOp::Div, xyz, w, coords_ty), Sample::Plain)
                }
            }
        }
    };
    Lowered::Done(Expr {
        kind: ExprKind::Sample {
            sampler,
            coords: Box::new(coords),
            kind,
        },
        ty: Type::Vector(Scalar::Float, 4),
    })
}

/// The value itself when it is cheap to read twice, otherwise a temp.
fn share(cx: &mut dyn Lowerer, value: Expr) -> Expr {
    match value.kind {
        ExprKind::Var(_) | ExprKind::Literal(_) => value,
        _ => cx.temp(value),
    }
}

/// Wraps the first part in `abs()`.
fn abs_first(parts: &mut [Expr]) {
    let first = parts[0].clone();
    let ty = first.ty.clone();
    parts[0] = call("abs", vec![first], ty);
}

fn call(wgsl: &'static str, args: Vec<Expr>, ty: Type) -> Expr {
    Expr {
        kind: ExprKind::Builtin(wgsl, args),
        ty,
    }
}

fn binary(op: BinaryOp, a: Expr, b: Expr, ty: Type) -> Expr {
    Expr {
        kind: ExprKind::Binary(op, Box::new(a), Box::new(b)),
        ty,
    }
}

fn select(cond: Expr, accept: Expr, reject: Expr) -> Expr {
    let ty = accept.ty.clone();
    Expr {
        kind: ExprKind::Select {
            cond: Box::new(cond),
            accept: Box::new(accept),
            reject: Box::new(reject),
        },
        ty,
    }
}

fn swizzle(value: Expr, components: Vec<u8>, ty: Type) -> Expr {
    Expr {
        kind: ExprKind::Swizzle(Box::new(value), components),
        ty,
    }
}

/// WGSL column `i` of a matrix, which is HLSL row `i`.
fn column(matrix: Expr, i: u8, cols: u8) -> Expr {
    Expr {
        kind: ExprKind::Index(Box::new(matrix), Box::new(Expr::int(i32::from(i)))),
        ty: Type::Vector(Scalar::Float, cols),
    }
}

/// A scalar repeated to `ty` when `ty` is a vector.
fn splat(value: Expr, ty: &Type) -> Expr {
    match ty {
        Type::Vector(..) => Expr {
            kind: ExprKind::Splat(Box::new(value)),
            ty: ty.clone(),
        },
        _ => value,
    }
}

fn zero(scalar: Scalar) -> Expr {
    let literal = match scalar {
        Scalar::Bool => Literal::Bool(false),
        Scalar::Int => Literal::Int(0),
        Scalar::Uint => Literal::Uint(0),
        Scalar::Float => Literal::Float(0.0),
    };
    Expr {
        kind: ExprKind::Literal(literal),
        ty: Type::Scalar(scalar),
    }
}

/// The value of a call that returns nothing: its statements are already emitted.
fn void() -> Expr {
    Expr {
        kind: ExprKind::Construct(Vec::new()),
        ty: Type::Void,
    }
}

#[cfg(test)]
mod tests {
    use wgslift_syntax::FileId;

    use super::*;
    use crate::Dialect;

    /// Temps start here, so they never clash with the locals a test passes in.
    const TEMP: usize = 100;

    struct Mock {
        dialect: Dialect,
        stmts: Vec<Stmt>,
        errors: Vec<String>,
        warnings: Vec<String>,
        locals: usize,
        samplers: Vec<(usize, SamplerDim)>,
        sampler_ok: bool,
    }

    impl Mock {
        fn legacy() -> Mock {
            Mock {
                dialect: Dialect::d3d9_legacy(),
                stmts: Vec::new(),
                errors: Vec::new(),
                warnings: Vec::new(),
                locals: TEMP,
                samplers: Vec::new(),
                sampler_ok: true,
            }
        }

        fn modern() -> Mock {
            let mut cx = Mock::legacy();
            cx.dialect.abs_domain_math = false;
            cx
        }
    }

    impl Lowerer for Mock {
        fn dialect(&self) -> &Dialect {
            &self.dialect
        }

        fn coerce(&mut self, value: Expr, to: &Type, span: Span) -> Expr {
            if &value.ty == to {
                return value;
            }
            let convert = |value: Expr| Expr {
                kind: ExprKind::Convert(Box::new(value)),
                ty: to.clone(),
            };
            match (value.ty.clone(), to) {
                (Type::Scalar(_), Type::Vector(s, _)) => {
                    let v = self.coerce(value, &Type::Scalar(*s), span);
                    splat(v, to)
                }
                (Type::Vector(s, n), _) if to.size().is_some_and(|m| m < n) => {
                    self.warn(span, "truncated".into());
                    let m = to.size().unwrap();
                    let v = swizzle(value, (0..m).collect(), Type::vector(s, m));
                    self.coerce(v, to, span)
                }
                (Type::Matrix { rows, cols }, Type::Matrix { rows: r, cols: c })
                    if *r <= rows && *c <= cols =>
                {
                    self.warn(span, "truncated".into());
                    convert(value)
                }
                (from, _) if from.size().is_some() && from.size() == to.size() => convert(value),
                _ => {
                    self.errors.push("no conversion".into());
                    convert(value)
                }
            }
        }

        fn error(&mut self, _span: Span, message: String) {
            self.errors.push(message);
        }

        fn warn(&mut self, _span: Span, message: String) {
            self.warnings.push(message);
        }

        fn temp(&mut self, value: Expr) -> Expr {
            let id = self.locals;
            self.locals += 1;
            let ty = value.ty.clone();
            self.stmts.push(Stmt::Local {
                id,
                init: Some(value),
            });
            local(id, ty)
        }

        fn emit(&mut self, stmt: Stmt) {
            self.stmts.push(stmt);
        }

        fn use_sampler(&mut self, global: usize, dim: SamplerDim, _span: Span) -> bool {
            self.samplers.push((global, dim));
            self.sampler_ok
        }
    }

    fn span() -> Span {
        Span::new(FileId(0), 0, 0)
    }

    fn local(id: usize, ty: Type) -> Expr {
        Expr {
            kind: ExprKind::Var(Root::Local(id)),
            ty,
        }
    }

    fn float(n: u8) -> Type {
        Type::vector(Scalar::Float, n)
    }

    fn int(n: u8) -> Type {
        Type::vector(Scalar::Int, n)
    }

    fn bools(n: u8) -> Type {
        Type::vector(Scalar::Bool, n)
    }

    fn mat(rows: u8, cols: u8) -> Type {
        Type::Matrix { rows, cols }
    }

    fn convert(value: Expr, ty: Type) -> Expr {
        Expr {
            kind: ExprKind::Convert(Box::new(value)),
            ty,
        }
    }

    fn construct(parts: Vec<Expr>, ty: Type) -> Expr {
        Expr {
            kind: ExprKind::Construct(parts),
            ty,
        }
    }

    fn place(id: usize) -> Place {
        Place {
            root: Root::Local(id),
            path: Vec::new(),
        }
    }

    fn arg(value: Expr) -> IntrinsicArg {
        IntrinsicArg {
            value,
            place: None,
            span: span(),
        }
    }

    fn out(id: usize, ty: Type) -> IntrinsicArg {
        IntrinsicArg {
            value: local(id, ty),
            place: Some(place(id)),
            span: span(),
        }
    }

    fn sampler(index: usize, dim: SamplerDim) -> Expr {
        Expr {
            kind: ExprKind::Var(Root::Global(index)),
            ty: Type::Sampler(dim),
        }
    }

    fn lowered(cx: &mut Mock, name: &str, args: Vec<Expr>) -> Lowered {
        lower(cx, name, args.into_iter().map(arg).collect(), span())
    }

    fn done(cx: &mut Mock, name: &str, args: Vec<Expr>) -> Expr {
        match lowered(cx, name, args) {
            Lowered::Done(e) => e,
            other => panic!("`{name}` gave {other:?}, errors {:?}", cx.errors),
        }
    }

    fn failed(cx: &mut Mock, name: &str, args: Vec<Expr>) {
        assert_eq!(lowered(cx, name, args), Lowered::Failed, "{name}");
        assert!(!cx.errors.is_empty(), "`{name}` failed without an error");
    }

    fn abs(e: Expr) -> Expr {
        let ty = e.ty.clone();
        call("abs", vec![e], ty)
    }

    fn times(a: Expr, b: Expr) -> Expr {
        let ty = a.ty.clone();
        binary(BinaryOp::Mul, a, b, ty)
    }

    #[test]
    fn unknown_names_and_argument_counts_are_not_intrinsics() {
        let mut cx = Mock::legacy();
        let x = local(0, Type::FLOAT);
        assert_eq!(
            lowered(&mut cx, "foo", vec![x.clone()]),
            Lowered::NotIntrinsic
        );
        let two = vec![x.clone(), x.clone()];
        assert_eq!(lowered(&mut cx, "sin", two), Lowered::NotIntrinsic);
        let s = sampler(0, SamplerDim::D2);
        let three = vec![s, x.clone(), x];
        assert_eq!(lowered(&mut cx, "tex2D", three), Lowered::NotIntrinsic);
        assert!(cx.errors.is_empty() && cx.stmts.is_empty());
    }

    #[test]
    fn one_argument_intrinsics_map_to_their_wgsl_names() {
        let table = [
            ("saturate", "saturate"),
            ("frac", "fract"),
            ("rsqrt", "inverseSqrt"),
            ("sqrt", "sqrt"),
            ("log", "log"),
            ("log2", "log2"),
            ("exp", "exp"),
            ("exp2", "exp2"),
            ("ddx", "dpdx"),
            ("ddy", "dpdy"),
            ("fwidth", "fwidth"),
            ("sin", "sin"),
            ("cos", "cos"),
            ("tan", "tan"),
            ("asin", "asin"),
            ("acos", "acos"),
            ("atan", "atan"),
            ("floor", "floor"),
            ("ceil", "ceil"),
            ("round", "round"),
            ("trunc", "trunc"),
            ("degrees", "degrees"),
            ("radians", "radians"),
            ("abs", "abs"),
        ];
        for (hlsl, wgsl) in table {
            let mut cx = Mock::modern();
            let x = local(0, float(2));
            let expected = call(wgsl, vec![x.clone()], float(2));
            assert_eq!(done(&mut cx, hlsl, vec![x]), expected, "{hlsl}");
        }
    }

    #[test]
    fn two_and_three_argument_intrinsics_keep_their_argument_order() {
        let (a, b, c) = (local(0, float(3)), local(1, float(3)), local(2, float(3)));
        for name in ["atan2", "step", "min", "max"] {
            let mut cx = Mock::modern();
            let expected = call(name, vec![a.clone(), b.clone()], float(3));
            assert_eq!(done(&mut cx, name, vec![a.clone(), b.clone()]), expected);
        }
        let mut cx = Mock::modern();
        let args = vec![a.clone(), b.clone(), c.clone()];
        let expected = call("smoothstep", args.clone(), float(3));
        assert_eq!(done(&mut cx, "smoothstep", args.clone()), expected);
        let expected = min_max(a.clone(), b.clone(), c.clone(), float(3));
        assert_eq!(done(&mut cx, "clamp", args), expected);
    }

    /// HLSL's clamp: `min(max(x, low), high)`.
    fn min_max(x: Expr, low: Expr, high: Expr, ty: Type) -> Expr {
        call("min", vec![call("max", vec![x, low], ty.clone()), high], ty)
    }

    #[test]
    fn lerp_lowers_to_mix_with_a_splatted_weight() {
        let mut cx = Mock::legacy();
        let (a, b, s) = (
            local(0, float(3)),
            local(1, float(3)),
            local(2, Type::FLOAT),
        );
        let expected = call(
            "mix",
            vec![a.clone(), b.clone(), splat(s.clone(), &float(3))],
            float(3),
        );
        assert_eq!(done(&mut cx, "lerp", vec![a, b, s]), expected);
    }

    #[test]
    fn vectors_of_different_sizes_truncate_to_the_shortest() {
        let mut cx = Mock::legacy();
        let (a, b) = (local(0, float(4)), local(1, float(2)));
        let expected = call(
            "min",
            vec![swizzle(a.clone(), vec![0, 1], float(2)), b.clone()],
            float(2),
        );
        assert_eq!(done(&mut cx, "min", vec![a, b]), expected);
        assert_eq!(cx.warnings, ["truncated"]);
    }

    #[test]
    fn int_arguments_stay_int_where_wgsl_takes_ints() {
        let mut cx = Mock::legacy();
        let (a, b) = (local(0, int(2)), local(1, Type::INT));
        let expected = call("max", vec![a.clone(), splat(b.clone(), &int(2))], int(2));
        assert_eq!(done(&mut cx, "max", vec![a, b]), expected);
        let flag = local(2, Type::BOOL);
        let expected = call("abs", vec![convert(flag.clone(), Type::INT)], Type::INT);
        assert_eq!(done(&mut cx, "abs", vec![flag]), expected);
    }

    #[test]
    fn mixed_int_and_float_arguments_become_float() {
        let mut cx = Mock::legacy();
        let (i, f) = (local(0, Type::INT), local(1, Type::FLOAT));
        let as_float = convert(i.clone(), Type::FLOAT);
        let expected = call("min", vec![as_float.clone(), f.clone()], Type::FLOAT);
        assert_eq!(done(&mut cx, "min", vec![i.clone(), f]), expected);
        let expected = call("sin", vec![as_float], Type::FLOAT);
        assert_eq!(done(&mut cx, "sin", vec![i]), expected);
    }

    #[test]
    fn component_wise_intrinsics_run_per_column_on_matrices() {
        let mut cx = Mock::legacy();
        let m = local(0, mat(2, 3));
        let row = |i| call("saturate", vec![column(m.clone(), i, 3)], float(3));
        let expected = construct(vec![row(0), row(1)], mat(2, 3));
        assert_eq!(done(&mut cx, "saturate", vec![m.clone()]), expected);

        let (lo, hi) = (Expr::float(0.0), Expr::float(1.0));
        let row = |i| {
            min_max(
                column(m.clone(), i, 3),
                splat(lo.clone(), &float(3)),
                splat(hi.clone(), &float(3)),
                float(3),
            )
        };
        let expected = construct(vec![row(0), row(1)], mat(2, 3));
        assert_eq!(
            done(&mut cx, "clamp", vec![m.clone(), lo.clone(), hi.clone()]),
            expected
        );
    }

    #[test]
    fn a_computed_matrix_is_read_once_through_a_temp() {
        let mut cx = Mock::legacy();
        let product = binary(
            BinaryOp::MatMul,
            local(0, mat(2, 2)),
            local(1, mat(2, 2)),
            mat(2, 2),
        );
        let result = done(&mut cx, "frac", vec![product.clone()]);
        let init = Some(product);
        assert_eq!(cx.stmts, [Stmt::Local { id: TEMP, init }]);
        let t = local(TEMP, mat(2, 2));
        let row = |i| call("fract", vec![column(t.clone(), i, 2)], float(2));
        assert_eq!(result, construct(vec![row(0), row(1)], mat(2, 2)));
    }

    #[test]
    fn mixing_vectors_and_matrices_or_passing_a_sampler_is_an_error() {
        let mut cx = Mock::legacy();
        failed(
            &mut cx,
            "max",
            vec![local(0, float(2)), local(1, mat(2, 2))],
        );
        let mut cx = Mock::legacy();
        failed(&mut cx, "sin", vec![sampler(0, SamplerDim::D2)]);
    }

    #[test]
    fn abs_domain_math_wraps_the_input_of_pow_log_sqrt_and_rsqrt() {
        let (x, y) = (local(0, float(2)), local(1, float(2)));
        for (hlsl, wgsl) in [
            ("sqrt", "sqrt"),
            ("rsqrt", "inverseSqrt"),
            ("log", "log"),
            ("log2", "log2"),
        ] {
            let mut cx = Mock::legacy();
            let expected = call(wgsl, vec![abs(x.clone())], float(2));
            assert_eq!(done(&mut cx, hlsl, vec![x.clone()]), expected, "{hlsl}");
        }
        let mut cx = Mock::legacy();
        let expected = call("pow", vec![abs(x.clone()), y.clone()], float(2));
        assert_eq!(done(&mut cx, "pow", vec![x.clone(), y]), expected);
        let factor = splat(Expr::float(std::f32::consts::LOG10_2), &float(2));
        let expected = times(call("log2", vec![abs(x.clone())], float(2)), factor);
        assert_eq!(done(&mut cx, "log10", vec![x.clone()]), expected);
        let expected = call("exp", vec![x.clone()], float(2));
        assert_eq!(done(&mut cx, "exp", vec![x]), expected);
    }

    #[test]
    fn without_abs_domain_math_inputs_pass_unchanged() {
        let mut cx = Mock::modern();
        let (x, y) = (local(0, Type::FLOAT), local(1, Type::FLOAT));
        let expected = call("pow", vec![x.clone(), y.clone()], Type::FLOAT);
        assert_eq!(done(&mut cx, "pow", vec![x.clone(), y]), expected);
        let factor = Expr::float(std::f32::consts::LOG10_2);
        let expected = times(call("log2", vec![x.clone()], Type::FLOAT), factor);
        assert_eq!(done(&mut cx, "log10", vec![x]), expected);
    }

    #[test]
    fn pow_with_a_whole_literal_exponent_from_one_to_four_multiplies() {
        for mut cx in [Mock::legacy(), Mock::modern()] {
            let x = local(0, float(3));
            let cube = times(times(x.clone(), x.clone()), x.clone());
            assert_eq!(done(&mut cx, "pow", vec![x.clone(), Expr::int(3)]), cube);
            let square = times(x.clone(), x.clone());
            assert_eq!(
                done(&mut cx, "pow", vec![x.clone(), Expr::float(2.0)]),
                square
            );
            assert_eq!(done(&mut cx, "pow", vec![x.clone(), Expr::int(1)]), x);
        }
        let mut cx = Mock::modern();
        let x = local(0, Type::FLOAT);
        for exponent in [Expr::int(5), Expr::int(0), Expr::float(2.5)] {
            let result = done(&mut cx, "pow", vec![x.clone(), exponent]);
            assert!(matches!(result.kind, ExprKind::Builtin("pow", _)));
        }
    }

    #[test]
    fn pow_reads_a_computed_base_once() {
        let mut cx = Mock::legacy();
        let base = call("sin", vec![local(0, Type::FLOAT)], Type::FLOAT);
        let result = done(&mut cx, "pow", vec![base.clone(), Expr::int(2)]);
        let t = local(TEMP, Type::FLOAT);
        assert_eq!(result, times(t.clone(), t));
        let init = Some(base);
        assert_eq!(cx.stmts, [Stmt::Local { id: TEMP, init }]);
    }

    #[test]
    fn sign_returns_int_components() {
        let mut cx = Mock::legacy();
        let x = local(0, float(3));
        let expected = convert(call("sign", vec![x.clone()], float(3)), int(3));
        assert_eq!(done(&mut cx, "sign", vec![x]), expected);
        let i = local(1, Type::INT);
        let expected = call("sign", vec![i.clone()], Type::INT);
        assert_eq!(done(&mut cx, "sign", vec![i]), expected);
        let u = local(2, Type::Scalar(Scalar::Uint));
        let expected = call("sign", vec![convert(u.clone(), Type::INT)], Type::INT);
        assert_eq!(done(&mut cx, "sign", vec![u]), expected);
    }

    #[test]
    fn fmod_is_the_wgsl_remainder() {
        let mut cx = Mock::legacy();
        let (x, y) = (local(0, float(2)), local(1, Type::FLOAT));
        let expected = binary(
            BinaryOp::Rem,
            x.clone(),
            splat(y.clone(), &float(2)),
            float(2),
        );
        assert_eq!(done(&mut cx, "fmod", vec![x, y]), expected);
    }

    #[test]
    fn geometric_intrinsics_have_their_result_types() {
        let mut cx = Mock::legacy();
        let (a, b) = (local(0, float(3)), local(1, float(3)));
        let one = || vec![a.clone()];
        let two = || vec![a.clone(), b.clone()];
        assert_eq!(
            done(&mut cx, "length", one()),
            call("length", one(), Type::FLOAT)
        );
        assert_eq!(
            done(&mut cx, "distance", two()),
            call("distance", two(), Type::FLOAT)
        );
        assert_eq!(done(&mut cx, "dot", two()), call("dot", two(), Type::FLOAT));
        assert_eq!(
            done(&mut cx, "normalize", one()),
            call("normalize", one(), float(3))
        );
        assert_eq!(
            done(&mut cx, "reflect", two()),
            call("reflect", two(), float(3))
        );
        assert_eq!(
            done(&mut cx, "cross", two()),
            call("cross", two(), float(3))
        );
        let ints = vec![local(2, int(2)), local(3, int(2))];
        assert_eq!(
            done(&mut cx, "dot", ints.clone()),
            call("dot", ints, Type::INT)
        );
        let (s, t) = (local(4, Type::FLOAT), local(5, Type::FLOAT));
        assert_eq!(
            done(&mut cx, "dot", vec![s.clone(), t.clone()]),
            times(s.clone(), t)
        );
        assert!(cx.errors.is_empty());
        assert_eq!(
            done(&mut cx, "normalize", vec![s.clone()]),
            call("sign", vec![s], Type::FLOAT)
        );
        let mut cx = Mock::legacy();
        failed(&mut cx, "length", vec![local(0, mat(3, 3))]);
    }

    #[test]
    fn cross_truncates_a_longer_vector() {
        let mut cx = Mock::legacy();
        let (a, b) = (local(0, float(4)), local(1, float(3)));
        let xyz = swizzle(a.clone(), vec![0, 1, 2], float(3));
        let expected = call("cross", vec![xyz, b.clone()], float(3));
        assert_eq!(done(&mut cx, "cross", vec![a, b]), expected);
        assert_eq!(cx.warnings, ["truncated"]);
    }

    #[test]
    fn transpose_swaps_the_shape_and_determinant_needs_a_square_matrix() {
        let mut cx = Mock::legacy();
        let m = local(0, mat(2, 3));
        let expected = call("transpose", vec![m.clone()], mat(3, 2));
        assert_eq!(done(&mut cx, "transpose", vec![m.clone()]), expected);
        let square = local(1, mat(3, 3));
        let expected = call("determinant", vec![square.clone()], Type::FLOAT);
        assert_eq!(done(&mut cx, "determinant", vec![square]), expected);
        failed(&mut cx, "determinant", vec![m]);
        let mut cx = Mock::legacy();
        failed(&mut cx, "transpose", vec![local(0, float(3))]);
    }

    #[test]
    fn all_and_any_compare_numbers_with_zero() {
        let mut cx = Mock::legacy();
        let v = local(0, float(3));
        let ne = binary(
            BinaryOp::Ne,
            v.clone(),
            splat(Expr::float(0.0), &float(3)),
            bools(3),
        );
        assert_eq!(
            done(&mut cx, "any", vec![v]),
            call("any", vec![ne], Type::BOOL)
        );
        let b = local(1, bools(3));
        assert_eq!(
            done(&mut cx, "all", vec![b.clone()]),
            call("all", vec![b], Type::BOOL)
        );
        let i = local(2, Type::INT);
        let expected = binary(BinaryOp::Ne, i.clone(), Expr::int(0), Type::BOOL);
        assert_eq!(done(&mut cx, "all", vec![i]), expected);
        let m = local(3, mat(2, 2));
        let row = |k| {
            let zero = splat(Expr::float(0.0), &float(2));
            let ne = binary(BinaryOp::Ne, column(m.clone(), k, 2), zero, bools(2));
            call("all", vec![ne], Type::BOOL)
        };
        let expected = binary(BinaryOp::And, row(0), row(1), Type::BOOL);
        assert_eq!(done(&mut cx, "all", vec![m.clone()]), expected);
    }

    #[test]
    fn isnan_and_isinf_become_comparisons_with_a_warning() {
        let mut cx = Mock::legacy();
        let x = local(0, float(2));
        let expected = binary(BinaryOp::Ne, x.clone(), x.clone(), bools(2));
        assert_eq!(done(&mut cx, "isnan", vec![x.clone()]), expected);
        let max = splat(Expr::float(f32::MAX), &float(2));
        let expected = binary(BinaryOp::Gt, abs(x.clone()), max, bools(2));
        assert_eq!(done(&mut cx, "isinf", vec![x]), expected);
        assert_eq!(cx.warnings.len(), 2);
        let mut cx = Mock::legacy();
        failed(&mut cx, "isnan", vec![local(0, mat(2, 2))]);
    }

    #[test]
    fn lit_builds_the_lighting_vector() {
        let mut cx = Mock::legacy();
        let (l, h, m) = (
            local(0, Type::FLOAT),
            local(1, Type::FLOAT),
            local(2, Type::FLOAT),
        );
        let below = |e: &Expr| binary(BinaryOp::Lt, e.clone(), Expr::float(0.0), Type::BOOL);
        let diffuse = select(below(&l), Expr::float(0.0), l.clone());
        let either = binary(BinaryOp::Or, below(&l), below(&h), Type::BOOL);
        let power = call("pow", vec![h.clone(), m.clone()], Type::FLOAT);
        let specular = select(either, Expr::float(0.0), power);
        let one = Expr::float(1.0);
        let expected = construct(vec![one.clone(), diffuse, specular, one], float(4));
        assert_eq!(done(&mut cx, "lit", vec![l, h, m]), expected);
    }

    #[test]
    fn mul_with_a_scalar_is_a_product() {
        let mut cx = Mock::legacy();
        let (s, t) = (local(0, Type::FLOAT), local(1, Type::FLOAT));
        assert_eq!(
            done(&mut cx, "mul", vec![s.clone(), t.clone()]),
            times(s.clone(), t)
        );
        let v = local(2, float(3));
        let s3 = splat(s.clone(), &float(3));
        assert_eq!(
            done(&mut cx, "mul", vec![s.clone(), v.clone()]),
            times(s3.clone(), v.clone())
        );
        assert_eq!(
            done(&mut cx, "mul", vec![v.clone(), s.clone()]),
            times(v, s3)
        );
        let m = local(3, mat(2, 2));
        let row = |i| times(splat(s.clone(), &float(2)), column(m.clone(), i, 2));
        let expected = construct(vec![row(0), row(1)], mat(2, 2));
        assert_eq!(done(&mut cx, "mul", vec![s.clone(), m.clone()]), expected);
    }

    #[test]
    fn mul_of_two_vectors_is_a_dot_product() {
        let mut cx = Mock::legacy();
        let (a, b) = (local(0, float(4)), local(1, float(4)));
        let expected = call("dot", vec![a.clone(), b.clone()], Type::FLOAT);
        assert_eq!(done(&mut cx, "mul", vec![a, b]), expected);
    }

    #[test]
    fn mul_with_matrices_swaps_the_wgsl_operands() {
        let mut cx = Mock::legacy();
        let v = local(0, float(4));
        let m = local(1, mat(4, 3));
        let expected = binary(BinaryOp::MatMul, m.clone(), v.clone(), float(3));
        assert_eq!(done(&mut cx, "mul", vec![v.clone(), m]), expected);
        let m = local(2, mat(3, 4));
        let expected = binary(BinaryOp::MatMul, v.clone(), m.clone(), float(3));
        assert_eq!(done(&mut cx, "mul", vec![m, v]), expected);
        let (a, b) = (local(3, mat(2, 3)), local(4, mat(3, 4)));
        let expected = binary(BinaryOp::MatMul, b.clone(), a.clone(), mat(2, 4));
        assert_eq!(done(&mut cx, "mul", vec![a, b]), expected);
        assert!(cx.warnings.is_empty() && cx.errors.is_empty());
    }

    #[test]
    fn mul_truncates_a_long_vector_or_a_large_matrix() {
        let mut cx = Mock::legacy();
        let v = local(0, float(4));
        let m = local(1, mat(3, 3));
        let xyz = swizzle(v.clone(), vec![0, 1, 2], float(3));
        let expected = binary(BinaryOp::MatMul, m.clone(), xyz, float(3));
        assert_eq!(done(&mut cx, "mul", vec![v, m]), expected);

        let v = local(2, float(3));
        let big = local(3, mat(4, 4));
        let cut = convert(big.clone(), mat(3, 4));
        let expected = binary(BinaryOp::MatMul, cut, v.clone(), float(4));
        assert_eq!(done(&mut cx, "mul", vec![v.clone(), big]), expected);

        let int_v = local(4, int(2));
        let m = local(5, mat(2, 2));
        let as_float = convert(int_v.clone(), float(2));
        let expected = binary(BinaryOp::MatMul, as_float, m.clone(), float(2));
        assert_eq!(done(&mut cx, "mul", vec![m, int_v]), expected);
        assert_eq!(cx.warnings, ["truncated", "truncated"]);
        failed(&mut cx, "mul", vec![sampler(0, SamplerDim::D2), v]);
    }

    #[test]
    fn modf_assigns_the_whole_part_and_returns_the_fraction() {
        let mut cx = Mock::legacy();
        let x = local(0, float(2));
        let args = vec![arg(x.clone()), out(1, float(2))];
        let Lowered::Done(result) = lower(&mut cx, "modf", args, span()) else {
            panic!("modf failed: {:?}", cx.errors);
        };
        // x is copied first: the out place can be x itself.
        let copy = local(TEMP, float(2));
        let whole = local(TEMP + 1, float(2));
        let init = Some(call("trunc", vec![copy.clone()], float(2)));
        let expected = [
            Stmt::Local {
                id: TEMP,
                init: Some(x),
            },
            Stmt::Local { id: TEMP + 1, init },
            Stmt::Assign(place(1), whole.clone()),
        ];
        assert_eq!(cx.stmts, expected);
        assert_eq!(result, binary(BinaryOp::Sub, copy, whole, float(2)));
    }

    #[test]
    fn sincos_assigns_both_outputs_and_returns_nothing() {
        let mut cx = Mock::legacy();
        let x = local(0, Type::FLOAT);
        let args = vec![arg(x.clone()), out(1, Type::FLOAT), out(2, Type::FLOAT)];
        assert_eq!(
            lower(&mut cx, "sincos", args, span()),
            Lowered::Done(void())
        );
        let copy = local(TEMP, Type::FLOAT);
        let expected = [
            Stmt::Local {
                id: TEMP,
                init: Some(x),
            },
            Stmt::Assign(place(1), call("sin", vec![copy.clone()], Type::FLOAT)),
            Stmt::Assign(place(2), call("cos", vec![copy], Type::FLOAT)),
        ];
        assert_eq!(cx.stmts, expected);
    }

    #[test]
    fn an_out_argument_without_a_place_is_an_error() {
        let mut cx = Mock::legacy();
        let x = local(0, Type::FLOAT);
        let args = vec![arg(x.clone()), arg(Expr::float(1.0))];
        assert_eq!(lower(&mut cx, "modf", args, span()), Lowered::Failed);
        let args = vec![arg(x), out(1, Type::FLOAT), arg(Expr::float(1.0))];
        assert_eq!(lower(&mut cx, "sincos", args, span()), Lowered::Failed);
        assert_eq!(cx.errors.len(), 2);
        assert!(cx.stmts.is_empty());
    }

    #[test]
    fn clip_discards_when_any_component_is_below_zero() {
        let mut cx = Mock::legacy();
        let x = local(0, float(3));
        assert_eq!(done(&mut cx, "clip", vec![x.clone()]), void());
        let below = binary(
            BinaryOp::Lt,
            x,
            splat(Expr::float(0.0), &float(3)),
            bools(3),
        );
        let cond = call("any", vec![below], Type::BOOL);
        assert_eq!(cx.stmts, [Stmt::If(cond, vec![Stmt::Discard], Vec::new())]);

        let mut cx = Mock::legacy();
        let s = local(1, Type::FLOAT);
        done(&mut cx, "clip", vec![s.clone()]);
        let cond = binary(BinaryOp::Lt, s, Expr::float(0.0), Type::BOOL);
        assert_eq!(cx.stmts, [Stmt::If(cond, vec![Stmt::Discard], Vec::new())]);
    }

    fn sampled(sampler: usize, coords: Expr, kind: Sample) -> Expr {
        Expr {
            kind: ExprKind::Sample {
                sampler,
                coords: Box::new(coords),
                kind,
            },
            ty: float(4),
        }
    }

    #[test]
    fn plain_tex_functions_sample_with_their_dimension() {
        let table = [
            ("tex1D", SamplerDim::D1, 1),
            ("tex2D", SamplerDim::D2, 2),
            ("tex3D", SamplerDim::D3, 3),
            ("texCUBE", SamplerDim::Cube, 3),
        ];
        for (name, dim, size) in table {
            let mut cx = Mock::legacy();
            let t = local(0, float(size));
            let expected = sampled(2, t.clone(), Sample::Plain);
            assert_eq!(
                done(&mut cx, name, vec![sampler(2, SamplerDim::Any), t]),
                expected
            );
            assert_eq!(cx.samplers, [(2, dim)], "{name}");
        }
    }

    #[test]
    fn tex2d_with_gradients_and_tex2dgrad_sample_with_gradients() {
        for name in ["tex2D", "tex2Dgrad"] {
            let mut cx = Mock::legacy();
            let (t, dx, dy) = (local(0, float(2)), local(1, float(2)), local(2, float(2)));
            let expected = sampled(
                0,
                t.clone(),
                Sample::Grad(Box::new(dx.clone()), Box::new(dy.clone())),
            );
            let args = vec![sampler(0, SamplerDim::D2), t, dx, dy];
            assert_eq!(done(&mut cx, name, args), expected, "{name}");
            assert_eq!(cx.samplers, [(0, SamplerDim::D2)]);
        }
    }

    #[test]
    fn lod_and_bias_functions_take_w_as_the_level_or_bias() {
        let table = [
            ("tex2Dlod", SamplerDim::D2, true),
            ("tex3Dlod", SamplerDim::D3, true),
            ("texCUBElod", SamplerDim::Cube, true),
            ("tex2Dbias", SamplerDim::D2, false),
            ("tex3Dbias", SamplerDim::D3, false),
            ("texCUBEbias", SamplerDim::Cube, false),
        ];
        for (name, dim, level) in table {
            let mut cx = Mock::legacy();
            let t = local(0, float(4));
            let size = if dim == SamplerDim::D2 { 2 } else { 3 };
            let coords = swizzle(t.clone(), (0..size).collect(), float(size));
            let w = Box::new(swizzle(t.clone(), vec![3], Type::FLOAT));
            let kind = if level {
                Sample::Level(w)
            } else {
                Sample::Bias(w)
            };
            let expected = sampled(1, coords, kind);
            assert_eq!(
                done(&mut cx, name, vec![sampler(1, dim), t]),
                expected,
                "{name}"
            );
            assert_eq!(cx.samplers, [(1, dim)], "{name}");
        }
    }

    #[test]
    fn tex2dproj_divides_by_w() {
        let mut cx = Mock::legacy();
        let t = local(0, float(4));
        let xy = swizzle(t.clone(), vec![0, 1], float(2));
        let w = splat(swizzle(t.clone(), vec![3], Type::FLOAT), &float(2));
        let expected = sampled(0, binary(BinaryOp::Div, xy, w, float(2)), Sample::Plain);
        assert_eq!(
            done(&mut cx, "tex2Dproj", vec![sampler(0, SamplerDim::D2), t]),
            expected
        );
    }

    #[test]
    fn tex_coordinates_are_coerced_to_the_dimension() {
        let mut cx = Mock::legacy();
        let t = local(0, float(4));
        let expected = sampled(0, swizzle(t.clone(), vec![0, 1], float(2)), Sample::Plain);
        assert_eq!(
            done(&mut cx, "tex2D", vec![sampler(0, SamplerDim::D2), t]),
            expected
        );
        assert_eq!(cx.warnings, ["truncated"]);

        let computed = call("fract", vec![local(1, float(4))], float(4));
        done(
            &mut cx,
            "tex2Dlod",
            vec![sampler(0, SamplerDim::D2), computed.clone()],
        );
        let init = Some(computed);
        assert_eq!(cx.stmts, [Stmt::Local { id: TEMP, init }]);
    }

    #[test]
    fn tex_functions_need_a_sampler_global_of_a_matching_dimension() {
        let mut cx = Mock::legacy();
        let t = local(1, float(2));
        failed(&mut cx, "tex2D", vec![local(0, float(2)), t.clone()]);
        let mut cx = Mock::legacy();
        cx.sampler_ok = false;
        let args = vec![sampler(0, SamplerDim::D3), t];
        assert_eq!(lowered(&mut cx, "tex2D", args), Lowered::Failed);
        assert_eq!(cx.samplers, [(0, SamplerDim::D2)]);
    }
}
