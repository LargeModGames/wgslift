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
        Some(expr) => Lowered::Done(expr),
        None => Lowered::Failed,
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
    let x = share(cx, x);
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
    let x = share(cx, x);
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
