//! Tests of the analysis, on the IR it builds.

use wgslift_syntax::{Diagnostic, Severity};

use super::{Settings, analyze};
use crate::ir::{Access, Arg, ExprKind, Module, Root, Space, Stmt};
use crate::types::{Scalar, Type};
use crate::{Dialect, Stage};

fn run(src: &str) -> (Option<Module>, Vec<Diagnostic>) {
    let parsed = wgslift_syntax::parse("t.hlsl", src, Default::default());
    assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
    let dialect = Dialect::d3d9_legacy();
    let settings = Settings {
        stage: Stage::Fragment,
        entry_point: "main",
        dialect: &dialect,
    };
    let mut diagnostics = Vec::new();
    let module = analyze(&[parsed.unit], &settings, &mut diagnostics);
    (module, diagnostics)
}

fn ok(src: &str) -> Module {
    let (module, d) = run(src);
    let errors: Vec<_> = d.iter().filter(|d| d.severity == Severity::Error).collect();
    assert!(errors.is_empty(), "{errors:?}");
    module.expect("a module")
}

fn errors(src: &str) -> Vec<String> {
    run(src)
        .1
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message)
        .collect()
}

fn entry_body(m: &Module) -> &[Stmt] {
    &m.functions[m.entry].body
}

#[test]
fn a_scalar_splats_and_a_longer_vector_truncates_with_a_warning() {
    let (module, d) =
        run("float4 main() : COLOR { float3 a = 1; float2 b = a; return float4(b, 0, 1); }");
    let m = module.unwrap();
    let Stmt::Local { init: Some(a), .. } = &entry_body(&m)[0] else {
        panic!()
    };
    assert!(matches!(a.kind, ExprKind::Splat(_)));
    assert_eq!(a.ty, Type::Vector(Scalar::Float, 3));
    let Stmt::Local { init: Some(b), .. } = &entry_body(&m)[1] else {
        panic!()
    };
    assert!(matches!(&b.kind, ExprKind::Swizzle(_, c) if c == &[0, 1]));
    assert!(
        d.iter()
            .any(|d| d.severity == Severity::Warning && d.message.contains("truncation"))
    );
}

#[test]
fn integer_literals_fold_into_float_operands() {
    let m = ok("float main() : COLOR { float x = 2; return x * 3; }");
    let Stmt::Return(Some(r)) = entry_body(&m).last().unwrap() else {
        panic!()
    };
    let ExprKind::Binary(_, _, rhs) = &r.kind else {
        panic!()
    };
    assert_eq!(rhs.ty, Type::FLOAT);
    assert!(matches!(rhs.kind, ExprKind::Literal(_)));
}

#[test]
fn a_swizzle_store_writes_each_component() {
    let m = ok("float4 main() : COLOR { float4 c = 0; c.zx = float2(1, 2); return c; }");
    let stores: Vec<_> = entry_body(&m)
        .iter()
        .filter_map(|s| match s {
            Stmt::Assign(p, _) => Some(p.path.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        stores,
        [vec![Access::Component(2)], vec![Access::Component(0)]]
    );
}

#[test]
fn an_out_argument_goes_through_a_temporary_and_back() {
    let m = ok(
        "void f(out float2 v) { v = 1; }\nfloat4 main() : COLOR { float4 c = 0; f(c.yz); return c; }",
    );
    let body = entry_body(&m);
    assert!(body.iter().any(|s| matches!(s, Stmt::Eval(e) if matches!(&e.kind, ExprKind::Call(_, args) if matches!(args[0], Arg::Ref(_))))));
    let writes = body
        .iter()
        .filter(|s| matches!(s, Stmt::Assign(p, _) if p.path.len() == 1))
        .count();
    assert_eq!(writes, 2);
}

#[test]
fn loops_become_wgsl_loops_with_the_condition_inside() {
    let m = ok(
        "float main() : COLOR { float s = 0; for (int i = 0; i < 4; i++) { s += i; } int j = 0; do { j++; } while (j < 3); while (s > 1) s -= 1; return s; }",
    );
    let loops = |b: &[Stmt]| -> usize {
        b.iter()
            .map(|s| match s {
                Stmt::Loop { .. } => 1,
                Stmt::Block(inner) => inner
                    .iter()
                    .filter(|s| matches!(s, Stmt::Loop { .. }))
                    .count(),
                _ => 0,
            })
            .sum()
    };
    assert_eq!(loops(entry_body(&m)), 3);
    let do_while = entry_body(&m).iter().find(|s| {
        matches!(
            s,
            Stmt::Loop {
                break_if: Some(_),
                ..
            }
        )
    });
    assert!(do_while.is_some());
}

#[test]
fn globals_are_uniform_private_or_const_by_their_use() {
    let m = ok(
        "float a;\nfloat b;\nfloat c = 2;\nstatic const float d = 3;\nstatic float e = a * 2;\nuniform float4 t;\n\
                float main() : COLOR { b = 1; return a + b + c + d + e + t.x; }",
    );
    let space = |name: &str| {
        m.globals
            .iter()
            .find(|g| g.name == name)
            .map(|g| (g.space, g.init.is_some(), g.entry_init.is_some()))
    };
    assert_eq!(space("a"), Some((Space::Uniform, false, false)));
    assert_eq!(space("b"), Some((Space::Private, false, false)));
    assert_eq!(space("c"), Some((Space::Private, true, false)));
    assert_eq!(space("d"), Some((Space::Const, true, false)));
    assert_eq!(space("e"), Some((Space::Private, false, true)));
    assert_eq!(space("t"), Some((Space::Uniform, false, false)));
}

#[test]
fn unused_globals_and_functions_are_removed() {
    let m = ok(
        "float unused;\nsampler s;\nfloat helper() { return 1; }\nfloat spare() { return unused; }\nfloat main() : COLOR { return helper(); }",
    );
    assert!(m.globals.is_empty());
    assert_eq!(m.functions.len(), 2);
    let names: Vec<&str> = m.functions.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"helper") && names.contains(&"main"));
    assert_eq!(m.functions[m.entry].name, "main");
}

#[test]
fn brace_lists_flatten_and_rebuild() {
    let m = ok(
        "static const float2 p[2] = { {1, 2}, 3, 4 };\nfloat main() : COLOR { float2x2 r = { 1, 0, 0, 1 }; return p[1].y + r[0].x; }",
    );
    let p = m.globals.iter().find(|g| g.name == "p").unwrap();
    assert_eq!(p.space, Space::Const);
    let Some(init) = &p.init else { panic!() };
    assert!(matches!(&init.kind, ExprKind::Construct(parts) if parts.len() == 2));
    let e = errors("float main() : COLOR { float2 v = { 1, 2, 3 }; return 0; }");
    assert!(e.iter().any(|m| m.contains("too many")), "{e:?}");
}

#[test]
fn matrix_rows_index_and_elements_read() {
    let m = ok(
        "float main() : COLOR { float3x3 m = 0; float3 r = m[1]; return r.x + m._m21 + m._33; }",
    );
    assert!(!entry_body(&m).is_empty());
}

#[test]
fn a_function_without_a_return_at_the_end_gets_one() {
    let m = ok("float f(float x) { if (x > 0) return 1; }\nfloat main() : COLOR { return f(2); }");
    let f = m.functions.iter().find(|f| f.name == "f").unwrap();
    assert!(matches!(f.body.last(), Some(Stmt::Return(Some(_)))));
}

#[test]
fn written_parameters_are_marked() {
    let m = ok("float f(float x) { x = x * 2; return x; }\nfloat main() : COLOR { return f(1); }");
    let f = m.functions.iter().find(|f| f.name == "f").unwrap();
    assert!(f.locals[f.params[0].local].written);
}

#[test]
fn increments_and_assignments_inside_expressions_become_statements() {
    let m = ok("float main() : COLOR { int i = 0; int j = i++ + (i = 5); return j; }");
    let assigns = entry_body(&m)
        .iter()
        .filter(|s| matches!(s, Stmt::Assign(p, _) if p.root == Root::Local(0)))
        .count();
    assert_eq!(assigns, 2);
}

#[test]
fn errors_name_the_problem() {
    let cases = [
        ("float main() : COLOR { return nothing; }", "unknown name"),
        (
            "float main() : COLOR { return nofunction(1); }",
            "unknown function",
        ),
        (
            "float main() : COLOR { int k = 0; switch (k) { case 0: k = 1; case 1: k = 2; break; } return k; }",
            "falls into",
        ),
        (
            "float main() : COLOR { break; return 0; }",
            "outside a loop",
        ),
        (
            "float t;\nfloat main() : COLOR { float3 v = 0; return v.xw; }",
            "not a swizzle",
        ),
        (
            "uniform float u;\nfloat main() : COLOR { u = 1; return u; }",
            "cannot be written",
        ),
        ("float other() { return 1; }", "no entry point"),
    ];
    for (src, expected) in cases {
        let e = errors(src);
        assert!(e.iter().any(|m| m.contains(expected)), "{src}: {e:?}");
    }
}
