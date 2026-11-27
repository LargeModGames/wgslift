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
            "static const float u = 1;\nfloat main() : COLOR { u = 2; return u; }",
            "is written, but it is a uniform or a constant",
        ),
        ("float other() { return 1; }", "no entry point"),
    ];
    for (src, expected) in cases {
        let e = errors(src);
        assert!(e.iter().any(|m| m.contains(expected)), "{src}: {e:?}");
    }
}

#[test]
fn a_written_uniform_gets_a_private_copy_filled_at_entry() {
    let m = ok("uniform float4 q;\nfloat main() : COLOR { q.x = 2; return q.x; }");
    let uniform = m
        .globals
        .iter()
        .position(|g| g.name == "q" && g.space == Space::Uniform);
    let copy = m
        .globals
        .iter()
        .find(|g| g.name == "q" && g.space == Space::Private);
    let (Some(u), Some(copy)) = (uniform, copy) else {
        panic!("{:?}", m.globals)
    };
    assert!(
        matches!(copy.entry_init.as_ref().map(|e| &e.kind), Some(ExprKind::Var(Root::Global(g))) if *g == u)
    );
}

#[test]
fn a_matrix_plus_a_scalar_splats_the_scalar() {
    let m = ok("float main() : COLOR { float2x2 r = 0; r = r - 0.5; r = r * 2; return r[0].x; }");
    let ops: Vec<_> = entry_body(&m)
        .iter()
        .filter_map(|s| match s {
            Stmt::Assign(_, e) => match &e.kind {
                ExprKind::Binary(_, a, b) => Some((a.ty.clone(), b.ty.clone())),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let mat = Type::Matrix { rows: 2, cols: 2 };
    assert_eq!(ops, [(mat.clone(), mat.clone()), (mat, Type::FLOAT)]);
}

#[test]
fn predeclared_units_are_visible_in_the_values_of_source_globals() {
    let source = wgslift_syntax::parse(
        "t.hlsl",
        "static const float k = q1 * 2;\nfloat main() : COLOR { return k; }",
        Default::default(),
    );
    let extra = wgslift_syntax::parse("<p>", "uniform float q1;", Default::default());
    let dialect = Dialect::d3d9_legacy();
    let settings = Settings {
        stage: Stage::Fragment,
        entry_point: "main",
        dialect: &dialect,
    };
    let mut d = Vec::new();
    let m = analyze(&[source.unit, extra.unit], &settings, &mut d);
    assert!(m.is_some(), "{d:?}");
}

#[test]
fn a_matrix_truncates_to_its_first_element_as_a_scalar() {
    let (module, d) = run("float4x3 m;\nfloat main() : COLOR { float x = 1; x -= m; return x; }");
    assert!(module.is_some(), "{d:?}");
    assert!(d.iter().any(|d| d.message.contains("truncation")));
}

/// True when `e` reads `root` anywhere.
fn reads(e: &crate::ir::Expr, root: Root) -> bool {
    let mut found = false;
    super::visit_expr(e, &mut |x| {
        if matches!(x.kind, ExprKind::Var(r) if r == root) {
            found = true;
        }
    });
    found
}

#[test]
fn a_swizzle_store_from_its_own_target_reads_the_old_value() {
    let m =
        ok("float4 main() : COLOR { float3 r = float3(1, 2, 3); r.zyx = r; return float4(r, 1); }");
    let parts: Vec<_> = entry_body(&m)
        .iter()
        .filter_map(|s| match s {
            Stmt::Assign(p, e) if p.path.len() == 1 => Some(e),
            _ => None,
        })
        .collect();
    assert_eq!(parts.len(), 3);
    assert!(parts.iter().all(|e| !reads(e, Root::Local(0))), "{parts:?}");
}

#[test]
fn a_post_increment_gives_the_value_before_the_update() {
    let m = ok("int main() : COLOR { int i = 0; int j = i++; return j; }");
    let body = entry_body(&m);
    let assign = body
        .iter()
        .position(|s| matches!(s, Stmt::Assign(p, _) if p.root == Root::Local(0)))
        .unwrap();
    let Some(Stmt::Local { init: Some(j), .. }) = body.get(assign + 1) else {
        panic!("{body:?}")
    };
    assert!(!reads(j, Root::Local(0)), "{body:?}");
}

#[test]
fn an_overloaded_call_lowers_its_arguments_once_and_picks_the_closest_overload() {
    let m = ok(
        "float f(float a) { return 1; }\nfloat f(float2 a) { return 2; }\n\
         float main() : COLOR { float i = 0; float r = f(i++); return r + f(1); }",
    );
    let body = entry_body(&m);
    let increments = body
        .iter()
        .filter(|s| matches!(s, Stmt::Assign(p, _) if p.root == Root::Local(0)))
        .count();
    assert_eq!(increments, 1, "{body:?}");
    let mut callees = Vec::new();
    for s in body {
        if let Stmt::Local { init: Some(e), .. } | Stmt::Return(Some(e)) = s {
            super::visit_expr(e, &mut |x| {
                if let ExprKind::Call(f, _) = x.kind {
                    callees.push(f);
                }
            });
        }
    }
    assert_eq!(callees.len(), 2);
    for f in callees {
        assert_eq!(m.functions[f].params.len(), 1);
        let local = m.functions[f].params[0].local;
        assert_eq!(m.functions[f].locals[local].ty, Type::FLOAT);
    }
}

#[test]
fn a_for_loop_variable_stays_visible_after_the_loop() {
    let m = ok(
        "static float i = 7;\nfloat main() : COLOR { for (int i = 0; i < 3; i++) {} return i; }",
    );
    let Some(Stmt::Return(Some(r))) = entry_body(&m).last() else {
        panic!()
    };
    assert!(reads(r, Root::Local(0)), "{r:?}");
}

#[test]
fn a_static_local_keeps_its_value_between_calls() {
    let m = ok("float count() { static float n = 0; n += 1; return n; }\n\
         float main() : COLOR { float a = count(); return a + count(); }");
    let n = m
        .globals
        .iter()
        .position(|g| g.name == "n")
        .expect("a global");
    assert_eq!(m.globals[n].space, Space::Private);
    let count = m.functions.iter().find(|f| f.name == "count").unwrap();
    assert!(count.locals.is_empty(), "{:?}", count.locals);
}

#[test]
fn a_constant_integer_with_a_leading_zero_is_octal() {
    let m = ok("static float a[010];\nfloat main() : COLOR { return a[0]; }");
    assert_eq!(m.globals[0].ty, Type::Array(Box::new(Type::FLOAT), 8));
}

#[test]
fn a_sampler_used_only_in_a_global_value_is_kept() {
    let m = ok(
        "sampler s;\nstatic float4 c = tex2D(s, float2(0.5, 0.5));\nfloat main() : COLOR { return c.x; }",
    );
    assert!(m.globals.iter().any(|g| g.name == "s"), "{:?}", m.globals);
}

#[test]
fn modf_and_sincos_read_the_input_before_they_store_into_it() {
    let m = ok(
        "float main() : COLOR { float a = 3.7; float f = modf(a, a); float b = 1; float c; sincos(b, b, c); return f + c; }",
    );
    let body = entry_body(&m);
    let fraction = body
        .iter()
        .find_map(|s| match s {
            Stmt::Local { init: Some(e), .. } if matches!(e.kind, ExprKind::Binary(..)) => Some(e),
            _ => None,
        })
        .unwrap();
    // a is local 0 and b is local 4; the calls write both.
    assert!(!reads(fraction, Root::Local(0)), "{body:?}");
    let cos = body
        .iter()
        .find_map(|s| match s {
            Stmt::Assign(_, e) if matches!(&e.kind, ExprKind::Builtin(n, _) if *n == "cos") => {
                Some(e)
            }
            _ => None,
        })
        .unwrap();
    assert!(!reads(cos, Root::Local(4)), "{body:?}");
}

#[test]
fn a_constant_integer_that_overflows_is_an_error() {
    for size in ["(1 << 63) / -1", "(1 << 63) % -1", "-(1 << 63)"] {
        let src = format!("float a[{size}];\nfloat4 main() : COLOR {{ return a[0]; }}");
        let e = errors(&src);
        assert!(
            e.iter().any(|m| m.contains("constant integer")),
            "{src}: {e:?}"
        );
    }
}

#[test]
fn a_write_in_the_value_of_a_global_is_an_error_with_a_message() {
    for value in ["modf(1.5, g)", "f(g)", "(clip(-1.0), 1.0)"] {
        let src = format!(
            "static float g;\nfloat f(inout float x) {{ x = 1; return 2; }}\n\
             static float h = {value};\nfloat4 main() : COLOR {{ return h; }}"
        );
        let e = errors(&src);
        assert!(
            e.iter().any(|m| m.contains("value of a global")),
            "{src}: {e:?}"
        );
    }
}

#[test]
fn a_swizzle_is_an_out_argument_of_sincos_and_modf() {
    let m = ok(
        "float4 main() : COLOR { float2 ag = 0; sincos(1.0, ag.y, ag.x); \
         float3 v = 0; float f = modf(2.5, v.z); return float4(ag, v.z + f, 1); }",
    );
    let components: Vec<_> = entry_body(&m)
        .iter()
        .filter_map(|s| match s {
            Stmt::Assign(p, _) if !p.path.is_empty() => Some((p.root, p.path.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        components,
        [
            (Root::Local(0), vec![Access::Component(1)]),
            (Root::Local(0), vec![Access::Component(0)]),
            (Root::Local(3), vec![Access::Component(2)]),
        ],
        "{:?}",
        entry_body(&m)
    );
}

#[test]
fn a_value_used_twice_in_a_global_value_is_kept_once() {
    let depth = 8;
    let value = (0..depth).fold("y".to_string(), |acc, _| format!("pow({acc}, 4)"));
    let m = ok(&format!(
        "float y;\nstatic float g = {value};\nfloat4 main() : COLOR {{ return g; }}"
    ));
    let mut nodes = 0;
    for g in &m.globals {
        for value in g.init.iter().chain(&g.entry_init) {
            super::visit_expr(value, &mut |_| nodes += 1);
        }
    }
    assert!(nodes < 100, "{nodes} nodes");
}

#[test]
fn nested_overloaded_calls_lower_each_argument_once() {
    let depth = 40;
    let call = (0..depth).fold("1.0".to_string(), |acc, _| format!("f({acc})"));
    let m = ok(&format!(
        "float f(float x) {{ return x; }}\nfloat f(int x) {{ return x; }}\n\
         float4 main() : COLOR {{ return {call}; }}"
    ));
    assert_eq!(m.functions.len(), 2);
}

#[test]
fn an_overloaded_call_takes_a_value_where_another_overload_writes() {
    for arg in ["tex2D(s, uv).r", "float3(1, 2, 3).x", "uv.xx.y"] {
        ok(&format!(
            "float g(float a, out float b) {{ b = a; return a; }}\n\
             float g(float2 a, float b) {{ return a.x + b; }}\nsampler2D s;\n\
             float4 main(float2 uv : TEXCOORD0) : COLOR {{ return g(uv, {arg}); }}"
        ));
    }
}

#[test]
fn an_overloaded_call_reads_an_inout_argument_before_later_arguments_write_it() {
    let m = ok(
        "void g(inout float a, float b) {}\nvoid g(inout float2 a, float b) {}\n\
         float4 main() : COLOR { float x = 1; g(x, x = 5); return x; }",
    );
    let body = entry_body(&m);
    let copy = body
        .iter()
        .position(|s| matches!(s, Stmt::Local { init: Some(e), .. } if matches!(e.kind, ExprKind::Var(Root::Local(0)))))
        .expect("a copy of x");
    let store = body
        .iter()
        .position(|s| matches!(s, Stmt::Assign(p, _) if p.root == Root::Local(0)))
        .unwrap();
    assert!(copy < store, "{body:?}");
}

#[test]
fn a_static_local_is_set_once_from_globals_when_the_shader_starts() {
    let m = ok("float4 u;\nstatic float g = 5;\n\
         float helper(float x) { static float s = u.x * 3 + g; s += x; return s; }\n\
         float4 main(float2 uv : TEXCOORD0) : COLOR { g = uv.x; return float4(helper(uv.x), helper(uv.y), 0, 1); }");
    let s = m.globals.iter().find(|g| g.name == "s").expect("a global");
    assert_eq!(s.space, Space::Private);
    let g = m.globals.iter().position(|g| g.name == "g").unwrap();
    assert!(
        s.init.is_none()
            && s.entry_init
                .as_ref()
                .is_some_and(|e| reads(e, Root::Global(g))),
        "{s:?}"
    );
    let helper = m.functions.iter().find(|f| f.name == "helper").unwrap();
    assert_eq!(helper.locals.len(), 1, "{:?}", helper.locals);
}

#[test]
fn a_static_local_cannot_use_a_local_or_leave_a_write_behind() {
    let cases = [
        "float4 main(float2 uv : TEXCOORD0) : COLOR { static float s = uv.x; return s; }",
        "float4 main() : COLOR { float k = 1; static float s = k * 2; return s; }",
        "float4 main() : COLOR { float x; static float s = (x = 1, 2); return s; }",
        "static float g;\nfloat4 main() : COLOR { static float s = (g = 1, 2); return s; }",
    ];
    for src in cases {
        assert!(!errors(src).is_empty(), "{src}");
    }
}

#[test]
fn a_sampler_in_a_struct_is_an_error() {
    for field in ["sampler2D tex;", "sampler s[2];"] {
        let src = format!(
            "struct Material {{ {field} float4 tint; }};\nfloat4 main() : COLOR0 {{ return 1; }}"
        );
        let e = errors(&src);
        assert!(
            e.iter().any(|m| m.contains("samplers in structs")),
            "{src}: {e:?}"
        );
    }
}

#[test]
fn a_type_with_too_many_elements_is_an_error() {
    let cases = [
        "float b[65536][65536];\nfloat4 main() : COLOR { return b[1][1]; }",
        "struct E {};\nstatic E a[65536][65536] = {};\nfloat4 main() : COLOR { return 0; }",
        "struct S0 { float a[65536]; };\nstruct S1 { S0 a, b; };\nfloat4 main() : COLOR { return 0; }",
    ];
    for src in cases {
        let e = errors(src);
        assert!(e.iter().any(|m| m.contains("too large")), "{src}: {e:?}");
    }
}

#[test]
fn a_prototype_and_its_definition_are_one_function() {
    let m = ok(
        "float f(float x);\nfloat4 main(float2 uv : TEXCOORD0) : COLOR0;\n\
         float4 main(float2 uv : TEXCOORD0) : COLOR0 { return f(uv.x); }\n\
         float f(float y) { return y * 2; }",
    );
    assert_eq!(m.functions.len(), 2);
    let f = m.functions.iter().find(|f| f.name == "f").unwrap();
    assert_eq!(f.locals[f.params[0].local].name, "y");
}

#[test]
fn constant_math_out_of_range_still_creates_a_shader() {
    let values = [
        "exp(89.0)",
        "log(0.0)",
        "sqrt(-1.0) + 1",
        "exp2(k) * 2",
        "pow(10.0, 50.0)",
        "(float4)exp(89.0)",
        "acos(2.0)",
        "0.0 / 0.0",
    ];
    for value in values {
        let src =
            format!("static const float k = 200.0;\nfloat4 main() : COLOR0 {{ return {value}; }}");
        let output = crate::translate("t.hlsl", &src, Default::default())
            .unwrap_or_else(|f| panic!("{src}: {f}"));
        let wgsl = &output.wgsl;
        let module = naga::front::wgsl::parse_str(wgsl)
            .unwrap_or_else(|e| panic!("{src}\n{}\n{wgsl}", e.emit_to_string(wgsl)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{src}\n{}\n{wgsl}", e.emit_to_string(wgsl)));
    }
}
