//! Parser tests through the public `parse` entry point.

use crate::ast::*;
use crate::{Options, Parsed, parse};

fn ok(text: &str) -> TranslationUnit {
    let parsed = parse("t.hlsl", text, Options::default());
    let errors: Vec<String> = parsed
        .diagnostics
        .iter()
        .map(|d| d.display(&parsed.files).to_string())
        .collect();
    assert!(errors.is_empty(), "{errors:#?}");
    parsed.unit
}

fn errors(text: &str) -> Vec<String> {
    let parsed: Parsed = parse("t.hlsl", text, Options::default());
    parsed
        .diagnostics
        .iter()
        .map(|d| d.display(&parsed.files).to_string())
        .collect()
}

/// The body statements of the only function.
fn body(text: &str) -> Vec<Stmt> {
    let unit = ok(text);
    match &unit.items[..] {
        [Item::Function(f)] => f.body.clone().expect("a body").stmts,
        other => panic!("{other:?}"),
    }
}

/// The expression in `void f() { <expr>; }` as an s-expression.
fn expr(text: &str) -> String {
    let stmts = body(&format!("void f() {{ {text}; }}"));
    match &stmts[0].kind {
        StmtKind::Expr(e) => sexpr(e),
        other => panic!("{other:?}"),
    }
}

fn sexpr(e: &Expr) -> String {
    let list = |items: &[Expr]| items.iter().map(sexpr).collect::<Vec<_>>().join(" ");
    match &e.kind {
        ExprKind::Literal(Literal::Int(s) | Literal::Float(s) | Literal::Str(s)) => s.clone(),
        ExprKind::Literal(Literal::Bool(b)) => b.to_string(),
        ExprKind::Ident(name) => name.clone(),
        ExprKind::Unary(op, x) => format!("({op:?} {})", sexpr(x)),
        ExprKind::Binary(op, a, b) => format!("({op:?} {} {})", sexpr(a), sexpr(b)),
        ExprKind::Assign(op, a, b) => format!("(= {op:?} {} {})", sexpr(a), sexpr(b)),
        ExprKind::Ternary(c, a, b) => format!("(? {} {} {})", sexpr(c), sexpr(a), sexpr(b)),
        ExprKind::Comma(items) => format!("(, {})", list(items)),
        ExprKind::Call(f, args) => format!("(call {} {})", sexpr(f), list(args)),
        ExprKind::Construct(ty, args) => format!("(new {} {})", ty.name.name, list(args)),
        ExprKind::Cast(ty, x) => format!("(cast {} {})", ty.name.name, sexpr(x)),
        ExprKind::Member(x, name) => format!("(. {} {})", sexpr(x), name.name),
        ExprKind::Index(x, i) => format!("([] {} {})", sexpr(x), sexpr(i)),
    }
}

#[test]
fn precedence_and_associativity() {
    assert_eq!(expr("a + b * c - d"), "(Sub (Add a (Mul b c)) d)");
    assert_eq!(
        expr("a || b && c | d ^ e & f"),
        "(Or a (And b (BitOr c (BitXor d (BitAnd e f)))))"
    );
    assert_eq!(expr("a == b < c << d"), "(Eq a (Lt b (Shl c d)))");
    assert_eq!(expr("a = b += c"), "(= None a (= Some(Add) b c))");
    assert_eq!(expr("c ? a : b ? d : e"), "(? c a (? b d e))");
    assert_eq!(expr("a, b = 1"), "(, a (= None b 1))");
}

#[test]
fn unary_postfix_and_casts() {
    assert_eq!(expr("- -x++"), "(Neg (Neg (PostInc x)))");
    assert_eq!(expr("!a.x[2].yz"), "(Not (. ([] (. a x) 2) yz))");
    assert_eq!(expr("(float)x * 2"), "(Mul (cast float x) 2)");
    assert_eq!(expr("(float2(1, 2)).y"), "(. (new float2 1 2) y)");
    assert_eq!(expr("(x) - 1"), "(Sub x 1)");
    assert_eq!(expr("float4(uv, 0, 1).xy"), "(. (new float4 uv 0 1) xy)");
    assert_eq!(
        expr("tex2D(sampler_main, uv).xyz"),
        "(. (call tex2D sampler_main uv) xyz)"
    );
    assert_eq!(expr("tex.Sample(s, uv)"), "(call (. tex Sample) s uv)");
    assert_eq!(expr("--i"), "(PreDec i)");
}

#[test]
fn template_types_split_a_double_close() {
    let unit = ok("Texture2D<vector<float, 4>> tex; vector<float, 2 + 1> v;");
    let Item::Variable(var) = &unit.items[0] else {
        panic!()
    };
    assert_eq!(var.ty.name.name, "Texture2D");
    let TypeArg::Type(inner) = &var.ty.args[0] else {
        panic!()
    };
    assert_eq!(inner.name.name, "vector");
    assert_eq!(inner.args.len(), 2);
    assert_eq!(expr("vector<float, 2>(1, 2)"), "(new vector 1 2)");
}

#[test]
fn a_milkdrop_style_shader_parses() {
    let text = r#"
#define sat saturate
sampler sampler_pw_noise_lq;
static const float PI = 3.14159265;
float4 _qa : register(c0);
float3 glow(float2 p, float r) { return sat(1 - length(p) / r).xxx; }
void main(float2 uv : TEXCOORD0, out float4 ret : COLOR0)
{
    float3 c = tex2D(sampler_main, uv).xyz * 0.98f;
    [unroll] for (int i = 0; i < 4; i++) {
        c += glow(uv - 0.5, 0.1 + i * 0.05) * (i % 2 == 0 ? 1 : 0.5);
    }
    if (c.x > 1) c = 1; else { c *= 0.5; }
    ret = float4(c, 1);
}
"#;
    let unit = ok(text);
    assert_eq!(unit.items.len(), 5);
    let Item::Function(main) = &unit.items[4] else {
        panic!()
    };
    assert_eq!(main.params.len(), 2);
    assert_eq!(main.params[1].modifiers, [Modifier::Out]);
    assert_eq!(
        main.params[1].bindings.semantic.as_ref().unwrap().name,
        "COLOR0"
    );
    let stmts = &main.body.as_ref().unwrap().stmts;
    assert_eq!(stmts[1].attributes[0].name.name, "unroll");
    assert!(matches!(stmts[1].kind, StmtKind::For { .. }));
    assert!(matches!(
        stmts[2].kind,
        StmtKind::If {
            otherwise: Some(_),
            ..
        }
    ));
}

#[test]
fn declarations_with_bindings_initializers_and_annotations() {
    let text = r#"
struct VSOut { float4 pos : SV_Position; nointerpolation float2 uv : TEXCOORD0; };
typedef float3 color, palette[4];
cbuffer Params : register(b0) { float4 tint : packoffset(c0); float gain : packoffset(c1.y); };
static const float2 offsets[2] = { {0.5, 0.5}, {1, 1}, };
color base = {1, 0, 0};
float amount < string UIName = "Amount"; > = 0.5;
texture tex;
sampler2D s : register(ps_3_0, s1) = sampler_state { Texture = <tex>; MinFilter = Linear; AddressU = (WRAP); };
float4 f(VSOut v, uniform float k = 2, float) : SV_Target { palette p; color c = (color)0; return 0; }
"#;
    let unit = ok(text);
    assert_eq!(unit.items.len(), 9);
    let Item::Struct(s) = &unit.items[0] else {
        panic!()
    };
    assert_eq!(s.fields[1].modifiers, [Modifier::Nointerpolation]);
    let Item::Buffer(b) = &unit.items[2] else {
        panic!()
    };
    let pack = b.members[1].declarators[0]
        .bindings
        .packoffset
        .as_ref()
        .unwrap();
    assert_eq!(
        (
            pack.slot.name.as_str(),
            pack.component.as_ref().unwrap().name.as_str()
        ),
        ("c1", "y")
    );
    let Item::Variable(offsets) = &unit.items[3] else {
        panic!()
    };
    assert!(
        matches!(&offsets.declarators[0].init, Some(Initializer::List(items, _)) if items.len() == 2)
    );
    let Item::Variable(amount) = &unit.items[5] else {
        panic!()
    };
    assert!(amount.declarators[0].annotations.is_some());
    let Item::Variable(sampler) = &unit.items[7] else {
        panic!()
    };
    let reg = &sampler.declarators[0].bindings.registers[0];
    assert_eq!(reg.profile.as_ref().unwrap().name, "ps_3_0");
    let Some(Initializer::SamplerState(states, _)) = &sampler.declarators[0].init else {
        panic!()
    };
    assert_eq!(states.len(), 3);
    assert_eq!(states[0].value.kind, ExprKind::Ident("tex".into()));
    let Item::Function(f) = &unit.items[8] else {
        panic!()
    };
    assert!(f.params[2].name.is_none());
    assert!(f.params[1].default.is_some());
    assert_eq!(
        f.return_bindings.semantic.as_ref().unwrap().name,
        "SV_Target"
    );
}

#[test]
fn statements() {
    let stmts = body(
        "void f() { ; { } while (a) a--; do { b++; } while (b < 3); \
         switch (k) { case 0: case 1: x = 1; break; default: discard; } \
         for (;;) { continue; } for (i = 0, j = 1; i < j; i++, j--) ; return; }",
    );
    let kinds: Vec<&str> = stmts
        .iter()
        .map(|s| match s.kind {
            StmtKind::Empty => "empty",
            StmtKind::Block(_) => "block",
            StmtKind::While { .. } => "while",
            StmtKind::DoWhile { .. } => "do",
            StmtKind::Switch { .. } => "switch",
            StmtKind::For { .. } => "for",
            StmtKind::Return(None) => "return",
            _ => "other",
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "empty", "block", "while", "do", "switch", "for", "for", "return"
        ]
    );
    let StmtKind::Switch { cases, .. } = &stmts[4].kind else {
        panic!()
    };
    assert_eq!(cases[0].labels.len(), 2);
    assert_eq!(cases[0].stmts.len(), 2);
    assert_eq!(cases[1].labels, [None]);
}

#[test]
fn variables_named_like_modifiers_and_types() {
    assert_eq!(
        expr("sample = linear * 2"),
        "(= None sample (Mul linear 2))"
    );
    let stmts = body("void f() { float3 sample = 1; float texture = sample.x; texture += 1; }");
    assert!(matches!(stmts[0].kind, StmtKind::Var(_)));
    assert!(matches!(stmts[2].kind, StmtKind::Expr(_)));
}

#[test]
fn unknown_type_names_still_declare() {
    let stmts = body("void f() { Light l; l.color = 1; }");
    let StmtKind::Var(var) = &stmts[0].kind else {
        panic!()
    };
    assert_eq!(var.ty.name.name, "Light");
}

#[test]
fn errors_are_reported_and_parsing_goes_on() {
    let e = errors("void f() { x = ; y = 2 }\nfloat4 g() { return 1; }\nint 3;");
    assert_eq!(e.len(), 3, "{e:#?}");
    assert!(e[0].starts_with("t.hlsl:1:16: error: expected an expression"));
    assert!(e[1].contains("expected `;`"));
    assert!(e[2].starts_with("t.hlsl:3:5"));
    // The error is inside the body, so `f` stays, without the bad statement.
    let parsed = parse(
        "t.hlsl",
        "void f() { x = ; y = 1; }\nfloat4 g() { return 1; }",
        Options::default(),
    );
    assert_eq!(parsed.unit.items.len(), 2);
    let Item::Function(f) = &parsed.unit.items[0] else {
        panic!()
    };
    assert_eq!(f.body.as_ref().unwrap().stmts.len(), 1);
}

#[test]
fn a_node_from_a_function_like_macro_covers_the_call() {
    let parsed = parse(
        "t.hlsl",
        "#define GET(uv) tex2D(s, uv)\nvoid f() { x = GET(t); }",
        Options::default(),
    );
    let Item::Function(f) = &parsed.unit.items[0] else {
        panic!()
    };
    let StmtKind::Expr(e) = &f.body.as_ref().unwrap().stmts[0].kind else {
        panic!()
    };
    assert_eq!(parsed.files.text(e.span), "x = GET(t)");
}

#[test]
fn errors_in_macros_point_at_the_call() {
    let e = errors("#define BAD (1 +)\nvoid f() { x = BAD; }");
    assert_eq!(e.len(), 1);
    assert!(e[0].starts_with("t.hlsl:2:16"), "{e:?}");
}

#[test]
fn techniques_and_stray_tokens_are_errors() {
    assert_eq!(errors("technique T { pass P { } }\nfloat x;").len(), 1);
    assert_eq!(errors("void f() { else x; }").len(), 1);
    assert_eq!(errors("[unroll] float x;").len(), 1);
    assert_eq!(errors("void f() {").len(), 1);
}

#[test]
fn deep_nesting_is_an_error_not_a_crash() {
    let depth = 20_000;
    let text = format!(
        "void f() {{ x = {}1{}; }}",
        "(".repeat(depth),
        ")".repeat(depth)
    );
    assert!(errors(&text).iter().any(|e| e.contains("nested more than")));
    let text = format!("void f() {}{}", "{".repeat(depth), "}".repeat(depth));
    assert!(errors(&text).iter().any(|e| e.contains("nested more than")));
    let text = format!("void f() {{ x = {}1; }}", "-".repeat(depth));
    assert!(errors(&text).iter().any(|e| e.contains("nested more than")));
    let text = format!("void f() {{ x = {}1; }}", "a ? b : ".repeat(depth));
    assert!(errors(&text).iter().any(|e| e.contains("nested more than")));
    let text = format!(
        "vector<{}float{}> v;",
        "vector<".repeat(depth),
        ">".repeat(depth)
    );
    assert!(errors(&text).iter().any(|e| e.contains("nested more than")));
    // Chains parse in loops, but their trees are deep: the tree limit stops them.
    for chain in ["a = ", "a + ", "a * b + "] {
        let text = format!("void f() {{ x = {}1; }}", chain.repeat(depth));
        assert!(
            errors(&text)
                .iter()
                .any(|e| e.contains("more than 256 levels")),
            "{chain}"
        );
    }
    let text = format!("void f() {{ x = a{}; }}", ".x".repeat(depth));
    assert!(
        errors(&text)
            .iter()
            .any(|e| e.contains("more than 256 levels"))
    );
    let text = format!("void f() {{ x = a{}; }}", "[0]".repeat(depth));
    assert!(
        errors(&text)
            .iter()
            .any(|e| e.contains("more than 256 levels"))
    );
    // Deep chains inside parentheses add up.
    let inner = format!("a{}", " + a".repeat(150));
    let text = format!(
        "void f() {{ x = ({inner}) + ({inner}){}; }}",
        " + a".repeat(150)
    );
    assert!(
        errors(&text)
            .iter()
            .any(|e| e.contains("more than 256 levels"))
    );
}

#[test]
fn long_chains_below_the_tree_limit_parse() {
    ok(&format!("void f() {{ x = a{}; }}", " + a".repeat(200)));
    ok(&format!("void f() {{ x = a{}; }}", ".x".repeat(200)));
    assert!(expr(&format!("{}1", "a = ".repeat(100))).starts_with("(= None a (= None a"));
}

#[test]
fn nesting_below_the_limit_parses() {
    let depth = 40;
    ok(&format!(
        "void f() {{ x = {}1{}; }}",
        "(".repeat(depth),
        ")".repeat(depth)
    ));
    ok(&format!(
        "void f() {{ x = {}1{}; }}",
        "a[".repeat(depth),
        "]".repeat(depth)
    ));
    ok(&format!(
        "void f() {}{}",
        "{".repeat(depth),
        "}".repeat(depth)
    ));
}

#[test]
fn spans_cover_the_source_of_each_node() {
    let parsed = parse("t.hlsl", "void f() { x = a + b; }", Options::default());
    let Item::Function(f) = &parsed.unit.items[0] else {
        panic!()
    };
    let StmtKind::Expr(e) = &f.body.as_ref().unwrap().stmts[0].kind else {
        panic!()
    };
    assert_eq!(parsed.files.text(e.span), "x = a + b");
    assert_eq!(parsed.files.text(f.span), "void f() { x = a + b; }");
}

#[test]
fn builtin_type_names() {
    use crate::parser::is_builtin_type;
    for t in [
        "float",
        "float1",
        "float4x4",
        "half3",
        "min16float2",
        "uint2x3",
        "sampler2D",
        "Texture2D",
    ] {
        assert!(is_builtin_type(t), "{t}");
    }
    for t in ["float5", "float4x", "float0", "floaty", "Light"] {
        assert!(!is_builtin_type(t), "{t}");
    }
}

#[test]
fn the_deepest_accepted_nesting_fits_a_small_stack() {
    let n = crate::parser::MAX_DEPTH - 2;
    let inputs = [
        format!("void f() {{ x = {}1{}; }}", "(".repeat(n), ")".repeat(n)),
        format!("void f() {{ x = {}1{}; }}", "a[".repeat(n), "]".repeat(n)),
        format!("void f() {{ x = {}1{}; }}", "f(".repeat(n), ")".repeat(n)),
        format!(
            "void f() {{ x = {}1{}; }}",
            "float4(".repeat(n),
            ")".repeat(n)
        ),
        format!("void f() {{ x = {}1; }}", "a ? b : ".repeat(n)),
        format!("void f() {{ x = {}1; }}", "-".repeat(n)),
        format!("void f() {}{}", "{".repeat(n), "}".repeat(n)),
        format!("void f() {{ {}x = 1; }}", "if (a) ".repeat(n)),
        format!("float a = {}1{};", "{".repeat(n), "}".repeat(n)),
    ];
    std::thread::Builder::new()
        .stack_size(768 * 1024)
        .spawn(move || {
            for text in inputs {
                let parsed = parse("t.hlsl", &text, Options::default());
                assert!(!parsed.has_errors(), "{text:.40}");
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn errors_in_a_switch_keep_the_other_cases_and_the_function() {
    let text = "void f() { switch (k) { case 1 x = 1; break; case 2: y = 2; break; } z = 3; }\nfloat4 g() : COLOR { return 1; }";
    let parsed = parse("t.hlsl", text, Options::default());
    assert_eq!(parsed.diagnostics.len(), 1, "{:?}", parsed.diagnostics);
    assert_eq!(parsed.unit.items.len(), 2);
    let Item::Function(f) = &parsed.unit.items[0] else {
        panic!()
    };
    let stmts = &f.body.as_ref().unwrap().stmts;
    assert_eq!(stmts.len(), 2);
    let StmtKind::Switch { cases, .. } = &stmts[0].kind else {
        panic!()
    };
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[1].stmts.len(), 2);

    let text = "void f() { switch (k) { case 0: x = 1 case 1: y = 2; break; } }";
    let parsed = parse("t.hlsl", text, Options::default());
    assert_eq!(parsed.diagnostics.len(), 1, "{:?}", parsed.diagnostics);
    let Item::Function(f) = &parsed.unit.items[0] else {
        panic!()
    };
    let StmtKind::Switch { cases, .. } = &f.body.as_ref().unwrap().stmts[0].kind else {
        panic!()
    };
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[1].stmts.len(), 2);
}

#[test]
fn errors_in_struct_buffer_and_sampler_bodies_stay_inside_the_body() {
    let cases = [
        (
            "struct S { float4 pos : SV_Position float2 uv; float3 n; };\nfloat4 main() : COLOR { return 0; }",
            1,
        ),
        ("cbuffer B { float a b; float c; };\nfloat x;", 1),
        (
            "sampler s = sampler_state { Texture = ; MinFilter = Linear; };\nfloat x;",
            1,
        ),
    ];
    for (text, members) in cases {
        let parsed = parse("t.hlsl", text, Options::default());
        assert_eq!(
            parsed.diagnostics.len(),
            1,
            "{text}: {:?}",
            parsed.diagnostics
        );
        assert_eq!(parsed.unit.items.len(), 2, "{text}");
        let kept = match &parsed.unit.items[0] {
            Item::Struct(s) => s.fields.len(),
            Item::Buffer(b) => b.members.len(),
            Item::Variable(v) => match &v.declarators[0].init {
                Some(Initializer::SamplerState(states, _)) => states.len(),
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        };
        assert_eq!(kept, members, "{text}");
    }
}

#[test]
fn geometry_hull_and_domain_shader_parameters() {
    let text = "struct VS_OUT { float4 p : SV_Position; };\nstruct HS_OUT { float4 p : POS; };\n\
        [maxvertexcount(3)] void GS(triangle float4 p[3] : SV_Position, inout TriangleStream<VS_OUT> s) { s.Append((VS_OUT)0); }\n\
        HS_OUT HS(InputPatch<VS_OUT, 4> ip, uint i : SV_OutputControlPointID) { HS_OUT o; return o; }\n\
        void DS(const OutputPatch<HS_OUT, 4> patch, lineadj float a[4] : A, point float b[1] : B, line float c[2] : C, triangleadj float d[6] : D) { }";
    let unit = ok(text);
    let Item::Function(gs) = &unit.items[2] else {
        panic!()
    };
    assert_eq!(gs.params[0].modifiers, [Modifier::Triangle]);
    assert_eq!(gs.params[1].ty.name.name, "TriangleStream");
    let Item::Function(ds) = &unit.items[4] else {
        panic!()
    };
    let mods: Vec<_> = ds.params[1..].iter().map(|p| p.modifiers[0]).collect();
    assert_eq!(
        mods,
        [
            Modifier::LineAdj,
            Modifier::Point,
            Modifier::Line,
            Modifier::TriangleAdj
        ]
    );
}

#[test]
fn the_d3d10_sampler_block() {
    let unit =
        ok("SamplerState samLinear { Filter = MIN_MAG_MIP_LINEAR; AddressU = Wrap; };\nfloat x;");
    let Item::Variable(var) = &unit.items[0] else {
        panic!()
    };
    let Some(Initializer::SamplerState(states, _)) = &var.declarators[0].init else {
        panic!()
    };
    assert_eq!(states.len(), 2);
    assert_eq!(errors("float x { a = 1; };").len(), 1);
}

#[test]
fn register_spaces_and_one_register_per_profile() {
    let unit = ok(
        "Texture2D tex : register(t0, space1);\nfloat4 c : register(vs_3_0, c0) : register(ps_3_0, c1);\nsampler s : register(s2);",
    );
    let reg = |i: usize, n: usize| {
        let Item::Variable(v) = &unit.items[i] else {
            panic!()
        };
        let r = &v.declarators[0].bindings.registers[n];
        let name = |x: &Option<Ident>| x.as_ref().map(|x| x.name.clone());
        (name(&r.profile), r.slot.name.clone(), name(&r.space))
    };
    assert_eq!(reg(0, 0), (None, "t0".into(), Some("space1".into())));
    assert_eq!(reg(1, 0), (Some("vs_3_0".into()), "c0".into(), None));
    assert_eq!(reg(1, 1), (Some("ps_3_0".into()), "c1".into(), None));
    assert_eq!(reg(2, 0), (None, "s2".into(), None));
    assert_eq!(errors("float a : register(a, b, c, d);").len(), 1);
}

#[test]
fn a_split_double_close_belongs_to_both_type_spans() {
    let parsed = parse(
        "t.hlsl",
        "Texture2D<vector<float, 4>> tex;",
        Options::default(),
    );
    let Item::Variable(var) = &parsed.unit.items[0] else {
        panic!()
    };
    let TypeArg::Type(inner) = &var.ty.args[0] else {
        panic!()
    };
    assert_eq!(parsed.files.text(inner.span), "vector<float, 4>");
    assert_eq!(
        parsed.files.text(var.ty.span),
        "Texture2D<vector<float, 4>>"
    );
}

struct Body;

impl crate::IncludeResolver for Body {
    fn resolve(&mut self, path: &str, _: bool, _: &str) -> Result<(String, String), String> {
        Ok((path.to_string(), "x = 1;".to_string()))
    }
}

#[test]
fn an_error_at_the_end_points_at_the_end_of_the_main_file() {
    let mut resolver = Body;
    let options = Options {
        includes: Some(&mut resolver),
        ..Options::default()
    };
    let parsed = parse("t.hlsl", "void f() {\n#include \"body.h\"\n", options);
    let e: Vec<String> = parsed
        .diagnostics
        .iter()
        .map(|d| d.display(&parsed.files).to_string())
        .collect();
    assert_eq!(
        e,
        ["t.hlsl:3:1: error: expected `}`, found the end of the input"]
    );
}

#[test]
fn the_prelude_runs_first_and_its_macros_apply_to_the_source() {
    let options = Options {
        prelude: Some("#define q1 _qa.x\nfloat4 _qa;\n".to_string()),
        ..Options::default()
    };
    let parsed = parse("t.hlsl", "float f() { return q1; }", options);
    assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
    assert_eq!(parsed.unit.items.len(), 2);
    let Item::Function(f) = &parsed.unit.items[1] else {
        panic!()
    };
    assert_eq!(parsed.files.get(f.span.file).unwrap().name, "t.hlsl");
}
