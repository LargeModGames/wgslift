use wgslift_syntax::{FileId, Span};

use super::*;
use crate::ir::{
    Access, Arg, BinaryOp, Expr, ExprKind, Field, Function, Global, Literal, Local, Module, Param,
    ParamDir, Place, Root, Sample, Stmt, StructDef, UnaryOp,
};
use crate::reflection::{IoBinding, UniformMember};
use crate::types::Scalar;

const F2: Type = Type::Vector(Scalar::Float, 2);
const F3: Type = Type::Vector(Scalar::Float, 3);
const F4: Type = Type::Vector(Scalar::Float, 4);

fn validate(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}\n{source}", e.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}\n{source}", e.emit_to_string(source)));
}

fn print_valid(module: &Module) -> (String, Reflection) {
    let (text, reflection) = print(
        module,
        &PrintOptions {
            stage: Stage::Fragment,
            uniformity: Uniformity::DirectiveOff,
            loop_limit: None,
        },
    );
    validate(&text);
    (text, reflection)
}

fn e(kind: ExprKind, ty: Type) -> Expr {
    Expr { kind, ty }
}

fn local(id: usize, ty: Type) -> Expr {
    e(ExprKind::Var(Root::Local(id)), ty)
}

fn global(id: usize, ty: Type) -> Expr {
    e(ExprKind::Var(Root::Global(id)), ty)
}

fn bin(op: BinaryOp, a: Expr, b: Expr, ty: Type) -> Expr {
    e(ExprKind::Binary(op, Box::new(a), Box::new(b)), ty)
}

fn construct(ty: Type, parts: Vec<Expr>) -> Expr {
    e(ExprKind::Construct(parts), ty)
}

fn x_of(v: Expr) -> Expr {
    e(ExprKind::Swizzle(Box::new(v), vec![0]), Type::FLOAT)
}

fn uint(value: u32) -> Expr {
    e(
        ExprKind::Literal(Literal::Uint(value)),
        Type::Scalar(Scalar::Uint),
    )
}

fn var(name: &str, ty: Type) -> Local {
    Local {
        name: name.to_string(),
        ty,
        written: true,
    }
}

fn param(local: usize, dir: ParamDir, semantic: Option<&str>) -> Param {
    Param {
        local,
        dir,
        semantic: semantic.map(str::to_string),
    }
}

fn func(
    name: &str,
    params: Vec<Param>,
    ret: Type,
    locals: Vec<Local>,
    body: Vec<Stmt>,
) -> Function {
    Function {
        name: name.to_string(),
        params,
        ret,
        ret_semantic: None,
        locals,
        body,
        span: Span::new(FileId(0), 0, 0),
    }
}

fn glob(name: &str, ty: Type, space: Space) -> Global {
    Global {
        name: name.to_string(),
        ty,
        space,
        init: None,
        entry_init: None,
        span: Span::new(FileId(0), 0, 0),
    }
}

fn set(id: usize, value: Expr) -> Stmt {
    Stmt::Assign(
        Place {
            root: Root::Local(id),
            path: Vec::new(),
        },
        value,
    )
}

/// `float4 main() : COLOR0 { <body>; return <ret>; }` with these locals.
fn entry(locals: Vec<Local>, mut body: Vec<Stmt>, ret: Expr) -> Function {
    body.push(Stmt::Return(Some(ret)));
    let mut f = func("main", Vec::new(), F4, locals, body);
    f.ret_semantic = Some("COLOR0".to_string());
    f
}

fn white() -> Expr {
    e(ExprKind::Splat(Box::new(Expr::float(1.0))), F4)
}

fn one_function(f: Function) -> Module {
    Module {
        functions: vec![f],
        ..Module::default()
    }
}

#[test]
fn a_minimal_entry_point_prints_exactly() {
    let mut main = func(
        "main",
        vec![param(0, ParamDir::In, Some("TEXCOORD0"))],
        F4,
        vec![Local {
            name: "uv".to_string(),
            ty: F2,
            written: false,
        }],
        vec![Stmt::Return(Some(construct(
            F4,
            vec![local(0, F2), Expr::float(0.0), Expr::float(1.0)],
        )))],
    );
    main.ret_semantic = Some("COLOR".to_string());
    let (text, reflection) = print_valid(&one_function(main));
    let expected = "\
diagnostic(off, derivative_uniformity);

fn main_1(uv: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(uv, 0.0f, 1.0f);
}

struct FragmentInput {
    @location(0) uv: vec2<f32>,
}

struct FragmentOutput {
    @location(0) result: vec4<f32>,
}

@fragment
fn main(input: FragmentInput) -> FragmentOutput {
    let result = main_1(input.uv);
    var output: FragmentOutput;
    output.result = result;
    return output;
}
";
    assert_eq!(text, expected);
    assert_eq!(
        reflection.inputs,
        vec![IoBinding {
            name: "uv".to_string(),
            semantic: "TEXCOORD0".to_string(),
            location: Some(0),
            builtin: None,
            wgsl_type: "vec2<f32>".to_string(),
        }]
    );
    assert_eq!(reflection.outputs[0].location, Some(0));
    assert_eq!(reflection.outputs[0].semantic, "COLOR");
}

#[test]
fn keep_uniformity_prints_no_directive() {
    let module = one_function(entry(Vec::new(), Vec::new(), white()));
    let (text, _) = print(
        &module,
        &PrintOptions {
            stage: Stage::Fragment,
            uniformity: Uniformity::Keep,
            loop_limit: None,
        },
    );
    validate(&text);
    assert!(text.starts_with("fn main_1()"), "{text}");
}

#[test]
fn uniforms_of_each_type_are_laid_out_and_read() {
    let light = Type::Struct(0);
    let types = vec![
        ("time", Type::FLOAT),
        ("offset", F2),
        ("tint", F3),
        ("count", Type::INT),
        ("mask", Type::Scalar(Scalar::Uint)),
        ("enabled", Type::BOOL),
        ("flags", Type::Vector(Scalar::Bool, 3)),
        ("view", Type::Matrix { rows: 4, cols: 4 }),
        ("basis", Type::Matrix { rows: 3, cols: 3 }),
        ("rot", Type::Matrix { rows: 2, cols: 2 }),
        ("skew", Type::Matrix { rows: 3, cols: 2 }),
        ("colors", Type::Array(Box::new(F4), 3)),
        ("weights", Type::Array(Box::new(Type::FLOAT), 4)),
        ("toggles", Type::Array(Box::new(Type::BOOL), 2)),
        ("light", light.clone()),
        ("rest", F4),
    ];
    let globals: Vec<Global> = types
        .iter()
        .map(|(n, t)| glob(n, t.clone(), Space::Uniform))
        .collect();
    let mut locals: Vec<Local> = types
        .iter()
        .map(|(n, t)| var(&format!("l_{n}"), t.clone()))
        .collect();
    let mut body: Vec<Stmt> = (0..types.len())
        .map(|i| Stmt::Local {
            id: i,
            init: Some(global(i, types[i].1.clone())),
        })
        .collect();
    let w = locals.len();
    locals.push(var("w", Type::FLOAT));
    let index = |base: Expr, i: i32, ty: Type| {
        e(ExprKind::Index(Box::new(base), Box::new(Expr::int(i))), ty)
    };
    // toggles[0] && light.on ? weights[1] : view[1].y
    let on = e(ExprKind::Field(Box::new(global(14, light)), 1), Type::BOOL);
    let row = index(global(7, types[7].1.clone()), 1, F4);
    let w_value = e(
        ExprKind::Select {
            cond: Box::new(bin(
                BinaryOp::And,
                index(global(13, types[13].1.clone()), 0, Type::BOOL),
                on,
                Type::BOOL,
            )),
            accept: Box::new(index(global(12, types[12].1.clone()), 1, Type::FLOAT)),
            reject: Box::new(e(ExprKind::Swizzle(Box::new(row), vec![1]), Type::FLOAT)),
        },
        Type::FLOAT,
    );
    body.push(Stmt::Local {
        id: w,
        init: Some(w_value),
    });
    let module = Module {
        structs: vec![StructDef {
            name: "Light".to_string(),
            fields: vec![
                Field {
                    name: "dir".to_string(),
                    ty: F3,
                    semantic: None,
                },
                Field {
                    name: "on".to_string(),
                    ty: Type::BOOL,
                    semantic: None,
                },
            ],
        }],
        globals,
        functions: vec![entry(locals, body, global(15, F4))],
        entry: 0,
    };
    let (text, reflection) = print_valid(&module);
    for needle in [
        "@group(0) @binding(0) var<uniform> uniforms: Uniforms;",
        "struct UniformElement {\n    @size(16) v: f32,\n}",
        "struct Light_uniform {\n    dir: vec3<f32>,\n    on: u32,\n}",
        "    @align(16) weights: array<UniformElement, 4>,",
        "(uniforms.enabled != 0u)",
        "(uniforms.flags != vec3<u32>(0u))",
        "array<f32, 4>(uniforms.weights[0].v, uniforms.weights[1].v,",
        "Light(uniforms.light.dir, (uniforms.light.on != 0u))",
        "select(uniforms.view[1i].y, uniforms.weights[1i].v, \
         (uniforms.toggles[0i].v != 0u) & (uniforms.light.on != 0u))",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in\n{text}");
    }
    let buffer = reflection.uniform_buffer.unwrap();
    let member = |name: &str| -> UniformMember {
        buffer
            .members
            .iter()
            .find(|m| m.name == name)
            .unwrap()
            .clone()
    };
    let at = |name: &str| (member(name).offset, member(name).size);
    assert_eq!(at("time"), (0, 4));
    assert_eq!(at("offset"), (8, 8));
    assert_eq!(at("tint"), (16, 12));
    assert_eq!(at("count"), (28, 4));
    assert_eq!(at("mask"), (32, 4));
    assert_eq!(at("enabled"), (36, 4));
    assert_eq!(member("enabled").wgsl_type, "u32");
    assert_eq!(at("flags"), (48, 12));
    assert_eq!(at("view"), (64, 64));
    assert_eq!(at("basis"), (128, 48));
    assert_eq!(at("rot"), (176, 16));
    assert_eq!(at("skew"), (192, 24));
    assert_eq!(at("colors"), (224, 48));
    assert_eq!(at("weights"), (272, 64));
    assert_eq!(at("toggles"), (336, 32));
    assert_eq!(at("light"), (368, 16));
    assert_eq!(member("light").wgsl_type, "Light_uniform");
    assert_eq!(at("rest"), (384, 16));
    assert_eq!(buffer.size, 400);
    assert_eq!(buffer.binding, 0);
}

#[test]
fn a_bool_uniform_size_rounds_up_to_sixteen() {
    let module = Module {
        globals: vec![glob("on", Type::BOOL, Space::Uniform)],
        functions: vec![entry(
            Vec::new(),
            Vec::new(),
            e(
                ExprKind::Select {
                    cond: Box::new(global(0, Type::BOOL)),
                    accept: Box::new(white()),
                    reject: Box::new(e(ExprKind::Splat(Box::new(Expr::float(0.0))), F4)),
                },
                F4,
            ),
        )],
        ..Module::default()
    };
    let (_, reflection) = print_valid(&module);
    assert_eq!(reflection.uniform_buffer.unwrap().size, 16);
}

#[test]
fn samplers_of_each_dimension_bind_a_texture_and_a_sampler() {
    let dims = [
        SamplerDim::D1,
        SamplerDim::D2,
        SamplerDim::D3,
        SamplerDim::Cube,
        SamplerDim::Any,
    ];
    let mut globals: Vec<Global> = dims
        .iter()
        .enumerate()
        .map(|(i, d)| glob(&format!("s{i}"), Type::Sampler(*d), Space::Resource))
        .collect();
    globals.push(glob("time", Type::FLOAT, Space::Uniform));
    let coords = |d: SamplerDim| match d {
        SamplerDim::D1 => Expr::float(0.5),
        SamplerDim::D3 | SamplerDim::Cube => e(ExprKind::Splat(Box::new(Expr::float(0.5))), F3),
        SamplerDim::D2 | SamplerDim::Any => e(ExprKind::Splat(Box::new(Expr::float(0.5))), F2),
    };
    let mut sum = white();
    for (i, d) in dims.iter().enumerate() {
        let kinds = [
            Sample::Plain,
            Sample::Level(Box::new(Expr::float(0.0))),
            Sample::Bias(Box::new(global(5, Type::FLOAT))),
            Sample::Grad(Box::new(coords(*d)), Box::new(coords(*d))),
        ];
        for kind in kinds {
            let sample = e(
                ExprKind::Sample {
                    sampler: i,
                    coords: Box::new(coords(*d)),
                    kind,
                },
                F4,
            );
            sum = bin(BinaryOp::Add, sum, sample, F4);
        }
    }
    let module = Module {
        globals,
        functions: vec![entry(Vec::new(), Vec::new(), sum)],
        ..Module::default()
    };
    let (text, reflection) = print_valid(&module);
    for needle in [
        "@group(0) @binding(1) var s0_texture: texture_1d<f32>;",
        "@group(0) @binding(2) var s0_sampler: sampler;",
        "@group(0) @binding(5) var s2_texture: texture_3d<f32>;",
        "@group(0) @binding(7) var s3_texture: texture_cube<f32>;",
        "@group(0) @binding(9) var s4_texture: texture_2d<f32>;",
        "textureSampleLevel(s0_texture, s0_sampler, 0.5f, 0.0f)",
        "textureSampleGrad(s1_texture, s1_sampler, vec2<f32>(0.5f), vec2<f32>(0.5f), vec2<f32>(0.5f))",
        "textureSampleBias(s3_texture, s3_sampler, vec3<f32>(0.5f), uniforms.time)",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in\n{text}");
    }
    assert!(!text.contains("textureSampleBias(s0_texture"), "{text}");
    let bindings: Vec<(SamplerDim, u32, u32)> = reflection
        .textures
        .iter()
        .map(|t| (t.dim, t.texture_binding, t.sampler_binding))
        .collect();
    assert_eq!(
        bindings,
        vec![
            (SamplerDim::D1, 1, 2),
            (SamplerDim::D2, 3, 4),
            (SamplerDim::D3, 5, 6),
            (SamplerDim::Cube, 7, 8),
            (SamplerDim::D2, 9, 10),
        ]
    );
}

#[test]
fn every_statement_prints_valid_wgsl() {
    let locals = vec![
        var("i", Type::INT),
        var("v", F4),
        var("s", Type::Struct(0)),
        var("a", Type::Array(Box::new(Type::FLOAT), 3)),
        var("m", Type::Matrix { rows: 2, cols: 3 }),
    ];
    let i = || local(0, Type::INT);
    let assign = |root: usize, path: Vec<Access>, value: Expr| {
        Stmt::Assign(
            Place {
                root: Root::Local(root),
                path,
            },
            value,
        )
    };
    let lt = |a: Expr, b: i32| bin(BinaryOp::Lt, a, Expr::int(b), Type::BOOL);
    let body = vec![
        Stmt::Local { id: 0, init: None },
        Stmt::Local {
            id: 1,
            init: Some(white()),
        },
        Stmt::Local { id: 2, init: None },
        Stmt::Local { id: 3, init: None },
        Stmt::Local { id: 4, init: None },
        assign(1, vec![Access::Component(2)], Expr::float(0.25)),
        assign(
            2,
            vec![Access::Field(0), Access::Component(1)],
            Expr::float(2.0),
        ),
        assign(3, vec![Access::Index(i())], Expr::float(3.0)),
        assign(
            4,
            vec![Access::Index(Expr::int(1)), Access::Component(0)],
            Expr::float(4.0),
        ),
        Stmt::Eval(e(ExprKind::Call(0, Vec::new()), Type::Void)),
        Stmt::Eval(e(
            ExprKind::Builtin("sin", vec![Expr::float(1.0)]),
            Type::FLOAT,
        )),
        Stmt::If(
            lt(i(), 2),
            vec![set(0, Expr::int(5))],
            vec![Stmt::If(lt(i(), 0), vec![Stmt::Discard], Vec::new())],
        ),
        Stmt::Loop {
            body: vec![
                Stmt::If(lt(i(), 0), vec![Stmt::Break], Vec::new()),
                Stmt::If(lt(i(), 1), vec![Stmt::Continue], Vec::new()),
            ],
            continuing: vec![set(0, bin(BinaryOp::Add, i(), Expr::int(1), Type::INT))],
            break_if: Some(bin(BinaryOp::Ge, i(), Expr::int(10), Type::BOOL)),
        },
        Stmt::Loop {
            body: vec![Stmt::Break],
            continuing: Vec::new(),
            break_if: None,
        },
        Stmt::Switch(
            i(),
            vec![
                (
                    vec![Some(0), Some(1)],
                    vec![assign(1, vec![Access::Component(0)], Expr::float(0.0))],
                ),
                (vec![Some(-2)], vec![Stmt::Break]),
            ],
        ),
        Stmt::Switch(
            i(),
            vec![
                (vec![Some(3), None], vec![set(0, Expr::int(1))]),
                (vec![Some(4)], Vec::new()),
            ],
        ),
        Stmt::Block(vec![set(0, Expr::int(7))]),
    ];
    let helper = func(
        "helper",
        Vec::new(),
        Type::Void,
        Vec::new(),
        vec![Stmt::Return(None)],
    );
    let module = Module {
        structs: vec![StructDef {
            name: "S".to_string(),
            fields: vec![Field {
                name: "p".to_string(),
                ty: F2,
                semantic: None,
            }],
        }],
        functions: vec![helper, entry(locals, body, local(1, F4))],
        entry: 1,
        ..Module::default()
    };
    let (text, _) = print_valid(&module);
    let expected_switch = "    switch i {
        case 0i, 1i {
            v.x = 0.0f;
        }
        case -2i {
            break;
        }
        default {}
    }
";
    for needle in [
        expected_switch,
        "        case 3i, default {\n",
        "        continuing {\n            i = i + 1i;\n            break if i >= 10i;\n        }\n",
        "    if i < 2i {\n        i = 5i;\n    } else {\n        if i < 0i {\n            discard;\n",
        "    loop {\n        break;\n    }\n",
        "    {\n        i = 7i;\n    }\n",
        "    v.z = 0.25f;\n",
        "    s.p.y = 2.0f;\n",
        "    a[i] = 3.0f;\n",
        "    m[1i].x = 4.0f;\n",
        "    helper();\n",
        "    _ = sin(1.0f);\n",
        "    var i: i32;\n",
        "fn helper() {\n    return;\n}\n",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in\n{text}");
    }
}

#[test]
fn every_expression_prints_valid_wgsl() {
    let s = Type::Struct(0);
    let m = Type::Matrix { rows: 4, cols: 4 };
    let floats = Type::Array(Box::new(Type::FLOAT), 2);
    let locals = vec![
        var("f", Type::FLOAT),
        var("n", Type::INT),
        var("u", Type::Scalar(Scalar::Uint)),
        var("b", Type::BOOL),
        var("v", F4),
        var("m", m.clone()),
        var("s", s.clone()),
        var("a", floats.clone()),
        var("iv", Type::Vector(Scalar::Int, 3)),
        var("bv", Type::Vector(Scalar::Bool, 2)),
    ];
    let f = || local(0, Type::FLOAT);
    let n = || local(1, Type::INT);
    let u = || local(2, Type::Scalar(Scalar::Uint));
    let b = || local(3, Type::BOOL);
    let v = || local(4, F4);
    let un = |op, x: Expr| {
        let ty = x.ty.clone();
        e(ExprKind::Unary(op, Box::new(x)), ty)
    };
    let xy = || e(ExprKind::Swizzle(Box::new(v()), vec![0, 1]), F2);
    let inits = [
        Expr::float(-1.5),
        Expr::int(i32::MIN),
        uint(7),
        Expr::bool(true),
        white(),
        construct(m.clone(), (0..16).map(|k| Expr::float(k as f32)).collect()),
        construct(s.clone(), vec![xy()]),
        construct(floats.clone(), vec![Expr::float(1e-7), Expr::float(3.0e20)]),
        e(
            ExprKind::Splat(Box::new(Expr::int(-3))),
            Type::Vector(Scalar::Int, 3),
        ),
        e(
            ExprKind::Convert(Box::new(xy())),
            Type::Vector(Scalar::Bool, 2),
        ),
    ];
    let mut body: Vec<Stmt> = inits
        .into_iter()
        .enumerate()
        .map(|(id, init)| Stmt::Local {
            id,
            init: Some(init),
        })
        .collect();
    for op in [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Rem,
        BinaryOp::And,
        BinaryOp::Or,
        BinaryOp::Xor,
    ] {
        body.push(set(1, bin(op, n(), Expr::int(3), Type::INT)));
    }
    body.push(set(1, bin(BinaryOp::Shl, n(), uint(2), Type::INT)));
    body.push(set(1, bin(BinaryOp::Shr, n(), uint(1), Type::INT)));
    body.push(set(1, un(UnaryOp::BitNot, n())));
    body.push(set(1, un(UnaryOp::Neg, un(UnaryOp::Neg, n()))));
    body.push(set(
        0,
        bin(BinaryOp::Rem, f(), Expr::float(2.0), Type::FLOAT),
    ));
    for op in [
        BinaryOp::Eq,
        BinaryOp::Ne,
        BinaryOp::Lt,
        BinaryOp::Le,
        BinaryOp::Gt,
        BinaryOp::Ge,
    ] {
        body.push(set(3, bin(op, f(), Expr::float(0.0), Type::BOOL)));
    }
    body.push(set(
        3,
        bin(BinaryOp::Or, b(), un(UnaryOp::Not, b()), Type::BOOL),
    ));
    let u_is_3 = bin(BinaryOp::Eq, u(), uint(3), Type::BOOL);
    body.push(set(3, bin(BinaryOp::And, b(), u_is_3, Type::BOOL)));
    body.push(set(
        2,
        e(ExprKind::Convert(Box::new(f())), Type::Scalar(Scalar::Uint)),
    ));
    body.push(set(4, bin(BinaryOp::MatMul, local(5, m.clone()), v(), F4)));
    body.push(set(4, bin(BinaryOp::MatMul, v(), local(5, m.clone()), F4)));
    let mask = e(
        ExprKind::Convert(Box::new(v())),
        Type::Vector(Scalar::Bool, 4),
    );
    body.push(set(
        4,
        e(
            ExprKind::Select {
                cond: Box::new(mask),
                accept: Box::new(v()),
                reject: Box::new(white()),
            },
            F4,
        ),
    ));
    let element = e(
        ExprKind::Index(Box::new(local(7, floats)), Box::new(n())),
        Type::FLOAT,
    );
    let field = x_of(e(ExprKind::Field(Box::new(local(6, s.clone())), 0), F2));
    body.push(set(0, bin(BinaryOp::Add, element, field, Type::FLOAT)));
    let row = e(
        ExprKind::Index(Box::new(local(5, m)), Box::new(Expr::int(2))),
        F4,
    );
    body.push(set(
        0,
        e(
            ExprKind::Index(Box::new(row), Box::new(Expr::int(3))),
            Type::FLOAT,
        ),
    ));
    body.push(set(
        0,
        e(ExprKind::Builtin("dot", vec![v(), white()]), Type::FLOAT),
    ));
    body.push(set(
        0,
        e(ExprKind::Call(0, vec![Arg::Value(f())]), Type::FLOAT),
    ));
    let twice = func(
        "twice",
        vec![param(0, ParamDir::In, None)],
        Type::FLOAT,
        vec![Local {
            name: "x".to_string(),
            ty: Type::FLOAT,
            written: false,
        }],
        vec![Stmt::Return(Some(bin(
            BinaryOp::Mul,
            local(0, Type::FLOAT),
            Expr::float(2.0),
            Type::FLOAT,
        )))],
    );
    let module = Module {
        structs: vec![StructDef {
            name: "S".to_string(),
            fields: vec![Field {
                name: "p".to_string(),
                ty: F2,
                semantic: None,
            }],
        }],
        functions: vec![twice, entry(locals, body, v())],
        entry: 1,
        ..Module::default()
    };
    let (text, _) = print_valid(&module);
    for needle in [
        "fn twice(x: f32) -> f32 {\n    return x * 2.0f;\n}",
        "var f: f32 = -1.5f;",
        "var n: i32 = i32(-2147483648);",
        "var u: u32 = 7u;",
        "var b: bool = true;",
        "var s: S = S(v.xy);",
        "array<f32, 2>(1e-7f, 3e20f)",
        "vec3<i32>(-3i)",
        "vec2<bool>(v.xy)",
        "n = n % 3i;",
        "n = n ^ 3i;",
        "n = n << 2u;",
        "n = ~n;",
        "n = -(-n);",
        "f = f % 2.0f;",
        "b = f <= 0.0f;",
        "b = b | (!b);",
        "b = b & (u == 3u);",
        "u = u32(f);",
        "v = m * v;",
        "v = v * m;",
        "select(vec4<f32>(1.0f), v, vec4<bool>(v))",
        "f = a[n] + s.p.x;",
        "f = m[2i][3i];",
        "f = dot(v, vec4<f32>(1.0f));",
        "f = twice(f);",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in\n{text}");
    }
}

#[test]
fn out_and_inout_parameters_pass_pointers() {
    // void inner(out float o) { o = 1; }
    let inner = func(
        "inner",
        vec![param(0, ParamDir::Out, None)],
        Type::Void,
        vec![var("o", Type::FLOAT)],
        vec![set(0, Expr::float(1.0))],
    );
    // void outer(inout float4 c, float k, out float o) { k *= 2; c.x = c.y + k; inner(o); }
    let c_y = e(
        ExprKind::Swizzle(Box::new(local(0, F4)), vec![1]),
        Type::FLOAT,
    );
    let outer = func(
        "outer",
        vec![
            param(0, ParamDir::InOut, None),
            param(1, ParamDir::In, None),
            param(2, ParamDir::Out, None),
        ],
        Type::Void,
        vec![var("c", F4), var("k", Type::FLOAT), var("o", Type::FLOAT)],
        vec![
            set(
                1,
                bin(
                    BinaryOp::Mul,
                    local(1, Type::FLOAT),
                    Expr::float(2.0),
                    Type::FLOAT,
                ),
            ),
            Stmt::Assign(
                Place {
                    root: Root::Local(0),
                    path: vec![Access::Component(0)],
                },
                bin(BinaryOp::Add, c_y, local(1, Type::FLOAT), Type::FLOAT),
            ),
            Stmt::Eval(e(ExprKind::Call(0, vec![Arg::Ref(2)]), Type::Void)),
        ],
    );
    let call = e(
        ExprKind::Call(
            1,
            vec![Arg::Ref(0), Arg::Value(Expr::float(0.5)), Arg::Ref(1)],
        ),
        Type::Void,
    );
    let main = entry(
        vec![var("color", F4), var("unused", Type::FLOAT)],
        vec![
            Stmt::Local {
                id: 0,
                init: Some(white()),
            },
            Stmt::Local { id: 1, init: None },
            Stmt::Eval(call),
        ],
        local(0, F4),
    );
    let module = Module {
        functions: vec![inner, outer, main],
        entry: 2,
        ..Module::default()
    };
    let (text, _) = print_valid(&module);
    for needle in [
        "fn inner(o: ptr<function, f32>) {\n    *o = 1.0f;\n}",
        "fn outer(c: ptr<function, vec4<f32>>, k_in: f32, o: ptr<function, f32>) {",
        "    var k: f32 = k_in;",
        "    (*c).x = (*c).y + k;",
        "    inner(o);",
        "    outer(&color, 0.5f, &unused);",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in\n{text}");
    }
}

#[test]
fn the_entry_wrapper_flattens_inputs_and_collects_outputs() {
    let fields = |list: &[(&str, Type, &str)]| -> Vec<Field> {
        list.iter()
            .map(|(n, t, s)| Field {
                name: n.to_string(),
                ty: t.clone(),
                semantic: (!s.is_empty()).then(|| s.to_string()),
            })
            .collect()
    };
    let structs = vec![
        StructDef {
            name: "PS_IN".to_string(),
            fields: fields(&[
                ("uv", F2, "TEXCOORD1"),
                ("tint", F4, "COLOR0"),
                ("id", Type::INT, "TEXCOORD2"),
                ("extra", F3, ""),
                ("normal", F3, "NORMAL"),
                ("xf", Type::Matrix { rows: 2, cols: 2 }, "TEXCOORD4"),
            ]),
        },
        StructDef {
            name: "PS_OUT".to_string(),
            fields: fields(&[("color", F4, "COLOR0"), ("depth", Type::FLOAT, "DEPTH")]),
        },
    ];
    let mut globals = vec![glob("scale", Type::FLOAT, Space::Private)];
    globals[0].entry_init = Some(bin(
        BinaryOp::Mul,
        Expr::float(2.0),
        Expr::float(3.0),
        Type::FLOAT,
    ));
    let input = Type::Struct(0);
    let tint = e(ExprKind::Field(Box::new(local(0, input.clone())), 1), F4);
    let main = func(
        "ps_main",
        vec![
            param(0, ParamDir::In, None),
            param(1, ParamDir::In, Some("VPOS")),
            param(2, ParamDir::Out, Some("COLOR1")),
            param(3, ParamDir::In, Some("VFACE")),
        ],
        Type::Struct(1),
        vec![
            var("i", input),
            var("pos", F2),
            var("glow", F4),
            var("face", Type::FLOAT),
            var("o", Type::Struct(1)),
        ],
        vec![
            Stmt::Local { id: 4, init: None },
            set(2, tint),
            Stmt::Return(Some(local(4, Type::Struct(1)))),
        ],
    );
    let module = Module {
        structs,
        globals,
        functions: vec![main],
        entry: 0,
    };
    let (text, reflection) = print_valid(&module);
    for needle in [
        "@fragment\nfn ps_main(input: FragmentInput) -> FragmentOutput {\n    scale = 2.0f * 3.0f;\n",
        "    var glow: vec4<f32>;\n",
        "(input.pos.xy - vec2<f32>(0.5f))",
        "f32(select(-1.0f, 1.0f, input.face))",
        "    @location(2) @interpolate(flat) id: i32,\n",
        "    @builtin(position) pos: vec4<f32>,\n",
        "PS_IN(input.uv, input.tint, input.id, vec3<f32>(), input.normal, \
         mat2x2<f32>(input.xf_0, input.xf_1))",
        "    output.color = result.color;\n    output.depth = result.depth;\n    output.glow = glow;\n",
        "    @builtin(frag_depth) depth: f32,\n",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in\n{text}");
    }
    let summary = |list: &[IoBinding]| -> Vec<(String, Option<u32>, Option<String>)> {
        list.iter()
            .map(|b| (b.semantic.clone(), b.location, b.builtin.clone()))
            .collect()
    };
    let row = |s: &str, l: Option<u32>, b: Option<&str>| (s.to_string(), l, b.map(str::to_string));
    assert_eq!(
        summary(&reflection.inputs),
        vec![
            row("TEXCOORD1", Some(1), None),
            row("COLOR0", Some(8), None),
            row("TEXCOORD2", Some(2), None),
            row("NORMAL", Some(0), None),
            row("TEXCOORD4", Some(4), None),
            row("TEXCOORD5", Some(5), None),
            row("VPOS", None, Some("position")),
            row("VFACE", None, Some("front_facing")),
        ]
    );
    assert_eq!(
        summary(&reflection.outputs),
        vec![
            row("COLOR0", Some(0), None),
            row("DEPTH", None, Some("frag_depth")),
            row("COLOR1", Some(1), None),
        ]
    );
    assert_eq!(reflection.inputs[3].name, "normal");
    assert_eq!(reflection.inputs[3].wgsl_type, "vec3<f32>");
}

#[test]
fn reserved_and_colliding_names_are_renamed() {
    let names = [
        "filter", "mod", "ref", "sampler", "vec2", "__x", "_", "a", "a",
    ];
    let locals: Vec<Local> = names.iter().map(|n| var(n, Type::FLOAT)).collect();
    let mut body: Vec<Stmt> = (0..8)
        .map(|i| Stmt::Local {
            id: i,
            init: Some(Expr::float(i as f32)),
        })
        .collect();
    // The second `a` lives in a nested block.
    body.push(Stmt::Block(vec![Stmt::Local {
        id: 8,
        init: Some(local(7, Type::FLOAT)),
    }]));
    let sum = (1..8).fold(local(0, Type::FLOAT), |acc, i| {
        bin(BinaryOp::Add, acc, local(i, Type::FLOAT), Type::FLOAT)
    });
    let mut main = entry(locals, body, e(ExprKind::Splat(Box::new(sum)), F4));
    main.name = "uniform".to_string();
    let module = Module {
        structs: vec![StructDef {
            name: "vec3f".to_string(),
            fields: vec![Field {
                name: "ref".to_string(),
                ty: Type::FLOAT,
                semantic: None,
            }],
        }],
        globals: vec![
            glob("storage", Type::FLOAT, Space::Private),
            glob("uniforms", Type::FLOAT, Space::Uniform),
            glob("main", Type::Struct(0), Space::Const),
        ],
        functions: vec![main],
        entry: 0,
    };
    let (text, _) = print_valid(&module);
    for needle in [
        "struct vec3f_ {\n    ref_: f32,\n}",
        "var<private> storage_: f32;",
        "const main: vec3f_ = vec3f_();",
        "    uniforms: f32,",
        "var<uniform> uniforms_1: Uniforms;",
        "var filter_: f32 = 0.0f;",
        "var mod_: f32 = 1.0f;",
        "var ref_: f32 = 2.0f;",
        "var sampler_: f32 = 3.0f;",
        "var vec2_: f32 = 4.0f;",
        "var v__x: f32 = 5.0f;",
        "var v_: f32 = 6.0f;",
        "var a: f32 = 7.0f;",
        "var a_1: f32 = a;",
        "fn uniform__1() -> vec4<f32> {",
        "fn uniform_() -> FragmentOutput {",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in\n{text}");
    }
}

#[test]
fn deep_expressions_are_split_into_lets() {
    let mut value = local(0, Type::FLOAT);
    for i in 0..300 {
        value = bin(BinaryOp::Add, Expr::float(i as f32), value, Type::FLOAT);
    }
    let mut call = local(0, Type::FLOAT);
    for _ in 0..300 {
        call = e(ExprKind::Builtin("abs", vec![call]), Type::FLOAT);
    }
    let main = entry(
        vec![var("x", Type::FLOAT), var("y", Type::FLOAT)],
        vec![
            Stmt::Local {
                id: 0,
                init: Some(Expr::float(1.0)),
            },
            Stmt::Local {
                id: 1,
                init: Some(value),
            },
            Stmt::If(
                bin(BinaryOp::Gt, call, Expr::float(0.0), Type::BOOL),
                vec![Stmt::Discard],
                Vec::new(),
            ),
        ],
        e(ExprKind::Splat(Box::new(local(1, Type::FLOAT))), F4),
    );
    let (text, _) = print_valid(&one_function(main));
    assert!(text.contains("    let t = "), "{text}");
    assert!(text.contains("    let t_4 = abs("), "{text}");
}

#[test]
fn an_empty_struct_gets_a_member() {
    let module = Module {
        structs: vec![StructDef {
            name: "Empty".to_string(),
            fields: Vec::new(),
        }],
        functions: vec![entry(
            vec![var("e", Type::Struct(0))],
            vec![Stmt::Local {
                id: 0,
                init: Some(construct(Type::Struct(0), Vec::new())),
            }],
            white(),
        )],
        ..Module::default()
    };
    let (text, _) = print_valid(&module);
    assert!(text.contains("struct Empty {\n    pad: u32,\n}"), "{text}");
    assert!(text.contains("var e: Empty = Empty();"), "{text}");
}

#[test]
fn private_and_const_globals_print_their_values() {
    let mut globals = vec![
        glob("gain", Type::FLOAT, Space::Const),
        glob("bias", F2, Space::Private),
        glob("zero", Type::INT, Space::Private),
    ];
    globals[0].init = Some(Expr::float(0.5));
    globals[1].init = Some(e(ExprKind::Splat(Box::new(global(0, Type::FLOAT))), F2));
    let ret = construct(
        F4,
        vec![global(1, F2), global(0, Type::FLOAT), Expr::float(1.0)],
    );
    let module = Module {
        globals,
        functions: vec![entry(Vec::new(), Vec::new(), ret)],
        ..Module::default()
    };
    let (text, reflection) = print_valid(&module);
    assert!(text.contains("const gain: f32 = 0.5f;\nvar<private> bias: vec2<f32> = vec2<f32>(gain);\nvar<private> zero: i32;\n"), "{text}");
    assert_eq!(reflection.uniform_buffer, None);
}

#[test]
fn float_literals_keep_their_value_and_type() {
    let show = |f: f32| body::literal_text(Literal::Float(f)).0;
    assert_eq!(show(1.0), "1.0f");
    assert_eq!(show(0.1), "0.1f");
    assert_eq!(show(-2.5), "-2.5f");
    assert_eq!(show(1e-7), "1e-7f");
    assert_eq!(show(f32::INFINITY), "3.4028235e38f");
    assert_eq!(show(f32::NAN), "0.0f");
    assert_eq!(
        body::literal_text(Literal::Int(-4)),
        ("-4i".to_string(), false)
    );
    assert_eq!(body::literal_text(Literal::Uint(4)).0, "4u");
}

fn translate_valid(source: &str) -> (String, Reflection) {
    let output = crate::translate("test.hlsl", source, crate::Options::default())
        .unwrap_or_else(|f| panic!("{f}"));
    validate(&output.wgsl);
    (output.wgsl, output.reflection)
}

#[test]
fn the_reflection_gives_array_strides_and_struct_fields() {
    let (_, reflection) = translate_valid(
        "struct Light { float3 dir; bool on; float w[2]; };\n\
         float weights[4]; Light light;\n\
         float4 main() : COLOR0 { return weights[1] + light.w[0] + (light.on ? 1 : 0); }",
    );
    let buffer = reflection.uniform_buffer.unwrap();
    let weights = &buffer.members[0];
    assert_eq!(
        (weights.offset, weights.size, weights.stride),
        (0, 64, Some(16))
    );
    assert_eq!(weights.members.len(), 1);
    assert_eq!((weights.members[0].offset, weights.members[0].size), (0, 4));
    assert_eq!(weights.members[0].wgsl_type, "f32");
    let light = &buffer.members[1];
    assert_eq!(light.stride, None);
    let fields: Vec<(&str, u32, u32, &str)> = light
        .members
        .iter()
        .map(|m| (m.name.as_str(), m.offset, m.size, m.wgsl_type.as_str()))
        .collect();
    assert_eq!(
        fields,
        [
            ("dir", 0, 12, "vec3<f32>"),
            ("on", 12, 4, "u32"),
            ("w", 16, 32, "array<UniformElement, 2>"),
        ]
    );
    assert_eq!(light.members[2].stride, Some(16));
}
#[test]
fn an_empty_struct_uniform_takes_sixteen_bytes() {
    let (_, reflection) =
        translate_valid("struct E {}; E e; float f; float4 main() : COLOR0 { E x = e; return f; }");
    let buffer = reflection.uniform_buffer.unwrap();
    let offsets: Vec<(u32, u32)> = buffer.members.iter().map(|m| (m.offset, m.size)).collect();
    assert_eq!(offsets, [(0, 16), (16, 4)]);
    let (_, reflection) =
        translate_valid("struct E {}; E a[2]; float4 main() : COLOR0 { E x = a[1]; return 0; }");
    let buffer = reflection.uniform_buffer.unwrap();
    assert_eq!(buffer.members[0].stride, Some(16));
}
#[test]
fn the_lowest_switch_label_prints_as_valid_wgsl() {
    let (text, _) = translate_valid(
        "float4 main(float2 uv : TEXCOORD0) : COLOR0 { int i = (int)uv.x; \
         switch (i) { case 0x80000000: return 1; default: return 0; } }",
    );
    assert!(text.contains("case i32(-2147483648) {"), "{text}");
}
