use super::*;

fn translate_ok(source: &str, options: Options<'_>) -> Output {
    let output = translate("test.hlsl", source, options).unwrap_or_else(|f| panic!("{f}"));
    let module = naga::front::wgsl::parse_str(&output.wgsl)
        .unwrap_or_else(|e| panic!("{}\n{}", e.emit_to_string(&output.wgsl), output.wgsl));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}\n{}", e.emit_to_string(&output.wgsl), output.wgsl));
    output
}

#[test]
fn the_reflection_names_a_renamed_entry_point() {
    let output = translate_ok(
        "float4 filter(float2 uv : TEXCOORD0) : COLOR0 { return uv.xyxy; }",
        Options {
            entry_point: "filter".to_string(),
            ..Default::default()
        },
    );
    assert_eq!(output.reflection.entry_point, "filter_");
    assert!(output.wgsl.contains("fn filter_("), "{}", output.wgsl);
}

#[test]
fn a_loop_limit_bounds_every_loop_with_one_budget() {
    let source = "float4 main(float2 uv : TEXCOORD0) : COLOR0 {
        float a = 0;
        for (int i = 0; i < 10; i++) { while (uv.x < 2) { a += 1; } }
        do { a -= 1; } while (a > 0);
        return a;
    }";
    let output = translate_ok(
        source,
        Options {
            loop_limit: Some(5000),
            ..Default::default()
        },
    );
    let wgsl = &output.wgsl;
    assert!(wgsl.contains("var<private> loop_budget: u32;"), "{wgsl}");
    assert!(wgsl.contains("loop_budget = 5000u;"), "{wgsl}");
    assert_eq!(
        wgsl.matches("if loop_budget == 0u { break; }").count(),
        3,
        "{wgsl}"
    );
}

#[test]
fn without_a_loop_limit_no_budget_is_printed() {
    let output = translate_ok(
        "float4 main(float2 uv : TEXCOORD0) : COLOR0 { float a = 0; while (a < 3) { a += 1; } return a; }",
        Options::default(),
    );
    assert!(!output.wgsl.contains("loop_budget"), "{}", output.wgsl);
}

#[test]
fn legacy_products_print_valid_wgsl() {
    let source = "float q1; float2 v; float3x2 m;
    static const float2 k = float2(0, -1);
    float4 main(float2 uv : TEXCOORD0) : COLOR0 {
        float2x2 r = float2x2(q1, uv, q1) * q1;
        float3 n = normalize(float3(uv / q1, q1));
        return float4(mul(m, v).xy / k + mul(r, v) + mul(v, r) + uv * q1 / v, dot(n, q1), n.x);
    }";
    let output = translate_ok(source, Options::default());
    assert!(output.wgsl.contains("select("), "{}", output.wgsl);
}

#[test]
fn a_constant_legacy_product_that_overflows_still_folds() {
    let source = "static const float c = 1e20, d = 1e20;
    static const float e = c * d;
    float4 main(float2 uv : TEXCOORD0) : COLOR0 {
        return float4(c * d * 2, e * 2, uv);
    }";
    translate_ok(source, Options::default());
}
