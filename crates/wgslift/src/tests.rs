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
