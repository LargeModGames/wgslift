//! The MilkDrop 2 shader environment and the preset reading for the corpus harness.

use wgslift::syntax::ast::Item;
use wgslift::syntax::lexer::lex;
use wgslift::syntax::{FileId, Punct, TokenKind};

/// Macros that every MilkDrop shader can use.
pub const PRELUDE: &str = include_str!("prelude.hlsl");

/// The HLSL name of the entry function that replaces `shader_body`.
pub const ENTRY_POINT: &str = "shader_body";

/// The two shaders of a preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The warp shader, which writes back into the feedback canvas.
    Warp,
    /// The composite shader, which draws the canvas to the screen.
    Comp,
}

impl Kind {
    /// The key prefix of the shader's lines in a preset file.
    pub fn key(self) -> &'static str {
        match self {
            Kind::Warp => "warp",
            Kind::Comp => "comp",
        }
    }

    /// The parameters of the entry function: the documented per-vertex inputs,
    /// then `ret`. The guide lists `uv_orig` only for warp; comp presets use it as
    /// well, so comp gets it too and the host passes the unwarped `uv` there.
    fn params(self) -> &'static str {
        match self {
            Kind::Warp => {
                "float2 uv : TEXCOORD0, float2 uv_orig : TEXCOORD1, float rad : TEXCOORD2, \
                 float ang : TEXCOORD3, out float3 ret : COLOR0"
            }
            Kind::Comp => {
                "float2 uv : TEXCOORD0, float2 uv_orig : TEXCOORD1, float rad : TEXCOORD2, \
                 float ang : TEXCOORD3, float3 hue_shader : TEXCOORD4, out float3 ret : COLOR0"
            }
        }
    }
}

/// The warp and comp shaders of a preset: the `warp_N=` and `comp_N=` lines in
/// order of N, each without its leading backtick, joined with newlines.
pub fn shaders_of(text: &str) -> Vec<(Kind, String)> {
    let mut out = Vec::new();
    for kind in [Kind::Warp, Kind::Comp] {
        let prefix = format!("{}_", kind.key());
        let mut lines: Vec<(u32, &str)> = text
            .lines()
            .filter_map(|line| {
                let (key, value) = line.split_once('=')?;
                let n = key.trim().strip_prefix(&prefix)?.parse().ok()?;
                Some((n, value.strip_prefix('`').unwrap_or(value)))
            })
            .collect();
        if lines.is_empty() {
            continue;
        }
        lines.sort_by_key(|(n, _)| *n);
        let body: Vec<&str> = lines.into_iter().map(|(_, l)| l).collect();
        out.push((kind, body.join("\n")));
    }
    out
}

/// The shader up to its last `}`. Some presets have a signature after the shader
/// body, and MilkDrop ignores it.
pub fn until_last_brace(text: &str) -> &str {
    let mut ignored = Vec::new();
    lex(FileId(0), text, &mut ignored)
        .iter()
        .rev()
        .find(|t| t.is(Punct::RBrace))
        .map_or(text, |t| &text[..t.span.end as usize])
}

/// Turns `shader_body { ... }` into an entry function named [`ENTRY_POINT`] that
/// takes the inputs of `kind` and sets `ret` to 0 before the body. Line numbers
/// stay the same.
pub fn wrap(kind: Kind, shader: &str) -> Result<String, &'static str> {
    let mut ignored = Vec::new();
    let tokens = lex(FileId(0), shader, &mut ignored);
    let body = tokens
        .iter()
        .position(|t| t.kind == TokenKind::Ident && t.text == "shader_body")
        .ok_or("the shader has no shader_body")?;
    let brace = tokens[body + 1..]
        .first()
        .filter(|t| t.is(Punct::LBrace))
        .ok_or("shader_body is not followed by `{`")?;
    let (name_start, name_end) = (
        tokens[body].span.start as usize,
        tokens[body].span.end as usize,
    );
    let brace_end = brace.span.end as usize;
    Ok(format!(
        "{}void {ENTRY_POINT}({}){} ret = 0;{}",
        &shader[..name_start],
        kind.params(),
        &shader[name_end..brace_end],
        &shader[brace_end..],
    ))
}

/// The declarations MilkDrop gives every shader, as (name, HLSL text) pairs for
/// `Options::predeclare`.
pub fn predeclared() -> Vec<(String, String)> {
    let text = include_str!("predeclare.hlsl");
    let parsed = wgslift::syntax::parse("predeclare.hlsl", text, Default::default());
    assert!(
        !parsed.has_errors(),
        "predeclare.hlsl does not parse: {:?}",
        parsed.diagnostics
    );
    parsed
        .unit
        .items
        .iter()
        .map(|item| {
            let (name, span) = match item {
                Item::Variable(v) => {
                    assert_eq!(v.declarators.len(), 1, "one name per declaration");
                    (&v.declarators[0].name.name, v.span)
                }
                Item::Function(f) => (&f.name.name, f.span),
                _ => panic!("predeclare.hlsl holds only variables and functions"),
            };
            let mut decl = text[span.start as usize..span.end as usize].to_string();
            if matches!(item, Item::Variable(_)) && !decl.ends_with(';') {
                decl.push(';');
            }
            (name.clone(), decl)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_lines_join_in_key_order_without_backticks() {
        let preset = "[preset00]\nwarp_2=`  ret = 1;\nwarp_1=`shader_body {\n\
                      comp_1=shader_body { ret = 0; }\nwarp_3=`}\nzoom=1.0\n";
        let shaders = shaders_of(preset);
        assert_eq!(
            shaders,
            vec![
                (Kind::Warp, "shader_body {\n  ret = 1;\n}".to_string()),
                (Kind::Comp, "shader_body { ret = 0; }".to_string()),
            ]
        );
    }

    #[test]
    fn a_preset_without_shader_lines_has_no_shaders() {
        assert!(shaders_of("[preset00]\nzoom=1.0\nper_frame_1=a=1;\n").is_empty());
    }

    #[test]
    fn text_after_the_last_brace_is_cut() {
        let shader = "shader_body { ret = 0; } // } in a comment\nsigned by someone";
        assert_eq!(until_last_brace(shader), "shader_body { ret = 0; }");
    }

    #[test]
    fn text_without_a_brace_stays_whole() {
        assert_eq!(until_last_brace("no body"), "no body");
    }

    #[test]
    fn warp_body_becomes_an_entry_with_warp_inputs() {
        let wrapped = wrap(
            Kind::Warp,
            "sampler sampler_x;\nshader_body\n{\n ret *= 2;\n}",
        )
        .unwrap();
        assert_eq!(
            wrapped,
            "sampler sampler_x;\nvoid shader_body(float2 uv : TEXCOORD0, float2 uv_orig : TEXCOORD1, \
             float rad : TEXCOORD2, float ang : TEXCOORD3, out float3 ret : COLOR0)\n{ ret = 0;\n \
             ret *= 2;\n}"
        );
    }

    #[test]
    fn comp_entry_also_takes_hue_shader() {
        let wrapped = wrap(Kind::Comp, "shader_body { ret = hue_shader; }").unwrap();
        assert!(wrapped.contains("float3 hue_shader : TEXCOORD4, out float3 ret : COLOR0)"));
        assert!(wrapped.ends_with("{ ret = 0; ret = hue_shader; }"));
    }

    #[test]
    fn shader_body_in_a_comment_is_not_the_body() {
        let wrapped = wrap(Kind::Warp, "// shader_body here\nshader_body { }").unwrap();
        assert!(wrapped.starts_with("// shader_body here\nvoid shader_body("));
    }

    #[test]
    fn a_shader_without_shader_body_is_an_error() {
        assert!(wrap(Kind::Warp, "float4 main() : COLOR { return 0; }").is_err());
        assert!(wrap(Kind::Warp, "shader_body ret = 0;").is_err());
    }

    #[test]
    fn prelude_and_predeclared_parse_with_one_name_each() {
        let parsed = wgslift::syntax::parse("<prelude test>", PRELUDE, Default::default());
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let decls = predeclared();
        let names: std::collections::HashSet<&str> =
            decls.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), decls.len(), "a name is declared twice");
        for name in [
            "time",
            "_qh",
            "rot_rand4",
            "sampler_pc_main",
            "texsize_noisevol_hq",
        ] {
            assert!(names.contains(name), "{name} is missing");
        }
        for (name, decl) in &decls {
            let parsed = wgslift::syntax::parse(name, decl, Default::default());
            assert!(!parsed.has_errors(), "{name}: {:?}", parsed.diagnostics);
        }
        let blur = &decls.iter().find(|(n, _)| n == "GetBlur2").unwrap().1;
        assert!(blur.starts_with("float3 GetBlur2(float2 uv)") && blur.ends_with('}'));
        let time = &decls.iter().find(|(n, _)| n == "time").unwrap().1;
        assert_eq!(time, "uniform float time;");
    }

    #[test]
    fn q_macros_name_the_components_of_the_q_vectors() {
        let lines: Vec<&str> = PRELUDE.lines().map(str::trim_end).collect();
        assert!(lines.contains(&"#define q1 _qa.x"));
        assert!(lines.contains(&"#define q32 _qh.w"));
    }
}
