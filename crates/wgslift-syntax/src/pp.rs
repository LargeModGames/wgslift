//! The C-style preprocessor: macros, conditional blocks and includes.
//!
//! Macro expansion uses hide sets: a token that came out of macro `M` is never
//! expanded as `M` again, so recursive macros stop.

use std::collections::{HashMap, VecDeque};

use crate::diagnostic::Diagnostic;
use crate::lexer::{Punct, Token, TokenKind, lex};
use crate::span::{FileId, SourceFiles, Span};

/// The most tokens one preprocessor run can produce.
pub const MAX_TOKENS: usize = 1 << 20;
/// The deepest nesting of `#include` files.
pub const MAX_INCLUDE_DEPTH: usize = 32;
/// The deepest nesting of macro arguments that are expanded inside other arguments.
const MAX_EXPANSION_DEPTH: usize = 200;

/// Finds the text of a file that `#include` names.
pub trait IncludeResolver {
    /// Returns the name and text of the file. `angled` is true for `<name>` and
    /// false for `"name"`. `includer` is the name of the file with the directive.
    fn resolve(
        &mut self,
        path: &str,
        angled: bool,
        includer: &str,
    ) -> Result<(String, String), String>;
}

/// A resolver that refuses every `#include`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoIncludes;

impl IncludeResolver for NoIncludes {
    fn resolve(&mut self, path: &str, _: bool, _: &str) -> Result<(String, String), String> {
        Err(format!("cannot open `{path}`: includes are not available"))
    }
}

#[derive(Debug, Clone)]
struct Macro {
    params: Option<Vec<String>>,
    body: Vec<Token>,
}

#[derive(Debug, Clone)]
struct PpToken {
    token: Token,
    /// Ids of the macros this token must not expand, sorted.
    hide: Vec<u32>,
}

impl PpToken {
    fn new(token: Token) -> PpToken {
        PpToken {
            token,
            hide: Vec::new(),
        }
    }
}

struct Cond {
    /// The branch that is open now is kept.
    active: bool,
    /// One branch of this block was kept.
    taken: bool,
    /// The block is inside a kept branch.
    parent_active: bool,
    seen_else: bool,
    span: Span,
}

/// Runs the preprocessor on `main` and returns the tokens for the parser.
pub fn preprocess(
    files: &mut SourceFiles,
    main: FileId,
    defines: &[(String, String)],
    includes: &mut dyn IncludeResolver,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<Token> {
    let mut pp = Preprocessor {
        files,
        includes,
        macros: HashMap::new(),
        ids: HashMap::new(),
        out: Vec::new(),
        diagnostics,
        include_depth: 0,
        expansion_depth: 0,
        produced: 0,
        over_budget: false,
    };
    for (name, value) in defines {
        let text = format!("#define {name} {value}\n");
        let file = pp.files.add(format!("<define {name}>"), text);
        pp.run_file(file);
    }
    pp.run_file(main);
    pp.out
}

struct Preprocessor<'a> {
    files: &'a mut SourceFiles,
    includes: &'a mut dyn IncludeResolver,
    macros: HashMap<String, Macro>,
    ids: HashMap<String, u32>,
    out: Vec<Token>,
    diagnostics: &'a mut Vec<Diagnostic>,
    include_depth: usize,
    expansion_depth: usize,
    produced: usize,
    over_budget: bool,
}

impl Preprocessor<'_> {
    fn error(&mut self, span: Span, message: impl Into<String>) {
        self.diagnostics.push(Diagnostic::error(span, message));
    }

    fn run_file(&mut self, file: FileId) {
        let text = self
            .files
            .get(file)
            .map(|f| f.text.clone())
            .unwrap_or_default();
        let tokens = lex(file, &text, self.diagnostics);
        let mut conds: Vec<Cond> = Vec::new();
        let mut pending = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            let mut end = i + 1;
            while end < tokens.len() && !tokens[end].line_start {
                end += 1;
            }
            let line = &tokens[i..end];
            if line[0].is(Punct::Hash) {
                let pending = std::mem::take(&mut pending);
                self.emit(pending);
                self.directive(line, &mut conds);
            } else if conds.last().is_none_or(|c| c.active) {
                pending.extend(line.iter().cloned().map(PpToken::new));
            }
            i = end;
        }
        self.emit(pending);
        for cond in conds {
            self.error(cond.span, "this `#if` has no `#endif`");
        }
    }

    fn emit(&mut self, tokens: Vec<PpToken>) {
        if tokens.is_empty() {
            return;
        }
        let expanded = self.expand(tokens);
        self.out.extend(expanded.into_iter().map(|t| t.token));
    }

    fn directive(&mut self, line: &[Token], conds: &mut Vec<Cond>) {
        let Some(name_token) = line.get(1) else {
            return;
        };
        let span = line[0].span.to(line[line.len() - 1].span);
        let name = name_token.text.as_str();
        let rest = &line[2.min(line.len())..];
        let active = conds.last().is_none_or(|c| c.active);
        match name {
            "if" | "ifdef" | "ifndef" => {
                let keep = active
                    && match name {
                        "if" => self.condition(rest, span),
                        _ => {
                            let defined = self.defined_name(rest, span);
                            defined == (name == "ifdef")
                        }
                    };
                conds.push(Cond {
                    active: keep,
                    taken: keep,
                    parent_active: active,
                    seen_else: false,
                    span,
                });
            }
            "elif" => match conds.last_mut() {
                Some(c) if !c.seen_else => {
                    let (parent, taken) = (c.parent_active, c.taken);
                    let keep = parent && !taken && self.condition(rest, span);
                    let c = conds.last_mut().expect("checked above");
                    c.active = keep;
                    c.taken |= keep;
                }
                Some(_) => self.error(span, "`#elif` after `#else`"),
                None => self.error(span, "`#elif` without `#if`"),
            },
            "else" => match conds.last_mut() {
                Some(c) if !c.seen_else => {
                    c.active = c.parent_active && !c.taken;
                    c.taken = true;
                    c.seen_else = true;
                }
                Some(_) => self.error(span, "a second `#else` in one `#if`"),
                None => self.error(span, "`#else` without `#if`"),
            },
            "endif" => {
                if conds.pop().is_none() {
                    self.error(span, "`#endif` without `#if`");
                }
            }
            _ if !active => {}
            "define" => self.define(rest, span),
            "undef" => match rest.first() {
                Some(t) if t.kind == TokenKind::Ident => {
                    self.macros.remove(&t.text);
                }
                _ => self.error(span, "`#undef` needs a macro name"),
            },
            "include" => self.include(rest, span),
            "error" => {
                let text: Vec<&str> = rest.iter().map(|t| t.text.as_str()).collect();
                self.error(span, format!("#error {}", text.join(" ")));
            }
            "line" | "pragma" => {}
            _ => self.error(name_token.span, format!("unknown directive `#{name}`")),
        }
    }

    fn define(&mut self, rest: &[Token], span: Span) {
        let Some(name) = rest.first().filter(|t| t.kind == TokenKind::Ident) else {
            self.error(span, "`#define` needs a macro name");
            return;
        };
        let mut body_start = 1;
        let mut params = None;
        if rest
            .get(1)
            .is_some_and(|t| t.is(Punct::LParen) && !t.space_before)
        {
            let mut list = Vec::new();
            let mut i = 2;
            loop {
                match rest.get(i) {
                    Some(t) if t.is(Punct::RParen) && list.is_empty() => {
                        i += 1;
                        break;
                    }
                    Some(t) if t.kind == TokenKind::Ident => {
                        list.push(t.text.clone());
                        match rest.get(i + 1) {
                            Some(n) if n.is(Punct::Comma) => i += 2,
                            Some(n) if n.is(Punct::RParen) => {
                                i += 2;
                                break;
                            }
                            _ => {
                                self.error(span, "a bad parameter list in `#define`");
                                return;
                            }
                        }
                    }
                    _ => {
                        self.error(
                            span,
                            "a bad parameter list in `#define` (variadic macros are not supported)",
                        );
                        return;
                    }
                }
            }
            params = Some(list);
            body_start = i;
        }
        let mut body: Vec<Token> = rest[body_start.min(rest.len())..].to_vec();
        if let Some(first) = body.first_mut() {
            first.space_before = false;
        }
        if body.first().is_some_and(|t| t.is(Punct::HashHash))
            || body.last().is_some_and(|t| t.is(Punct::HashHash))
        {
            self.error(span, "`##` cannot be at the start or end of a macro");
            return;
        }
        let next_id = self.ids.len() as u32;
        self.ids.entry(name.text.clone()).or_insert(next_id);
        self.macros
            .insert(name.text.clone(), Macro { params, body });
    }

    fn include(&mut self, rest: &[Token], span: Span) {
        let (path, angled) = match rest.first() {
            Some(t) if t.kind == TokenKind::Str && t.text.len() >= 2 => {
                (t.text[1..t.text.len() - 1].to_string(), false)
            }
            Some(t) if t.is(Punct::Lt) => {
                let Some(close) = rest.iter().position(|t| t.is(Punct::Gt)) else {
                    self.error(span, "`#include <` has no closing `>`");
                    return;
                };
                (
                    rest[1..close].iter().map(|t| t.text.as_str()).collect(),
                    true,
                )
            }
            _ => {
                self.error(span, "`#include` needs a file name in quotes or `<>`");
                return;
            }
        };
        if self.include_depth >= MAX_INCLUDE_DEPTH {
            self.error(
                span,
                format!("includes are nested more than {MAX_INCLUDE_DEPTH} deep"),
            );
            return;
        }
        let includer = self
            .files
            .get(span.file)
            .map(|f| f.name.clone())
            .unwrap_or_default();
        match self.includes.resolve(&path, angled, &includer) {
            Ok((name, text)) => {
                let file = self.files.add(name, text);
                self.include_depth += 1;
                self.run_file(file);
                self.include_depth -= 1;
            }
            Err(message) => self.error(span, message),
        }
    }

    /// The name after `#ifdef` or `#ifndef` is a macro.
    fn defined_name(&mut self, rest: &[Token], span: Span) -> bool {
        match rest.first() {
            Some(t) if t.kind == TokenKind::Ident => self.macros.contains_key(&t.text),
            _ => {
                self.error(span, "this directive needs a macro name");
                false
            }
        }
    }

    /// Evaluates the expression of an `#if` or `#elif`.
    fn condition(&mut self, rest: &[Token], span: Span) -> bool {
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < rest.len() {
            let t = &rest[i];
            if t.is_word("defined") {
                let (name, used) = match (rest.get(i + 1), rest.get(i + 2), rest.get(i + 3)) {
                    (Some(open), Some(n), Some(close))
                        if open.is(Punct::LParen)
                            && n.kind == TokenKind::Ident
                            && close.is(Punct::RParen) =>
                    {
                        (Some(n), 4)
                    }
                    (Some(n), _, _) if n.kind == TokenKind::Ident => (Some(n), 2),
                    _ => (None, 1),
                };
                let Some(name) = name else {
                    self.error(t.span, "`defined` needs a macro name");
                    return false;
                };
                let value = if self.macros.contains_key(&name.text) {
                    "1"
                } else {
                    "0"
                };
                tokens.push(PpToken::new(Token {
                    kind: TokenKind::Int,
                    text: value.to_string(),
                    ..t.clone()
                }));
                i += used;
            } else {
                tokens.push(PpToken::new(t.clone()));
                i += 1;
            }
        }
        let tokens: Vec<Token> = self.expand(tokens).into_iter().map(|t| t.token).collect();
        if tokens.is_empty() {
            self.error(span, "`#if` needs an expression");
            return false;
        }
        let mut eval = Eval {
            tokens: &tokens,
            pos: 0,
            error: None,
        };
        let value = eval.ternary();
        if eval.error.is_none() && eval.pos < tokens.len() {
            eval.error = Some((
                tokens[eval.pos].site(),
                "unexpected token in `#if`".to_string(),
            ));
        }
        match eval.error {
            Some((at, message)) => {
                self.error(at, message);
                false
            }
            None => value != 0,
        }
    }

    fn expand(&mut self, input: Vec<PpToken>) -> Vec<PpToken> {
        let mut input: VecDeque<PpToken> = input.into();
        let mut out = Vec::new();
        while let Some(t) = input.pop_front() {
            let name = &t.token.text;
            let id = match (t.token.kind, self.ids.get(name)) {
                (TokenKind::Ident, Some(id))
                    if self.macros.contains_key(name) && t.hide.binary_search(id).is_err() =>
                {
                    *id
                }
                _ => {
                    out.push(t);
                    continue;
                }
            };
            let mac = self.macros[name].clone();
            let site = t.token.site();
            let (args, hide) = match &mac.params {
                None => (Vec::new(), insert(&t.hide, id)),
                Some(params) => {
                    if !input.front().is_some_and(|n| n.token.is(Punct::LParen)) {
                        out.push(t);
                        continue;
                    }
                    let Some((mut args, close)) = collect_args(&mut input) else {
                        self.error(
                            site,
                            format!("the call of macro `{name}` has no closing `)`"),
                        );
                        continue;
                    };
                    if params.is_empty() && args.len() == 1 && args[0].is_empty() {
                        args.clear();
                    }
                    if args.len() != params.len() {
                        self.error(
                            site,
                            format!(
                                "macro `{name}` takes {} arguments, not {}",
                                params.len(),
                                args.len()
                            ),
                        );
                        continue;
                    }
                    (args, insert(&intersect(&t.hide, &close.hide), id))
                }
            };
            let body = self.substitute(&mac, &args, &hide, site);
            self.produced += body.len();
            if self.produced > MAX_TOKENS {
                if !self.over_budget {
                    self.over_budget = true;
                    self.error(
                        site,
                        format!("macro expansion makes more than {MAX_TOKENS} tokens"),
                    );
                }
                return out;
            }
            for token in body.into_iter().rev() {
                input.push_front(token);
            }
        }
        out
    }

    /// The body of `mac` with its parameters replaced, `#` and `##` applied.
    fn substitute(
        &mut self,
        mac: &Macro,
        args: &[Vec<PpToken>],
        hide: &[u32],
        site: Span,
    ) -> Vec<PpToken> {
        let params = mac.params.as_deref().unwrap_or(&[]);
        let param = |t: &Token| {
            if t.kind == TokenKind::Ident {
                params.iter().position(|p| *p == t.text)
            } else {
                None
            }
        };
        let body = &mac.body;
        let mut out: Vec<PpToken> = Vec::new();
        // The left operand of the next `##` is an empty argument.
        let mut empty_left = false;
        let mut i = 0;
        while i < body.len() {
            let t = &body[i];
            let pasted = body.get(i + 1).is_some_and(|n| n.is(Punct::HashHash));
            let after_paste = i > 0 && body[i - 1].is(Punct::HashHash);
            if t.is(Punct::HashHash) {
                let Some(rhs) = body.get(i + 1) else { break };
                let rhs: Vec<PpToken> = match param(rhs) {
                    Some(p) => args[p].clone(),
                    None => vec![PpToken::new(rhs.clone())],
                };
                let empty_right = rhs.is_empty();
                if empty_left {
                    out.extend(rhs);
                } else {
                    self.paste(&mut out, rhs, site);
                }
                empty_left &= empty_right;
                i += 2;
                continue;
            }
            empty_left = false;
            if mac.params.is_some()
                && t.is(Punct::Hash)
                && let Some(p) = body.get(i + 1).and_then(param)
            {
                out.push(PpToken::new(stringify(&args[p], t)));
                i += 2;
                continue;
            }
            match param(t) {
                Some(p) if pasted || after_paste => {
                    empty_left = args[p].is_empty();
                    out.extend(args[p].iter().cloned());
                }
                Some(p) => {
                    let mut arg = args[p].clone();
                    if self.expansion_depth < MAX_EXPANSION_DEPTH {
                        self.expansion_depth += 1;
                        arg = self.expand(arg);
                        self.expansion_depth -= 1;
                    } else {
                        self.error(site, "macro arguments are nested too deep");
                    }
                    if let Some(first) = arg.first_mut() {
                        first.token.space_before = t.space_before;
                    }
                    out.extend(arg);
                }
                None => {
                    let mut token = t.clone();
                    token.expanded_from = Some(site);
                    out.push(PpToken::new(token));
                }
            }
            i += 1;
        }
        for t in &mut out {
            t.hide = union(&t.hide, hide);
        }
        out
    }

    /// Joins the last token of `out` with the first of `rhs`, then appends the rest.
    /// An empty `rhs` leaves `out` as it is.
    fn paste(&mut self, out: &mut Vec<PpToken>, rhs: Vec<PpToken>, site: Span) {
        let mut rhs = rhs.into_iter();
        let Some(right) = rhs.next() else {
            return;
        };
        let Some(left) = out.pop() else {
            out.push(right);
            out.extend(rhs);
            return;
        };
        let text = format!("{}{}", left.token.text, right.token.text);
        let mut ignored = Vec::new();
        let relexed = lex(site.file, &text, &mut ignored);
        match relexed.as_slice() {
            [single] if ignored.is_empty() => {
                let mut token = left.token.clone();
                token.kind = single.kind;
                token.text = text;
                token.expanded_from = Some(site);
                out.push(PpToken {
                    token,
                    hide: left.hide,
                });
            }
            _ => {
                self.error(
                    site,
                    format!(
                        "pasting `{}` and `{}` does not make one token",
                        left.token.text, right.token.text
                    ),
                );
                out.push(left);
                out.push(right);
            }
        }
        out.extend(rhs);
    }
}

/// Takes `( args )` from the front of `input`. Returns the arguments and the `)`.
fn collect_args(input: &mut VecDeque<PpToken>) -> Option<(Vec<Vec<PpToken>>, PpToken)> {
    input.pop_front();
    let mut args = vec![Vec::new()];
    let mut depth = 1;
    while let Some(t) = input.pop_front() {
        if t.token.is(Punct::LParen) {
            depth += 1;
        } else if t.token.is(Punct::RParen) {
            depth -= 1;
            if depth == 0 {
                return Some((args, t));
            }
        } else if t.token.is(Punct::Comma) && depth == 1 {
            args.push(Vec::new());
            continue;
        }
        args.last_mut().expect("never empty").push(t);
    }
    None
}

fn stringify(arg: &[PpToken], hash: &Token) -> Token {
    let mut text = String::from("\"");
    for (i, t) in arg.iter().enumerate() {
        if i > 0 && t.token.space_before {
            text.push(' ');
        }
        if t.token.kind == TokenKind::Str {
            for c in t.token.text.chars() {
                if c == '"' || c == '\\' {
                    text.push('\\');
                }
                text.push(c);
            }
        } else {
            text.push_str(&t.token.text);
        }
    }
    text.push('"');
    Token {
        kind: TokenKind::Str,
        text,
        ..hash.clone()
    }
}

fn insert(set: &[u32], id: u32) -> Vec<u32> {
    union(set, &[id])
}

fn union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out: Vec<u32> = a.iter().chain(b).copied().collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn intersect(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter()
        .copied()
        .filter(|x| b.binary_search(x).is_ok())
        .collect()
}

/// Evaluates an integer expression for `#if`. Names that are left are 0.
struct Eval<'a> {
    tokens: &'a [Token],
    pos: usize,
    error: Option<(Span, String)>,
}

impl Eval<'_> {
    fn fail(&mut self, message: &str) -> i64 {
        if self.error.is_none() {
            let at = self
                .tokens
                .get(self.pos)
                .or(self.tokens.last())
                .map(Token::site);
            if let Some(at) = at {
                self.error = Some((at, message.to_string()));
            }
        }
        self.pos = self.tokens.len();
        0
    }

    fn eat(&mut self, punct: Punct) -> bool {
        let found = self.tokens.get(self.pos).is_some_and(|t| t.is(punct));
        self.pos += usize::from(found);
        found
    }

    fn ternary(&mut self) -> i64 {
        let cond = self.binary(0);
        if !self.eat(Punct::Question) {
            return cond;
        }
        let a = self.ternary();
        if !self.eat(Punct::Colon) {
            return self.fail("expected `:` in `#if`");
        }
        let b = self.ternary();
        if cond != 0 { a } else { b }
    }

    fn binary(&mut self, min: u8) -> i64 {
        let mut lhs = self.unary();
        loop {
            let Some(TokenKind::Punct(op)) = self.tokens.get(self.pos).map(|t| t.kind) else {
                return lhs;
            };
            let Some(level) = binary_level(op) else {
                return lhs;
            };
            if level < min {
                return lhs;
            }
            self.pos += 1;
            let rhs = self.binary(level + 1);
            lhs = match op {
                Punct::OrOr => i64::from(lhs != 0 || rhs != 0),
                Punct::AndAnd => i64::from(lhs != 0 && rhs != 0),
                Punct::Pipe => lhs | rhs,
                Punct::Caret => lhs ^ rhs,
                Punct::Amp => lhs & rhs,
                Punct::EqEq => i64::from(lhs == rhs),
                Punct::NotEq => i64::from(lhs != rhs),
                Punct::Lt => i64::from(lhs < rhs),
                Punct::Gt => i64::from(lhs > rhs),
                Punct::LtEq => i64::from(lhs <= rhs),
                Punct::GtEq => i64::from(lhs >= rhs),
                Punct::Shl => lhs.wrapping_shl(rhs as u32),
                Punct::Shr => lhs.wrapping_shr(rhs as u32),
                Punct::Plus => lhs.wrapping_add(rhs),
                Punct::Minus => lhs.wrapping_sub(rhs),
                Punct::Star => lhs.wrapping_mul(rhs),
                Punct::Slash | Punct::Percent if rhs == 0 => {
                    return self.fail("division by zero in `#if`");
                }
                Punct::Slash => lhs.wrapping_div(rhs),
                _ => lhs.wrapping_rem(rhs),
            };
        }
    }

    fn unary(&mut self) -> i64 {
        let Some(t) = self.tokens.get(self.pos) else {
            return self.fail("`#if` ends too early");
        };
        self.pos += 1;
        match t.kind {
            TokenKind::Punct(Punct::Minus) => self.unary().wrapping_neg(),
            TokenKind::Punct(Punct::Plus) => self.unary(),
            TokenKind::Punct(Punct::Bang) => i64::from(self.unary() == 0),
            TokenKind::Punct(Punct::Tilde) => !self.unary(),
            TokenKind::Punct(Punct::LParen) => {
                let v = self.ternary();
                if self.eat(Punct::RParen) {
                    v
                } else {
                    self.fail("expected `)` in `#if`")
                }
            }
            TokenKind::Int => match parse_int(&t.text) {
                Some(v) => v,
                None => {
                    self.pos -= 1;
                    self.fail("a bad integer in `#if`")
                }
            },
            TokenKind::Ident => 0,
            _ => {
                self.pos -= 1;
                self.fail("unexpected token in `#if`")
            }
        }
    }
}

fn binary_level(op: Punct) -> Option<u8> {
    Some(match op {
        Punct::OrOr => 0,
        Punct::AndAnd => 1,
        Punct::Pipe => 2,
        Punct::Caret => 3,
        Punct::Amp => 4,
        Punct::EqEq | Punct::NotEq => 5,
        Punct::Lt | Punct::Gt | Punct::LtEq | Punct::GtEq => 6,
        Punct::Shl | Punct::Shr => 7,
        Punct::Plus | Punct::Minus => 8,
        Punct::Star | Punct::Slash | Punct::Percent => 9,
        _ => return None,
    })
}

/// Parses a decimal, hex or octal integer with an optional `u` or `l` suffix.
pub(crate) fn parse_int(text: &str) -> Option<i64> {
    let digits = text.trim_end_matches(['u', 'U', 'l', 'L']);
    let value = if let Some(hex) = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()?
    } else if digits.len() > 1 && digits.starts_with('0') {
        u64::from_str_radix(&digits[1..], 8).ok()?
    } else {
        digits.parse::<u64>().ok()?
    };
    Some(value as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_with(text: &str, includes: &mut dyn IncludeResolver) -> (String, Vec<Diagnostic>) {
        let mut files = SourceFiles::default();
        let main = files.add("main", text);
        let mut d = Vec::new();
        let tokens = preprocess(&mut files, main, &[], includes, &mut d);
        let text: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        (text.join(" "), d)
    }

    fn run(text: &str) -> String {
        let (out, d) = run_with(text, &mut NoIncludes);
        assert!(d.is_empty(), "{d:?}");
        out
    }

    fn errors(text: &str) -> Vec<String> {
        run_with(text, &mut NoIncludes)
            .1
            .into_iter()
            .map(|d| d.message)
            .collect()
    }

    #[test]
    fn object_like_macros_expand_and_rescan() {
        assert_eq!(
            run("#define sat saturate\n#define X sat(Y)\n#define Y 2\nX;"),
            "saturate ( 2 ) ;"
        );
    }

    #[test]
    fn a_recursive_macro_stops_at_its_own_name() {
        assert_eq!(run("#define a a + b\n#define b a\na"), "a + a");
    }

    #[test]
    fn function_like_macros_take_arguments_over_lines() {
        let out = run("#define MAD(a, b, c) ((a) * (b) + (c))\nMAD(x,\n f(1, 2), 3)");
        assert_eq!(out, "( ( x ) * ( f ( 1 , 2 ) ) + ( 3 ) )");
    }

    #[test]
    fn a_function_like_macro_without_parentheses_is_a_name() {
        assert_eq!(run("#define F(x) x\nF + F(1)"), "F + 1");
    }

    #[test]
    fn a_space_before_the_parenthesis_makes_an_object_like_macro() {
        assert_eq!(run("#define F (x)\nF(1)"), "( x ) ( 1 )");
    }

    #[test]
    fn arguments_are_expanded_before_substitution() {
        assert_eq!(run("#define ONE 1\n#define ID(x) x\nID(ONE)"), "1");
    }

    #[test]
    fn stringify_and_paste() {
        assert_eq!(run("#define S(x) #x\nS(a  +\"q\")"), r#""a +\"q\"""#);
        assert_eq!(
            run("#define CAT(a, b) a ## b\nCAT(float, 4) CAT(x, ) CAT(, y)"),
            "float4 x y"
        );
        assert_eq!(
            run("#define ONE 1\n#define CAT(a, b) a ## b\nCAT(ONE, 2)"),
            "ONE2"
        );
    }

    #[test]
    fn a_bad_paste_is_an_error() {
        assert_eq!(errors("#define CAT(a, b) a ## b\nCAT(+, /)").len(), 1);
    }

    #[test]
    fn conditional_blocks() {
        let text = "#if 0\na\n#elif 1 + 1 == 2\nb\n#else\nc\n#endif\n#ifdef B\nd\n#endif\n#define B\n#ifndef B\ne\n#else\nf\n#endif";
        assert_eq!(run(text), "b f");
    }

    #[test]
    fn nested_conditionals_in_a_skipped_block_stay_skipped() {
        assert_eq!(
            run("#if 0\n#if 1\na\n#else\nb\n#endif\n#elif 1\nc\n#endif"),
            "c"
        );
    }

    #[test]
    fn defined_and_operators_in_if() {
        let text = "#define A 3\n#if defined(A) && !defined B && (A << 2) == 12 && -1 < 0 && 1 ? 0x10 % 7 == 2 : 0\nyes\n#endif";
        assert_eq!(run(text), "yes");
        assert_eq!(
            run("#if UNKNOWN\nno\n#endif\n#if 010 == 8\nyes\n#endif"),
            "yes"
        );
    }

    #[test]
    fn skipped_blocks_ignore_unknown_directives_and_bad_code() {
        assert_eq!(run("#if 0\n#bogus\n#define F(\n#endif\nok"), "ok");
    }

    #[test]
    fn directive_errors_are_reported() {
        assert_eq!(errors("#if 1\na").len(), 1);
        assert_eq!(errors("#endif").len(), 1);
        assert_eq!(errors("#if 1 / 0\n#endif").len(), 1);
        assert_eq!(errors("#if\n#endif").len(), 1);
        assert_eq!(errors("#bogus").len(), 1);
        assert_eq!(errors("#error stop here"), ["#error stop here"]);
        assert_eq!(errors("#if 1\n#else\n#else\n#endif").len(), 1);
        assert_eq!(errors("#define F(a) a\nF(1, 2)").len(), 1);
        assert_eq!(errors("#define F(a) a\nF(1").len(), 1);
        assert_eq!(errors("#define F(a, ...) a").len(), 1);
    }

    #[test]
    fn line_and_pragma_are_ignored_and_undef_removes() {
        assert_eq!(
            run("#pragma warning(disable: 1)\n#line 10\n#define A 1\n#undef A\nA"),
            "A"
        );
    }

    #[test]
    fn tokens_from_a_macro_point_at_the_call() {
        let mut files = SourceFiles::default();
        let main = files.add("main", "#define X a + b\n  X");
        let mut d = Vec::new();
        let tokens = preprocess(&mut files, main, &[], &mut NoIncludes, &mut d);
        assert!(tokens.iter().all(|t| t.site() == Span::new(main, 18, 19)));
        assert_eq!(files.text(tokens[0].span), "a");
    }

    #[test]
    fn exponential_macros_stop_at_the_token_limit() {
        let mut text = String::from("#define m0 x x\n");
        for i in 1..30 {
            text.push_str(&format!("#define m{i} m{} m{}\n", i - 1, i - 1));
        }
        text.push_str("m29");
        let (_, d) = run_with(&text, &mut NoIncludes);
        assert!(d.iter().any(|d| d.message.contains("more than")));
    }

    #[test]
    fn deep_argument_nesting_does_not_overflow_the_stack() {
        let depth = 5_000;
        let text = format!(
            "#define F(x) x\n{}1{}",
            "F(".repeat(depth),
            ")".repeat(depth)
        );
        let (_, d) = run_with(&text, &mut NoIncludes);
        assert!(d.iter().any(|d| d.message.contains("nested too deep")));
    }

    struct Files;

    impl IncludeResolver for Files {
        fn resolve(
            &mut self,
            path: &str,
            angled: bool,
            includer: &str,
        ) -> Result<(String, String), String> {
            match (path, angled) {
                ("common.h", false) => Ok((path.into(), format!("#define FROM \"{includer}\"\n"))),
                ("loop.h", true) => Ok((path.into(), "#include <loop.h>\n".into())),
                _ => Err(format!("no file {path}")),
            }
        }
    }

    #[test]
    fn includes_go_through_the_resolver() {
        let (out, d) = run_with("#include \"common.h\"\nFROM", &mut Files);
        assert!(d.is_empty(), "{d:?}");
        assert_eq!(out, "\"main\"");
        let (_, d) = run_with("#include <loop.h>", &mut Files);
        assert!(d.iter().any(|d| d.message.contains("nested more than")));
        assert_eq!(errors("#include \"x.h\"").len(), 1);
    }

    #[test]
    fn predefined_macros() {
        let mut files = SourceFiles::default();
        let main = files.add("main", "LEVEL");
        let mut d = Vec::new();
        let defines = [("LEVEL".to_string(), "3".to_string())];
        let tokens = preprocess(&mut files, main, &defines, &mut NoIncludes, &mut d);
        assert_eq!(tokens[0].text, "3");
    }

    #[test]
    fn integers_in_if() {
        assert_eq!(parse_int("0x1fU"), Some(31));
        assert_eq!(parse_int("017"), Some(15));
        assert_eq!(parse_int("0"), Some(0));
        assert_eq!(parse_int("9z"), None);
    }
}
