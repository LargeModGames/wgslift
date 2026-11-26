//! The C-style preprocessor: macros, conditional blocks and includes.
//!
//! Macro expansion uses hide sets: a token that came out of macro `M` is never
//! expanded as `M` again, so recursive macros stop.

use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::diagnostic::Diagnostic;
use crate::lexer::{Punct, Token, TokenKind, lex};
use crate::span::{FileId, SourceFiles, Span};

/// The most macro expansion work one preprocessor run may do. One unit is one
/// expansion, one copied argument token, one stringified or pasted byte, or one
/// hide-set entry. This bounds the time and memory of expansion.
pub const MAX_EXPANSION_WORK: usize = 1 << 22;
/// The deepest nesting of `#include` files.
pub const MAX_INCLUDE_DEPTH: usize = 32;
/// The most files one run may include, counting a file each time it is included.
pub const MAX_INCLUDES: usize = 4096;
/// The deepest nesting of macro arguments that are expanded inside other arguments.
const MAX_EXPANSION_DEPTH: usize = 64;

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

#[derive(Debug)]
struct Macro {
    /// The number of parameters of a function-like macro; `None` for object-like.
    params: Option<usize>,
    body: Vec<Token>,
    /// For each body token, the index of the parameter it names.
    body_params: Vec<Option<usize>>,
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
        included: 0,
        expansion_depth: 0,
        work: 0,
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
    macros: HashMap<String, Rc<Macro>>,
    ids: HashMap<String, u32>,
    out: Vec<Token>,
    diagnostics: &'a mut Vec<Diagnostic>,
    include_depth: usize,
    /// Files included so far, counting repeats.
    included: usize,
    expansion_depth: usize,
    work: usize,
    over_budget: bool,
}

impl Preprocessor<'_> {
    fn error(&mut self, span: Span, message: impl Into<String>) {
        self.diagnostics.push(Diagnostic::error(span, message));
    }

    /// Charges `units` of expansion work. Returns false when the budget is used up;
    /// the first time, it reports an error.
    fn spend(&mut self, units: usize, site: Span) -> bool {
        if self.over_budget {
            return false;
        }
        self.work = self.work.saturating_add(units);
        if self.work > MAX_EXPANSION_WORK {
            self.over_budget = true;
            self.error(
                site,
                format!("macro expansion needs more than {MAX_EXPANSION_WORK} steps"),
            );
            return false;
        }
        true
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
                // The message as written, from the first to the last token of the line.
                let text = match (rest.first(), rest.last()) {
                    (Some(a), Some(b)) => self.files.text(a.span.to(b.span)).to_string(),
                    _ => String::new(),
                };
                self.error(span, format!("#error {text}").trim_end().to_string());
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
        let mut index = HashMap::new();
        for (i, p) in params.iter().flatten().enumerate() {
            if index.insert(p.as_str(), i).is_some() {
                self.error(span, format!("the parameter `{p}` is named twice"));
                return;
            }
        }
        let body_params = body
            .iter()
            .map(|t| {
                (t.kind == TokenKind::Ident)
                    .then(|| index.get(t.text.as_str()).copied())
                    .flatten()
            })
            .collect();
        let next_id = self.ids.len() as u32;
        self.ids.entry(name.text.clone()).or_insert(next_id);
        let mac = Macro {
            params: params.map(|p| p.len()),
            body,
            body_params,
        };
        self.macros.insert(name.text.clone(), Rc::new(mac));
    }

    fn include(&mut self, rest: &[Token], span: Span) {
        let (path, angled) = match rest.first() {
            Some(t) if t.kind == TokenKind::Str => {
                match t.text.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                    Some(path) => (path.to_string(), false),
                    None => {
                        self.error(span, "the `#include` file name has no closing quote");
                        return;
                    }
                }
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
        if self.included >= MAX_INCLUDES {
            if self.included == MAX_INCLUDES {
                self.included += 1;
                self.error(span, format!("more than {MAX_INCLUDES} files are included"));
            }
            return;
        }
        self.included += 1;
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
        let mut eval = Eval::new(&tokens);
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
            None => value.is_true(),
        }
    }

    fn expand(&mut self, input: Vec<PpToken>) -> Vec<PpToken> {
        let mut input: VecDeque<PpToken> = input.into();
        let mut out = Vec::new();
        while let Some(t) = input.pop_front() {
            if self.over_budget {
                // The error is reported: the rest passes through unexpanded.
                out.push(t);
                out.extend(input);
                return out;
            }
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
            let mac = Rc::clone(&self.macros[name]);
            let site = t.token.site();
            if !self.spend(1 + mac.body.len(), site) {
                continue;
            }
            let (args, hide) = match mac.params {
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
                    if params == 0 && args.len() == 1 && args[0].is_empty() {
                        args.clear();
                    }
                    if args.len() != params {
                        self.error(
                            site,
                            format!(
                                "macro `{name}` takes {params} arguments, not {}",
                                args.len()
                            ),
                        );
                        continue;
                    }
                    (args, insert(&intersect(&t.hide, &close.hide), id))
                }
            };
            let body = self.substitute(&mac, &args, &hide, site);
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
        let param = |i: usize| mac.body_params.get(i).copied().flatten();
        let body = &mac.body;
        let mut out: Vec<PpToken> = Vec::new();
        // The left operand of the next `##` is an empty argument.
        let mut empty_left = false;
        let mut i = 0;
        while i < body.len() {
            if self.over_budget {
                return out;
            }
            let t = &body[i];
            let pasted = body.get(i + 1).is_some_and(|n| n.is(Punct::HashHash));
            let after_paste = i > 0 && body[i - 1].is(Punct::HashHash);
            if t.is(Punct::HashHash) {
                let Some(rhs) = body.get(i + 1) else { break };
                let rhs: Vec<PpToken> = match param(i + 1) {
                    Some(p) => {
                        self.spend(args[p].len(), site);
                        args[p].clone()
                    }
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
                && let Some(p) = param(i + 1)
            {
                let bytes = args[p].iter().map(|a| a.token.text.len() + 1).sum();
                if self.spend(bytes, site) {
                    out.push(PpToken::new(stringify(&args[p], t)));
                }
                i += 2;
                continue;
            }
            let arg = param(i);
            if let Some(p) = arg
                && !self.spend(args[p].len(), site)
            {
                return out;
            }
            match arg {
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
        let entries = out.iter().map(|t| t.hide.len() + hide.len()).sum();
        if self.spend(entries, site) {
            for t in &mut out {
                t.hide = union(&t.hide, hide);
            }
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
        self.spend(text.len(), site);
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

/// The deepest nesting of an `#if` expression.
const MAX_IF_DEPTH: usize = 64;

/// A value in `#if`: 64 bits, signed or unsigned as in C.
#[derive(Debug, Clone, Copy)]
struct Value {
    bits: u64,
    unsigned: bool,
}

impl Value {
    fn signed(value: i64) -> Value {
        Value {
            bits: value as u64,
            unsigned: false,
        }
    }

    fn truth(value: bool) -> Value {
        Value::signed(i64::from(value))
    }

    fn is_true(self) -> bool {
        self.bits != 0
    }
}

/// Evaluates an integer expression for `#if`. Names that are left are 0.
struct Eval<'a> {
    tokens: &'a [Token],
    pos: usize,
    error: Option<(Span, String)>,
    depth: usize,
    /// Above 0 inside an operand that is not evaluated, such as the right side of
    /// `0 && x`. There a division by zero is not an error.
    skip: usize,
}

impl Eval<'_> {
    fn new(tokens: &[Token]) -> Eval<'_> {
        Eval {
            tokens,
            pos: 0,
            error: None,
            depth: 0,
            skip: 0,
        }
    }

    fn fail(&mut self, message: &str) -> Value {
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
        Value::signed(0)
    }

    fn eat(&mut self, punct: Punct) -> bool {
        let found = self.tokens.get(self.pos).is_some_and(|t| t.is(punct));
        self.pos += usize::from(found);
        found
    }

    fn nested(&mut self, f: impl FnOnce(&mut Self) -> Value) -> Value {
        if self.depth >= MAX_IF_DEPTH {
            return self.fail("the `#if` expression is nested too deep");
        }
        self.depth += 1;
        let value = f(self);
        self.depth -= 1;
        value
    }

    /// Evaluates `f` as an operand that is not used when `unused` is true.
    fn operand(&mut self, unused: bool, f: impl FnOnce(&mut Self) -> Value) -> Value {
        self.skip += usize::from(unused);
        let value = f(self);
        self.skip -= usize::from(unused);
        value
    }

    fn ternary(&mut self) -> Value {
        let cond = self.binary(0);
        if !self.eat(Punct::Question) {
            return cond;
        }
        self.nested(|e| {
            let a = e.operand(!cond.is_true(), Self::ternary);
            if !e.eat(Punct::Colon) {
                return e.fail("expected `:` in `#if`");
            }
            let b = e.operand(cond.is_true(), Self::ternary);
            let chosen = if cond.is_true() { a } else { b };
            Value {
                bits: chosen.bits,
                unsigned: a.unsigned || b.unsigned,
            }
        })
    }

    fn binary(&mut self, min: u8) -> Value {
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
            let unused = match op {
                Punct::AndAnd => !lhs.is_true(),
                Punct::OrOr => lhs.is_true(),
                _ => false,
            };
            let rhs = self.operand(unused, |e| e.binary(level + 1));
            lhs = match self.apply(op, lhs, rhs) {
                Some(value) => value,
                None => return self.fail("division by zero in `#if`"),
            };
        }
    }

    /// Applies a binary operator; `None` is a division by zero in a used operand.
    fn apply(&self, op: Punct, a: Value, b: Value) -> Option<Value> {
        let unsigned = a.unsigned || b.unsigned;
        let number = |bits: u64| Value { bits, unsigned };
        let order = if unsigned {
            a.bits.cmp(&b.bits)
        } else {
            (a.bits as i64).cmp(&(b.bits as i64))
        };
        let shift = b.bits.min(63) as u32;
        Some(match op {
            Punct::OrOr => Value::truth(a.is_true() || b.is_true()),
            Punct::AndAnd => Value::truth(a.is_true() && b.is_true()),
            Punct::Pipe => number(a.bits | b.bits),
            Punct::Caret => number(a.bits ^ b.bits),
            Punct::Amp => number(a.bits & b.bits),
            Punct::EqEq => Value::truth(a.bits == b.bits),
            Punct::NotEq => Value::truth(a.bits != b.bits),
            Punct::Lt => Value::truth(order.is_lt()),
            Punct::Gt => Value::truth(order.is_gt()),
            Punct::LtEq => Value::truth(order.is_le()),
            Punct::GtEq => Value::truth(order.is_ge()),
            Punct::Shl => Value {
                bits: a.bits << shift,
                ..a
            },
            Punct::Shr if a.unsigned => Value {
                bits: a.bits >> shift,
                ..a
            },
            Punct::Shr => Value {
                bits: ((a.bits as i64) >> shift) as u64,
                ..a
            },
            Punct::Plus => number(a.bits.wrapping_add(b.bits)),
            Punct::Minus => number(a.bits.wrapping_sub(b.bits)),
            Punct::Star => number(a.bits.wrapping_mul(b.bits)),
            Punct::Slash | Punct::Percent if b.bits == 0 => {
                return (self.skip > 0).then_some(number(0));
            }
            Punct::Slash if unsigned => number(a.bits / b.bits),
            Punct::Slash => number((a.bits as i64).wrapping_div(b.bits as i64) as u64),
            _ if unsigned => number(a.bits % b.bits),
            _ => number((a.bits as i64).wrapping_rem(b.bits as i64) as u64),
        })
    }

    fn unary(&mut self) -> Value {
        self.nested(Self::unary_inner)
    }

    fn unary_inner(&mut self) -> Value {
        let Some(t) = self.tokens.get(self.pos) else {
            return self.fail("`#if` ends too early");
        };
        self.pos += 1;
        match t.kind {
            TokenKind::Punct(Punct::Minus) => {
                let v = self.unary();
                Value {
                    bits: v.bits.wrapping_neg(),
                    ..v
                }
            }
            TokenKind::Punct(Punct::Plus) => self.unary(),
            TokenKind::Punct(Punct::Bang) => Value::truth(!self.unary().is_true()),
            TokenKind::Punct(Punct::Tilde) => {
                let v = self.unary();
                Value { bits: !v.bits, ..v }
            }
            TokenKind::Punct(Punct::LParen) => {
                let v = self.ternary();
                if self.eat(Punct::RParen) {
                    v
                } else {
                    self.fail("expected `)` in `#if`")
                }
            }
            TokenKind::Int => match parse_int(&t.text) {
                Some((bits, unsigned)) => Value { bits, unsigned },
                None => {
                    self.pos -= 1;
                    self.fail("a bad integer in `#if`")
                }
            },
            TokenKind::Ident => Value::signed(0),
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
/// Returns its bits and if it is unsigned: a `u` suffix, or too large for `i64`.
pub(crate) fn parse_int(text: &str) -> Option<(u64, bool)> {
    let digits = text.trim_end_matches(['u', 'U', 'l', 'L']);
    let suffix = &text[digits.len()..];
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
    Some((
        value,
        suffix.contains(['u', 'U']) || value > i64::MAX as u64,
    ))
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
    fn expensive_expansions_stop_at_the_work_budget() {
        let hides: String = (1..=30_000)
            .map(|i| format!("#define M{i} M{} y\n", i - 1))
            .collect();
        let inputs = [
            format!("#define F(x) x x\n{}1{}", "F(".repeat(40), ")".repeat(40)),
            format!(
                "#define S(x){}\nS({})",
                " #x".repeat(100_000),
                "a ".repeat(100_000)
            ),
            format!(
                "#define P a{}\n{}",
                "##a".repeat(20_000),
                "P ".repeat(50_000)
            ),
            format!(
                "#define E\n#define F(x){}\nF({})",
                " x".repeat(200_000),
                "E ".repeat(200_000)
            ),
            format!("#define M0 x\n{hides}M30000"),
        ];
        for text in inputs {
            let (_, d) = run_with(&text, &mut NoIncludes);
            assert!(
                d.iter().any(|d| d.message.contains("more than")),
                "{text:.50}"
            );
        }
    }

    #[test]
    fn a_macro_with_many_parameters_expands_in_linear_time() {
        let n = 20_000;
        let params: Vec<String> = (0..n).map(|i| format!("p{i}")).collect();
        let text = format!(
            "#define F({}) {}\nF({}1)",
            params.join(","),
            "y ".repeat(100_000),
            ",".repeat(n - 1)
        );
        let (out, d) = run_with(&text, &mut NoIncludes);
        assert!(d.is_empty(), "{d:?}");
        assert_eq!(out.len(), "y ".repeat(100_000).len() - 1);
        assert_eq!(errors("#define F(a, a) a").len(), 1);
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
                ("twice.h", false) => Ok((
                    path.into(),
                    "#include \"twice.h\"\n#include \"twice.h\"\n".into(),
                )),
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
        // A header that includes itself twice would open 2^32 files.
        let (_, d) = run_with("#include \"twice.h\"", &mut Files);
        let count = d
            .iter()
            .filter(|d| d.message.contains("files are included"))
            .count();
        assert_eq!(count, 1, "{:?}", d.len());
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
        assert_eq!(parse_int("0x1fU"), Some((31, true)));
        assert_eq!(parse_int("017"), Some((15, false)));
        assert_eq!(parse_int("0"), Some((0, false)));
        assert_eq!(parse_int("0xFFFFFFFFFFFFFFFF"), Some((u64::MAX, true)));
        assert_eq!(parse_int("9z"), None);
    }

    #[test]
    fn logical_operators_and_the_conditional_short_circuit() {
        let text = "#if defined(N) && 100 / N > 2\nno\n#elif 1 || 1 / 0\nyes\n#endif\n#if 1 ? 2 : 1 % 0\nyes\n#endif";
        assert_eq!(run(text), "yes yes");
        assert_eq!(errors("#if 1 && 1 / 0\n#endif").len(), 1);
    }

    #[test]
    fn unsigned_operands_make_unsigned_arithmetic() {
        assert_eq!(run("#if -1 > 0u\nyes\n#endif"), "yes");
        assert_eq!(run("#if 0xFFFFFFFFFFFFFFFF > 0\nyes\n#endif"), "yes");
        assert_eq!(run("#if -1 > 0\nno\n#endif"), "");
        assert_eq!(
            run("#if (0u - 1) >> 63 == 1 && -2 >> 1 == -1\nyes\n#endif"),
            "yes"
        );
        assert_eq!(
            run("#if -7 / 2 == -3 && -7 % 2 == -1 && 1 << 99 != 0\nyes\n#endif"),
            "yes"
        );
    }

    #[test]
    fn deep_if_expressions_are_an_error_not_a_crash() {
        let n = 100_000;
        for expr in [
            format!("{}1{}", "(".repeat(n), ")".repeat(n)),
            format!("{}1", "- ".repeat(n)),
            format!("{}1{}", "1 ? ".repeat(n), " : 1".repeat(n)),
        ] {
            let e = errors(&format!("#if {expr}\n#endif"));
            assert!(e.iter().any(|m| m.contains("nested too deep")), "{e:?}");
        }
    }

    #[test]
    fn an_unclosed_include_name_is_an_error_not_a_panic() {
        for text in ["#include \"é", "#include \"é\n", "#include \""] {
            assert!(!errors(text).is_empty(), "{text}");
        }
    }

    #[test]
    fn the_error_directive_keeps_the_text_as_written() {
        assert_eq!(
            errors("#error unsupported: x-y  (now)"),
            ["#error unsupported: x-y  (now)"]
        );
        assert_eq!(errors("#error"), ["#error"]);
    }
}
