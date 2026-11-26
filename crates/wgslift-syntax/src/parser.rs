//! The parser: preprocessed tokens to a syntax tree.
//!
//! After an error the parser skips to the end of the statement or item and goes
//! on, so one run reports all independent errors.

use std::collections::HashSet;

use crate::ast::*;
use crate::diagnostic::Diagnostic;
use crate::lexer::{Punct, Token, TokenKind};
use crate::span::{FileId, Span};

/// The deepest nesting of expressions and statements the parser accepts.
pub const MAX_DEPTH: usize = 64;
/// The deepest expression tree the parser builds. Any recursive walk over a
/// tree, `Drop` included, then stays within a small stack.
pub const MAX_TREE_DEPTH: usize = 256;

const KEYWORDS: [&str; 23] = [
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "default",
    "break",
    "continue",
    "discard",
    "return",
    "struct",
    "typedef",
    "cbuffer",
    "tbuffer",
    "true",
    "false",
    "technique",
    "technique10",
    "technique11",
    "pass",
    "sampler_state",
];

const SCALARS: [&str; 21] = [
    "bool",
    "int",
    "uint",
    "dword",
    "half",
    "float",
    "double",
    "min16float",
    "min10float",
    "min16int",
    "min12int",
    "min16uint",
    "int16_t",
    "uint16_t",
    "int32_t",
    "uint32_t",
    "int64_t",
    "uint64_t",
    "float16_t",
    "float32_t",
    "float64_t",
];

/// Types that take arguments in `<>`.
const TEMPLATE_TYPES: [&str; 28] = [
    "vector",
    "matrix",
    "Texture1D",
    "Texture1DArray",
    "Texture2D",
    "Texture2DArray",
    "Texture2DMS",
    "Texture2DMSArray",
    "Texture3D",
    "TextureCube",
    "TextureCubeArray",
    "Buffer",
    "RWBuffer",
    "StructuredBuffer",
    "RWStructuredBuffer",
    "AppendStructuredBuffer",
    "ConsumeStructuredBuffer",
    "RWTexture1D",
    "RWTexture1DArray",
    "RWTexture2D",
    "RWTexture2DArray",
    "RWTexture3D",
    "ConstantBuffer",
    "PointStream",
    "LineStream",
    "TriangleStream",
    "InputPatch",
    "OutputPatch",
];

const OTHER_TYPES: [&str; 15] = [
    "void",
    "string",
    "sampler",
    "sampler1D",
    "sampler2D",
    "sampler3D",
    "samplerCUBE",
    "SamplerState",
    "SamplerComparisonState",
    "texture",
    "texture1D",
    "texture2D",
    "texture3D",
    "textureCUBE",
    "ByteAddressBuffer",
];

fn is_keyword(word: &str) -> bool {
    KEYWORDS.contains(&word)
}

fn is_sampler_type(word: &str) -> bool {
    word.starts_with("sampler") || word == "SamplerState" || word == "SamplerComparisonState"
}

fn is_template_type(word: &str) -> bool {
    TEMPLATE_TYPES.contains(&word)
}

/// `word` names a type that HLSL defines, for example `float3x4` or `sampler2D`.
pub fn is_builtin_type(word: &str) -> bool {
    if is_template_type(word) || OTHER_TYPES.contains(&word) || word == "RWByteAddressBuffer" {
        return true;
    }
    let dim = |c: u8| (b'1'..=b'4').contains(&c);
    SCALARS.iter().any(|scalar| {
        word.strip_prefix(scalar)
            .is_some_and(|rest| match rest.as_bytes() {
                [] => true,
                [n] => dim(*n),
                [n, b'x', m] => dim(*n) && dim(*m),
                _ => false,
            })
    })
}

/// An error was reported; the caller recovers.
struct Stop;

type PResult<T> = Result<T, Stop>;

/// Parses preprocessed tokens into a syntax tree.
pub fn parse_tokens(tokens: Vec<Token>, diagnostics: &mut Vec<Diagnostic>) -> TranslationUnit {
    let eof = tokens.last().map_or(Span::new(FileId(0), 0, 0), |t| {
        let site = t.site();
        Span {
            start: site.end,
            ..site
        }
    });
    let mut parser = Parser {
        tokens,
        pos: 0,
        types: HashSet::new(),
        diagnostics,
        depth: 0,
        tree_depth: 0,
        eof,
    };
    let mut items = Vec::new();
    while parser.pos < parser.tokens.len() {
        let start = parser.pos;
        if parser.item(&mut items).is_err() {
            parser.recover(false);
        }
        if parser.pos == start {
            parser.pos += 1;
        }
    }
    TranslationUnit { items }
}

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    /// Struct and typedef names declared so far.
    types: HashSet<String>,
    diagnostics: &'a mut Vec<Diagnostic>,
    depth: usize,
    /// The depth of the expression tree that was built last.
    tree_depth: usize,
    eof: Span,
}

impl Parser<'_> {
    // Token access.

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn peek_at(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.pos + n)
    }

    fn at(&self, punct: Punct) -> bool {
        self.peek().is_some_and(|t| t.is(punct))
    }

    fn at_word(&self, word: &str) -> bool {
        self.peek().is_some_and(|t| t.is_word(word))
    }

    fn eat(&mut self, punct: Punct) -> bool {
        let found = self.at(punct);
        self.pos += usize::from(found);
        found
    }

    fn eat_word(&mut self, word: &str) -> bool {
        let found = self.at_word(word);
        self.pos += usize::from(found);
        found
    }

    /// Takes one `>`. A `>>` is split, so that `vector<vector<float, 2>>` closes twice.
    fn eat_gt(&mut self) -> bool {
        if self.eat(Punct::Gt) {
            return true;
        }
        if let Some(t) = self.tokens.get_mut(self.pos)
            && t.is(Punct::Shr)
        {
            t.kind = TokenKind::Punct(Punct::Gt);
            t.text = ">".to_string();
            t.span.start += 1;
            return true;
        }
        false
    }

    fn here(&self) -> Span {
        self.peek().map_or(self.eof, Token::site)
    }

    fn prev(&self) -> Span {
        self.pos
            .checked_sub(1)
            .and_then(|i| self.tokens.get(i))
            .map_or(self.eof, Token::site)
    }

    fn since(&self, start: Span) -> Span {
        start.to(self.prev())
    }

    fn error(&mut self, span: Span, message: impl Into<String>) -> Stop {
        self.diagnostics.push(Diagnostic::error(span, message));
        Stop
    }

    fn expected(&mut self, what: &str) -> Stop {
        let message = match self.peek() {
            Some(t) => format!("expected {what}, found `{}`", t.text),
            None => format!("expected {what}, found the end of the input"),
        };
        let here = self.here();
        self.error(here, message)
    }

    fn expect(&mut self, punct: Punct) -> PResult<()> {
        if self.eat(punct) {
            Ok(())
        } else {
            Err(self.expected(&format!("`{}`", punct.as_str())))
        }
    }

    /// Runs `f` one level deeper, or reports an error past `MAX_DEPTH`.
    fn nested<T>(&mut self, f: impl FnOnce(&mut Self) -> PResult<T>) -> PResult<T> {
        if self.depth >= MAX_DEPTH {
            let here = self.here();
            return Err(self.error(
                here,
                format!("the code is nested more than {MAX_DEPTH} levels deep"),
            ));
        }
        self.depth += 1;
        let result = f(self);
        self.depth -= 1;
        result
    }

    /// Skips to the end of the statement or item: past a `;` or a `{}` block. In a
    /// block, a `}` that closes the block stays for the block.
    fn recover(&mut self, in_block: bool) {
        self.skip(in_block, false);
    }

    /// Like `recover` in a block, but also stops before a `case` or `default` label.
    fn recover_in_switch(&mut self) {
        self.skip(true, true);
    }

    fn skip(&mut self, in_block: bool, stop_at_labels: bool) {
        let mut depth = 0usize;
        while let Some(t) = self.peek() {
            if stop_at_labels && depth == 0 && (t.is_word("case") || t.is_word("default")) {
                return;
            }
            if t.is(Punct::LBrace) {
                depth += 1;
            } else if t.is(Punct::RBrace) {
                if depth == 0 {
                    if !in_block {
                        self.pos += 1;
                    }
                    return;
                }
                depth -= 1;
                if depth == 0 {
                    self.pos += 1;
                    return;
                }
            } else if t.is(Punct::Semi) && depth == 0 {
                self.pos += 1;
                return;
            }
            self.pos += 1;
        }
    }

    // Names and types.

    fn ident(&mut self) -> PResult<Ident> {
        match self.peek() {
            Some(t) if t.kind == TokenKind::Ident && !is_keyword(&t.text) => {
                let ident = Ident {
                    name: t.text.clone(),
                    span: t.site(),
                };
                self.pos += 1;
                Ok(ident)
            }
            _ => Err(self.expected("a name")),
        }
    }

    /// Any word, also a keyword: for members, swizzles and sampler states.
    fn word(&mut self) -> PResult<Ident> {
        match self.peek() {
            Some(t) if t.kind == TokenKind::Ident => {
                let ident = Ident {
                    name: t.text.clone(),
                    span: t.site(),
                };
                self.pos += 1;
                Ok(ident)
            }
            _ => Err(self.expected("a name")),
        }
    }

    fn is_type(&self, t: Option<&Token>) -> bool {
        t.is_some_and(|t| {
            t.kind == TokenKind::Ident && (is_builtin_type(&t.text) || self.types.contains(&t.text))
        })
    }

    fn type_ref(&mut self) -> PResult<TypeRef> {
        let name = self.ident().map_err(|_| {
            self.diagnostics.pop();
            self.expected("a type")
        })?;
        let mut args = Vec::new();
        if is_template_type(&name.name) && self.eat(Punct::Lt) {
            args = self.nested(|p| {
                let mut args = Vec::new();
                loop {
                    if p.is_type(p.peek()) {
                        args.push(TypeArg::Type(p.type_ref()?));
                    } else {
                        // Above the comparisons and shifts, so that `>` closes the list.
                        args.push(TypeArg::Expr(p.binary(8)?));
                    }
                    if p.eat_gt() {
                        return Ok(args);
                    }
                    p.expect(Punct::Comma)?;
                }
            })?;
        }
        Ok(TypeRef {
            span: self.since(name.span),
            name,
            args,
        })
    }

    fn modifiers(&mut self) -> Vec<Modifier> {
        let mut modifiers = Vec::new();
        while let (Some(t), Some(next)) = (self.peek(), self.peek_at(1)) {
            match Modifier::from_word(&t.text) {
                Some(m) if t.kind == TokenKind::Ident && next.kind == TokenKind::Ident => {
                    modifiers.push(m);
                    self.pos += 1;
                }
                _ => break,
            }
        }
        modifiers
    }

    /// The tokens from here start a declaration: modifiers, then a type and a name.
    fn at_declaration(&self) -> bool {
        let mut p = self.pos;
        while let (Some(t), Some(next)) = (self.tokens.get(p), self.tokens.get(p + 1)) {
            if t.kind == TokenKind::Ident
                && next.kind == TokenKind::Ident
                && Modifier::from_word(&t.text).is_some()
            {
                p += 1;
            } else {
                break;
            }
        }
        let Some(t) = self.tokens.get(p) else {
            return false;
        };
        if t.kind != TokenKind::Ident || is_keyword(&t.text) {
            return false;
        }
        p += 1;
        if is_template_type(&t.text) && self.tokens.get(p).is_some_and(|n| n.is(Punct::Lt)) {
            // Skip the template arguments: `vector<float, 2> v` declares, but
            // `vector<float, 2>(1, 2)` constructs.
            let mut depth = 0usize;
            while let Some(n) = self.tokens.get(p) {
                p += 1;
                match n.kind {
                    TokenKind::Punct(Punct::Lt) => depth += 1,
                    TokenKind::Punct(Punct::Gt) => depth -= 1,
                    TokenKind::Punct(Punct::Shr) => depth = depth.saturating_sub(2),
                    TokenKind::Punct(Punct::Semi | Punct::LBrace | Punct::RBrace) => return false,
                    _ => {}
                }
                if depth == 0 {
                    break;
                }
            }
        }
        self.tokens
            .get(p)
            .is_some_and(|n| n.kind == TokenKind::Ident && !is_keyword(&n.text))
    }

    // Items.

    fn item(&mut self, items: &mut Vec<Item>) -> PResult<()> {
        if self.eat(Punct::Semi) {
            return Ok(());
        }
        let start = self.here();
        if self.at_word("struct") {
            let def = self.struct_def()?;
            let ty = TypeRef {
                name: def.name.clone(),
                args: Vec::new(),
                span: def.name.span,
            };
            items.push(Item::Struct(def));
            if self.eat(Punct::Semi) {
                return Ok(());
            }
            let first = self.ident()?;
            let declarators = self.declarators(first, &ty)?;
            self.expect(Punct::Semi)?;
            items.push(Item::Variable(VarDecl {
                modifiers: Vec::new(),
                ty,
                declarators,
                span: self.since(start),
            }));
            return Ok(());
        }
        if self.at_word("typedef") {
            let typedef = self.typedef()?;
            items.push(Item::Typedef(typedef));
            return Ok(());
        }
        if self.at_word("cbuffer") || self.at_word("tbuffer") {
            let buffer = self.buffer()?;
            items.push(Item::Buffer(buffer));
            return Ok(());
        }
        if ["technique", "technique10", "technique11"]
            .iter()
            .any(|w| self.at_word(w))
        {
            return Err(self.error(start, "effect techniques are not supported"));
        }
        let attributes = self.attributes()?;
        let modifiers = self.modifiers();
        let ty = self.type_ref()?;
        let name = self.ident()?;
        if self.at(Punct::LParen) {
            let function = self.function(start, attributes, modifiers, ty, name)?;
            items.push(Item::Function(function));
            return Ok(());
        }
        if let Some(a) = attributes.first() {
            return Err(self.error(
                a.span,
                "attributes are only allowed on functions and statements",
            ));
        }
        let declarators = self.declarators(name, &ty)?;
        self.expect(Punct::Semi)?;
        items.push(Item::Variable(VarDecl {
            modifiers,
            ty,
            declarators,
            span: self.since(start),
        }));
        Ok(())
    }

    fn struct_def(&mut self) -> PResult<StructDef> {
        let start = self.here();
        self.pos += 1;
        let name = self.ident()?;
        self.types.insert(name.name.clone());
        self.expect(Punct::LBrace)?;
        let fields = self.members()?;
        Ok(StructDef {
            name,
            fields,
            span: self.since(start),
        })
    }

    /// Runs `item` for each entry of a `{}` body after the `{`, through the `}`.
    /// After an error it goes on at the next entry.
    fn body_list<T>(&mut self, mut item: impl FnMut(&mut Self) -> PResult<T>) -> PResult<Vec<T>> {
        let mut list = Vec::new();
        while !self.eat(Punct::RBrace) {
            if self.peek().is_none() {
                return Err(self.expected("`}`"));
            }
            let before = self.pos;
            match item(self) {
                Ok(entry) => list.push(entry),
                Err(Stop) => {
                    self.recover(true);
                    if self.pos == before && !self.at(Punct::RBrace) {
                        self.pos += 1;
                    }
                }
            }
        }
        Ok(list)
    }

    /// The declarations of a struct or buffer body, through the `}`.
    fn members(&mut self) -> PResult<Vec<VarDecl>> {
        self.body_list(Self::var_decl)
    }

    fn typedef(&mut self) -> PResult<Typedef> {
        let start = self.here();
        self.pos += 1;
        let modifiers = self.modifiers();
        let ty = self.type_ref()?;
        let mut names = Vec::new();
        loop {
            let name = self.ident()?;
            self.types.insert(name.name.clone());
            let array = self.array_dims()?;
            names.push((name, array));
            if !self.eat(Punct::Comma) {
                break;
            }
        }
        self.expect(Punct::Semi)?;
        Ok(Typedef {
            modifiers,
            ty,
            names,
            span: self.since(start),
        })
    }

    fn buffer(&mut self) -> PResult<BufferBlock> {
        let start = self.here();
        let kind = if self.eat_word("cbuffer") {
            BufferKind::Constant
        } else {
            self.pos += 1;
            BufferKind::Texture
        };
        let name = self.ident()?;
        let bindings = self.bindings()?;
        self.expect(Punct::LBrace)?;
        let members = self.members()?;
        self.eat(Punct::Semi);
        Ok(BufferBlock {
            kind,
            name,
            bindings,
            members,
            span: self.since(start),
        })
    }

    fn function(
        &mut self,
        start: Span,
        attributes: Vec<Attribute>,
        modifiers: Vec<Modifier>,
        return_type: TypeRef,
        name: Ident,
    ) -> PResult<Function> {
        self.expect(Punct::LParen)?;
        let mut params = Vec::new();
        if self.at_word("void") && self.peek_at(1).is_some_and(|t| t.is(Punct::RParen)) {
            self.pos += 1;
        }
        if !self.eat(Punct::RParen) {
            loop {
                params.push(self.param()?);
                if self.eat(Punct::RParen) {
                    break;
                }
                self.expect(Punct::Comma)?;
            }
        }
        let return_bindings = self.bindings()?;
        let body = if self.eat(Punct::Semi) {
            None
        } else {
            Some(self.block()?)
        };
        Ok(Function {
            attributes,
            modifiers,
            return_type,
            name,
            params,
            return_bindings,
            body,
            span: self.since(start),
        })
    }

    fn param(&mut self) -> PResult<Param> {
        let start = self.here();
        let modifiers = self.modifiers();
        let ty = self.type_ref()?;
        let name = match self.peek() {
            Some(t) if t.kind == TokenKind::Ident && !is_keyword(&t.text) => Some(self.ident()?),
            _ => None,
        };
        let array = self.array_dims()?;
        let bindings = self.bindings()?;
        let default = if self.eat(Punct::Eq) {
            Some(self.assign()?)
        } else {
            None
        };
        Ok(Param {
            modifiers,
            ty,
            name,
            array,
            bindings,
            default,
            span: self.since(start),
        })
    }

    fn attributes(&mut self) -> PResult<Vec<Attribute>> {
        let mut attributes = Vec::new();
        while self.at(Punct::LBracket) {
            let start = self.here();
            self.pos += 1;
            let name = self.word()?;
            let args = if self.at(Punct::LParen) {
                self.call_args()?
            } else {
                Vec::new()
            };
            self.expect(Punct::RBracket)?;
            attributes.push(Attribute {
                name,
                args,
                span: self.since(start),
            });
        }
        Ok(attributes)
    }

    // Declarations.

    fn var_decl(&mut self) -> PResult<VarDecl> {
        let start = self.here();
        let modifiers = self.modifiers();
        let ty = self.type_ref()?;
        let first = self.ident()?;
        let declarators = self.declarators(first, &ty)?;
        self.expect(Punct::Semi)?;
        Ok(VarDecl {
            modifiers,
            ty,
            declarators,
            span: self.since(start),
        })
    }

    fn declarators(&mut self, first: Ident, ty: &TypeRef) -> PResult<Vec<Declarator>> {
        let mut declarators = vec![self.declarator(first, ty)?];
        while self.eat(Punct::Comma) {
            let name = self.ident()?;
            declarators.push(self.declarator(name, ty)?);
        }
        Ok(declarators)
    }

    fn declarator(&mut self, name: Ident, ty: &TypeRef) -> PResult<Declarator> {
        let array = self.array_dims()?;
        let bindings = self.bindings()?;
        let annotations = if self.at(Punct::Lt) {
            Some(self.annotations()?)
        } else {
            None
        };
        let init = if self.eat(Punct::Eq) {
            Some(self.initializer()?)
        } else if self.at(Punct::LBrace) && is_sampler_type(&ty.name.name) {
            // The D3D10 form: `SamplerState s { Filter = ...; };`
            let start = self.here();
            self.pos += 1;
            let states = self.body_list(Self::state_assignment)?;
            Some(Initializer::SamplerState(states, self.since(start)))
        } else {
            None
        };
        Ok(Declarator {
            span: self.since(name.span),
            name,
            array,
            bindings,
            annotations,
            init,
        })
    }

    fn array_dims(&mut self) -> PResult<Vec<Option<Expr>>> {
        let mut dims = Vec::new();
        while self.eat(Punct::LBracket) {
            if self.eat(Punct::RBracket) {
                dims.push(None);
            } else {
                dims.push(Some(self.expr()?));
                self.expect(Punct::RBracket)?;
            }
        }
        Ok(dims)
    }

    fn bindings(&mut self) -> PResult<Bindings> {
        let mut bindings = Bindings::default();
        while self.eat(Punct::Colon) {
            let start = self.here();
            if self.eat_word("register") {
                self.expect(Punct::LParen)?;
                let mut words = vec![self.word()?];
                while words.len() < 3 && self.eat(Punct::Comma) {
                    words.push(self.word()?);
                }
                self.expect(Punct::RParen)?;
                let is_space = |w: &Ident| {
                    w.name
                        .strip_prefix("space")
                        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
                };
                let space = match words.last() {
                    Some(last) if words.len() > 1 && is_space(last) => words.pop(),
                    _ => None,
                };
                let slot = words.pop().expect("one word at least");
                let profile = words.pop();
                if !words.is_empty() {
                    return Err(self.error(self.since(start), "too many words in `register(...)`"));
                }
                bindings.registers.push(Register {
                    profile,
                    slot,
                    space,
                    span: self.since(start),
                });
            } else if self.eat_word("packoffset") {
                self.expect(Punct::LParen)?;
                let slot = self.word()?;
                let component = if self.eat(Punct::Dot) {
                    Some(self.word()?)
                } else {
                    None
                };
                self.expect(Punct::RParen)?;
                bindings.packoffset = Some(PackOffset {
                    slot,
                    component,
                    span: self.since(start),
                });
            } else {
                bindings.semantic = Some(self.word()?);
            }
        }
        Ok(bindings)
    }

    /// Skips an annotation block `<...>` and returns its span.
    fn annotations(&mut self) -> PResult<Span> {
        let start = self.here();
        self.pos += 1;
        let mut depth = 1usize;
        while let Some(t) = self.peek() {
            match t.kind {
                TokenKind::Punct(Punct::Lt) => depth += 1,
                TokenKind::Punct(Punct::Gt) => depth -= 1,
                TokenKind::Punct(Punct::Shr) => depth = depth.saturating_sub(2),
                _ => {}
            }
            self.pos += 1;
            if depth == 0 {
                return Ok(self.since(start));
            }
        }
        Err(self.expected("`>` to close the annotations"))
    }

    fn initializer(&mut self) -> PResult<Initializer> {
        self.nested(|p| {
            let start = p.here();
            if p.eat(Punct::LBrace) {
                let mut list = Vec::new();
                while !p.eat(Punct::RBrace) {
                    list.push(p.initializer()?);
                    if !p.eat(Punct::Comma) {
                        p.expect(Punct::RBrace)?;
                        break;
                    }
                }
                return Ok(Initializer::List(list, p.since(start)));
            }
            if p.eat_word("sampler_state") {
                p.expect(Punct::LBrace)?;
                let states = p.body_list(Self::state_assignment)?;
                return Ok(Initializer::SamplerState(states, p.since(start)));
            }
            Ok(Initializer::Expr(p.assign()?))
        })
    }

    /// `Name = value;` in a sampler state block.
    fn state_assignment(&mut self) -> PResult<StateAssignment> {
        let name = self.word()?;
        self.expect(Punct::Eq)?;
        let value = if self.at(Punct::Lt) {
            let open = self.here();
            self.pos += 1;
            let texture = self.ident()?;
            if !self.eat_gt() {
                return Err(self.expected("`>`"));
            }
            Expr {
                kind: ExprKind::Ident(texture.name),
                span: self.since(open),
            }
        } else {
            self.assign()?
        };
        self.expect(Punct::Semi)?;
        Ok(StateAssignment {
            span: self.since(name.span),
            name,
            value,
        })
    }

    // Statements.

    fn block(&mut self) -> PResult<Block> {
        let start = self.here();
        self.expect(Punct::LBrace)?;
        let mut stmts = Vec::new();
        while !self.at(Punct::RBrace) {
            if self.peek().is_none() {
                return Err(self.expected("`}`"));
            }
            let before = self.pos;
            match self.stmt() {
                Ok(stmt) => stmts.push(stmt),
                Err(Stop) => self.recover(true),
            }
            if self.pos == before {
                self.pos += 1;
            }
        }
        self.pos += 1;
        Ok(Block {
            stmts,
            span: self.since(start),
        })
    }

    fn stmt(&mut self) -> PResult<Stmt> {
        self.nested(Self::stmt_inner)
    }

    fn stmt_inner(&mut self) -> PResult<Stmt> {
        let start = self.here();
        let attributes = self.attributes()?;
        let kind = self.stmt_kind()?;
        Ok(Stmt {
            attributes,
            kind,
            span: self.since(start),
        })
    }

    fn stmt_kind(&mut self) -> PResult<StmtKind> {
        if self.at(Punct::LBrace) {
            return Ok(StmtKind::Block(self.block()?));
        }
        if self.eat(Punct::Semi) {
            return Ok(StmtKind::Empty);
        }
        let word = self
            .peek()
            .filter(|t| t.kind == TokenKind::Ident)
            .map(|t| t.text.clone())
            .unwrap_or_default();
        match word.as_str() {
            "if" => self.if_stmt(),
            "for" => self.for_stmt(),
            "while" => self.while_stmt(),
            "do" => self.do_stmt(),
            "switch" => self.switch_stmt(),
            "break" | "continue" | "discard" | "return" => self.jump_stmt(),
            _ => self.simple_stmt(),
        }
    }

    fn if_stmt(&mut self) -> PResult<StmtKind> {
        self.pos += 1;
        let cond = self.paren_expr()?;
        let then = Box::new(self.stmt()?);
        let otherwise = if self.eat_word("else") {
            Some(Box::new(self.stmt()?))
        } else {
            None
        };
        Ok(StmtKind::If {
            cond,
            then,
            otherwise,
        })
    }

    fn while_stmt(&mut self) -> PResult<StmtKind> {
        self.pos += 1;
        let cond = self.paren_expr()?;
        let body = Box::new(self.stmt()?);
        Ok(StmtKind::While { cond, body })
    }

    fn do_stmt(&mut self) -> PResult<StmtKind> {
        self.pos += 1;
        let body = Box::new(self.stmt()?);
        if !self.eat_word("while") {
            return Err(self.expected("`while`"));
        }
        let cond = self.paren_expr()?;
        self.expect(Punct::Semi)?;
        Ok(StmtKind::DoWhile { body, cond })
    }

    fn jump_stmt(&mut self) -> PResult<StmtKind> {
        let kind = if self.eat_word("break") {
            StmtKind::Break
        } else if self.eat_word("continue") {
            StmtKind::Continue
        } else if self.eat_word("discard") {
            StmtKind::Discard
        } else {
            self.pos += 1;
            if self.at(Punct::Semi) {
                StmtKind::Return(None)
            } else {
                StmtKind::Return(Some(self.expr()?))
            }
        };
        self.expect(Punct::Semi)?;
        Ok(kind)
    }

    /// A declaration or an expression statement.
    fn simple_stmt(&mut self) -> PResult<StmtKind> {
        if self.at_declaration() {
            return Ok(StmtKind::Var(self.var_decl()?));
        }
        let expr = self.expr()?;
        self.expect(Punct::Semi)?;
        Ok(StmtKind::Expr(expr))
    }

    fn paren_expr(&mut self) -> PResult<Expr> {
        self.expect(Punct::LParen)?;
        let expr = self.expr()?;
        self.expect(Punct::RParen)?;
        Ok(expr)
    }

    fn for_stmt(&mut self) -> PResult<StmtKind> {
        self.pos += 1;
        self.expect(Punct::LParen)?;
        let start = self.here();
        let init = if self.eat(Punct::Semi) {
            None
        } else {
            let kind = if self.at_declaration() {
                StmtKind::Var(self.var_decl()?)
            } else {
                let expr = self.expr()?;
                self.expect(Punct::Semi)?;
                StmtKind::Expr(expr)
            };
            Some(Box::new(Stmt {
                attributes: Vec::new(),
                kind,
                span: self.since(start),
            }))
        };
        let cond = if self.at(Punct::Semi) {
            None
        } else {
            Some(self.expr()?)
        };
        self.expect(Punct::Semi)?;
        let step = if self.at(Punct::RParen) {
            None
        } else {
            Some(self.expr()?)
        };
        self.expect(Punct::RParen)?;
        let body = Box::new(self.stmt()?);
        Ok(StmtKind::For {
            init,
            cond,
            step,
            body,
        })
    }

    fn switch_stmt(&mut self) -> PResult<StmtKind> {
        self.pos += 1;
        let value = self.paren_expr()?;
        self.expect(Punct::LBrace)?;
        let mut cases: Vec<SwitchCase> = Vec::new();
        while !self.eat(Punct::RBrace) {
            if self.peek().is_none() {
                return Err(self.expected("`}`"));
            }
            let before = self.pos;
            if self.at_word("case") || self.at_word("default") {
                let start = self.here();
                let mut labels = Vec::new();
                let parsed = self.labels(&mut labels);
                // A case with a bad label stays, so that its statements have a place.
                cases.push(SwitchCase {
                    labels,
                    stmts: Vec::new(),
                    span: self.since(start),
                });
                if parsed.is_err() {
                    self.recover_in_switch();
                }
                continue;
            }
            let stmt = match cases.last() {
                Some(_) => self.stmt(),
                None => Err(self.expected("`case` or `default`")),
            };
            match stmt {
                Ok(stmt) => {
                    let case = cases.last_mut().expect("checked above");
                    case.span = case.span.to(stmt.span);
                    case.stmts.push(stmt);
                }
                Err(Stop) => {
                    self.recover_in_switch();
                    let at_label = self.at_word("case") || self.at_word("default");
                    if self.pos == before && !self.at(Punct::RBrace) && !at_label {
                        self.pos += 1;
                    }
                }
            }
        }
        Ok(StmtKind::Switch { value, cases })
    }

    /// One or more `case value:` and `default:` labels.
    fn labels(&mut self, labels: &mut Vec<Option<Expr>>) -> PResult<()> {
        while self.at_word("case") || self.at_word("default") {
            if self.eat_word("case") {
                labels.push(Some(self.ternary()?));
            } else {
                self.pos += 1;
                labels.push(None);
            }
            self.expect(Punct::Colon)?;
        }
        Ok(())
    }

    // Expressions. Each function that returns an expression leaves the depth of
    // its tree in `tree_depth`, and `node` builds every node and checks the limit.

    /// Builds a node one level above its deepest child, of depth `child_depth`.
    fn node(&mut self, kind: ExprKind, span: Span, child_depth: usize) -> PResult<Expr> {
        if child_depth >= MAX_TREE_DEPTH {
            return Err(self.error(
                span,
                format!("the expression is nested more than {MAX_TREE_DEPTH} levels deep"),
            ));
        }
        self.tree_depth = child_depth + 1;
        Ok(Expr { kind, span })
    }

    fn expr(&mut self) -> PResult<Expr> {
        let first = self.assign()?;
        if !self.at(Punct::Comma) {
            return Ok(first);
        }
        let start = first.span;
        let mut deepest = self.tree_depth;
        let mut list = vec![first];
        while self.eat(Punct::Comma) {
            list.push(self.assign()?);
            deepest = deepest.max(self.tree_depth);
        }
        let span = self.since(start);
        self.node(ExprKind::Comma(list), span, deepest)
    }

    /// Assignment is right-associative. A loop collects the chain, so that
    /// `a = b = c = ...` uses no stack per link.
    fn assign(&mut self) -> PResult<Expr> {
        let mut targets = Vec::new();
        let mut value = self.ternary()?;
        while let Some(op) = self.peek().and_then(|t| assign_op(t.kind)) {
            self.pos += 1;
            targets.push((value, op, self.tree_depth));
            value = self.ternary()?;
        }
        while let Some((target, op, target_depth)) = targets.pop() {
            let span = target.span.to(value.span);
            let deepest = target_depth.max(self.tree_depth);
            let kind = ExprKind::Assign(op, Box::new(target), Box::new(value));
            value = self.node(kind, span, deepest)?;
        }
        Ok(value)
    }

    fn ternary(&mut self) -> PResult<Expr> {
        let cond = self.binary(0)?;
        if !self.eat(Punct::Question) {
            return Ok(cond);
        }
        let cond_depth = self.tree_depth;
        self.nested(|p| {
            let then = p.expr()?;
            let then_depth = p.tree_depth;
            p.expect(Punct::Colon)?;
            let otherwise = p.assign()?;
            let deepest = cond_depth.max(then_depth).max(p.tree_depth);
            let span = cond.span.to(otherwise.span);
            let kind = ExprKind::Ternary(Box::new(cond), Box::new(then), Box::new(otherwise));
            p.node(kind, span, deepest)
        })
    }

    fn binary(&mut self, min: u8) -> PResult<Expr> {
        let mut lhs = self.unary()?;
        loop {
            let lhs_depth = self.tree_depth;
            let Some(TokenKind::Punct(punct)) = self.peek().map(|t| t.kind) else {
                return Ok(lhs);
            };
            let Some((level, op)) = binary_op(punct) else {
                return Ok(lhs);
            };
            if level < min {
                return Ok(lhs);
            }
            self.pos += 1;
            let rhs = self.binary(level + 1)?;
            let deepest = lhs_depth.max(self.tree_depth);
            let span = lhs.span.to(rhs.span);
            lhs = self.node(
                ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)),
                span,
                deepest,
            )?;
        }
    }

    // The functions on the recursive path stay small and send the bulky work to
    // helpers that return first: a debug build gives every local its own stack slot.

    fn unary(&mut self) -> PResult<Expr> {
        self.nested(Self::unary_inner)
    }

    fn unary_inner(&mut self) -> PResult<Expr> {
        if let Some(op) = self.peek().and_then(|t| prefix_op(t.kind)) {
            return self.prefix(op);
        }
        if self.at_cast() {
            return self.cast();
        }
        let primary = if self.at(Punct::LParen) {
            self.paren()?
        } else {
            self.atom()?
        };
        self.postfix(primary)
    }

    fn prefix(&mut self, op: UnaryOp) -> PResult<Expr> {
        let start = self.here();
        self.pos += 1;
        let operand = self.unary()?;
        let span = start.to(operand.span);
        let depth = self.tree_depth;
        self.node(ExprKind::Unary(op, Box::new(operand)), span, depth)
    }

    fn cast(&mut self) -> PResult<Expr> {
        let start = self.here();
        self.pos += 1;
        let ty = self.type_ref()?;
        self.expect(Punct::RParen)?;
        let operand = self.unary()?;
        let span = start.to(operand.span);
        let depth = self.tree_depth;
        self.node(ExprKind::Cast(Box::new(ty), Box::new(operand)), span, depth)
    }

    fn paren(&mut self) -> PResult<Expr> {
        let start = self.here();
        self.pos += 1;
        let mut inner = self.expr()?;
        self.expect(Punct::RParen)?;
        inner.span = self.since(start);
        Ok(inner)
    }

    /// `(type)` comes next: a type name in parentheses.
    fn at_cast(&self) -> bool {
        if !self.at(Punct::LParen) || !self.is_type(self.peek_at(1)) {
            return false;
        }
        match self.peek_at(2) {
            Some(t) if t.is(Punct::RParen) => true,
            Some(t) if t.is(Punct::Lt) => {
                self.peek_at(1).is_some_and(|t| is_template_type(&t.text))
            }
            _ => false,
        }
    }

    fn postfix(&mut self, mut expr: Expr) -> PResult<Expr> {
        loop {
            let depth = self.tree_depth;
            if self.eat(Punct::Dot) {
                let name = self.word()?;
                let span = expr.span.to(name.span);
                expr = self.node(ExprKind::Member(Box::new(expr), name), span, depth)?;
            } else if self.eat(Punct::LBracket) {
                let index = self.expr()?;
                let deepest = depth.max(self.tree_depth);
                self.expect(Punct::RBracket)?;
                let span = self.since(expr.span);
                expr = self.node(
                    ExprKind::Index(Box::new(expr), Box::new(index)),
                    span,
                    deepest,
                )?;
            } else if self.at(Punct::LParen) {
                let args = self.call_args()?;
                let deepest = depth.max(self.tree_depth);
                let span = self.since(expr.span);
                expr = self.node(ExprKind::Call(Box::new(expr), args), span, deepest)?;
            } else if let Some(op) = self.peek().and_then(|t| postfix_op(t.kind)) {
                self.pos += 1;
                let span = self.since(expr.span);
                expr = self.node(ExprKind::Unary(op, Box::new(expr)), span, depth)?;
            } else {
                return Ok(expr);
            }
        }
    }

    /// Parses `(a, b, ...)`. Leaves the depth of the deepest argument in `tree_depth`.
    fn call_args(&mut self) -> PResult<Vec<Expr>> {
        self.expect(Punct::LParen)?;
        let mut args = Vec::new();
        let mut deepest = 0;
        if !self.eat(Punct::RParen) {
            loop {
                args.push(self.assign()?);
                deepest = deepest.max(self.tree_depth);
                if self.eat(Punct::RParen) {
                    break;
                }
                self.expect(Punct::Comma)?;
            }
        }
        self.tree_depth = deepest;
        Ok(args)
    }

    /// A literal, a name or a constructor.
    fn atom(&mut self) -> PResult<Expr> {
        let Some(t) = self.peek().cloned() else {
            return Err(self.expected("an expression"));
        };
        let span = t.site();
        let literal = match t.kind {
            TokenKind::Int => Some(Literal::Int(t.text.clone())),
            TokenKind::Float => Some(Literal::Float(t.text.clone())),
            TokenKind::Str => Some(Literal::Str(t.text.clone())),
            TokenKind::Ident if t.text == "true" || t.text == "false" => {
                Some(Literal::Bool(t.text == "true"))
            }
            _ => None,
        };
        if let Some(literal) = literal {
            self.pos += 1;
            return self.node(ExprKind::Literal(literal), span, 0);
        }
        if t.kind != TokenKind::Ident || is_keyword(&t.text) {
            return Err(self.expected("an expression"));
        }
        let next = self.peek_at(1);
        let constructor = next.is_some_and(|n| n.is(Punct::LParen))
            || (next.is_some_and(|n| n.is(Punct::Lt)) && is_template_type(&t.text));
        if constructor && self.is_type(Some(&t)) {
            let ty = self.type_ref()?;
            if !self.at(Punct::LParen) {
                return Err(self.expected("`(`"));
            }
            let args = self.call_args()?;
            let span = self.since(span);
            let depth = self.tree_depth;
            return self.node(ExprKind::Construct(Box::new(ty), args), span, depth);
        }
        self.pos += 1;
        self.node(ExprKind::Ident(t.text), span, 0)
    }
}

fn postfix_op(kind: TokenKind) -> Option<UnaryOp> {
    match kind {
        TokenKind::Punct(Punct::PlusPlus) => Some(UnaryOp::PostInc),
        TokenKind::Punct(Punct::MinusMinus) => Some(UnaryOp::PostDec),
        _ => None,
    }
}

fn prefix_op(kind: TokenKind) -> Option<UnaryOp> {
    let TokenKind::Punct(punct) = kind else {
        return None;
    };
    Some(match punct {
        Punct::Minus => UnaryOp::Neg,
        Punct::Plus => UnaryOp::Plus,
        Punct::Bang => UnaryOp::Not,
        Punct::Tilde => UnaryOp::BitNot,
        Punct::PlusPlus => UnaryOp::PreInc,
        Punct::MinusMinus => UnaryOp::PreDec,
        _ => return None,
    })
}

/// The operator of `=` (`None`) or a compound assignment such as `+=`.
fn assign_op(kind: TokenKind) -> Option<Option<BinaryOp>> {
    let TokenKind::Punct(punct) = kind else {
        return None;
    };
    Some(match punct {
        Punct::Eq => None,
        Punct::PlusEq => Some(BinaryOp::Add),
        Punct::MinusEq => Some(BinaryOp::Sub),
        Punct::StarEq => Some(BinaryOp::Mul),
        Punct::SlashEq => Some(BinaryOp::Div),
        Punct::PercentEq => Some(BinaryOp::Rem),
        Punct::ShlEq => Some(BinaryOp::Shl),
        Punct::ShrEq => Some(BinaryOp::Shr),
        Punct::AmpEq => Some(BinaryOp::BitAnd),
        Punct::PipeEq => Some(BinaryOp::BitOr),
        Punct::CaretEq => Some(BinaryOp::BitXor),
        _ => return None,
    })
}

fn binary_op(punct: Punct) -> Option<(u8, BinaryOp)> {
    Some(match punct {
        Punct::OrOr => (0, BinaryOp::Or),
        Punct::AndAnd => (1, BinaryOp::And),
        Punct::Pipe => (2, BinaryOp::BitOr),
        Punct::Caret => (3, BinaryOp::BitXor),
        Punct::Amp => (4, BinaryOp::BitAnd),
        Punct::EqEq => (5, BinaryOp::Eq),
        Punct::NotEq => (5, BinaryOp::Ne),
        Punct::Lt => (6, BinaryOp::Lt),
        Punct::Gt => (6, BinaryOp::Gt),
        Punct::LtEq => (6, BinaryOp::Le),
        Punct::GtEq => (6, BinaryOp::Ge),
        Punct::Shl => (7, BinaryOp::Shl),
        Punct::Shr => (7, BinaryOp::Shr),
        Punct::Plus => (8, BinaryOp::Add),
        Punct::Minus => (8, BinaryOp::Sub),
        Punct::Star => (9, BinaryOp::Mul),
        Punct::Slash => (9, BinaryOp::Div),
        Punct::Percent => (9, BinaryOp::Rem),
        _ => return None,
    })
}
