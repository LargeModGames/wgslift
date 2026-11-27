//! Function bodies: statements and expressions.

use std::collections::HashSet;
use std::fmt::Write;

use crate::ir::{
    Access, Arg, BinaryOp, Block, Expr, ExprKind, Literal, ParamDir, Place, Root, Sample, Space,
    Stmt, UnaryOp,
};
use crate::types::{SamplerDim, Scalar, Type};

use super::Printer;
use super::names::Namer;
use super::uniforms::Repr;

/// Expressions nested deeper than this are moved into `let` temporaries. naga's
/// WGSL parser stops at a recursion depth of 200.
const MAX_DEPTH: usize = 64;

/// The printer for one scope: a function, the entry wrapper or module scope.
pub(super) struct Body<'p, 'm> {
    p: &'p Printer<'m>,
    /// WGSL names of the locals.
    locals: Vec<String>,
    /// The types of the locals.
    local_types: Vec<Type>,
    /// Locals that are pointer parameters.
    pointer: Vec<bool>,
    namer: Namer,
    /// `let` lines that must come before the statement being printed.
    lets: Vec<String>,
    /// The keyword of those lines: module scope holds `const` lines instead.
    hoist: &'static str,
    out: String,
}

impl<'p, 'm> Body<'p, 'm> {
    /// A printer for module-scope initializers.
    pub(super) fn module_scope(p: &'p Printer<'m>) -> Body<'p, 'm> {
        let mut body = Body::detached(p);
        body.hoist = "const";
        body
    }

    /// The names this scope took, the module's included.
    pub(super) fn into_namer(self) -> Namer {
        self.namer
    }

    /// A printer for a scope without locals, such as the entry wrapper.
    pub(super) fn detached(p: &'p Printer<'m>) -> Body<'p, 'm> {
        Body {
            p,
            locals: Vec::new(),
            local_types: Vec::new(),
            pointer: Vec::new(),
            namer: p.names.module.clone(),
            lets: Vec::new(),
            hoist: "let",
            out: String::new(),
        }
    }

    /// Prints function `index`.
    pub(super) fn function(p: &'p Printer<'m>, index: usize) -> String {
        let func = &p.module.functions[index];
        let mut body = Body::detached(p);
        body.locals = func
            .locals
            .iter()
            .map(|l| body.namer.name(&l.name))
            .collect();
        body.local_types = func.locals.iter().map(|l| l.ty.clone()).collect();
        body.pointer = vec![false; func.locals.len()];
        let mut params = Vec::new();
        let mut copies = Vec::new();
        for param in &func.params {
            let local = &func.locals[param.local];
            let name = body.locals[param.local].clone();
            let ty = p.ty(&local.ty);
            match param.dir {
                ParamDir::In if local.written => {
                    let input = body.namer.name(&format!("{}_in", local.name));
                    copies.push(format!("var {name}: {ty} = {input};"));
                    params.push(format!("{input}: {ty}"));
                }
                ParamDir::In => params.push(format!("{name}: {ty}")),
                ParamDir::Out | ParamDir::InOut => {
                    body.pointer[param.local] = true;
                    params.push(format!("{name}: ptr<function, {ty}>"));
                }
            }
        }
        let ret = match func.ret {
            Type::Void => String::new(),
            ref ty => format!(" -> {}", p.ty(ty)),
        };
        let _ = writeln!(
            body.out,
            "fn {}({}){ret} {{",
            p.names.functions[index],
            params.join(", ")
        );
        for copy in copies {
            body.line(1, &copy);
        }
        let mut declared = HashSet::new();
        declared_locals(&func.body, &mut declared);
        declared.extend(func.params.iter().map(|p| p.local));
        for id in 0..func.locals.len() {
            if !declared.contains(&id) {
                let line = format!("var {}: {};", body.locals[id], p.ty(&func.locals[id].ty));
                body.line(1, &line);
            }
        }
        body.block(&func.body, 1);
        body.out.push_str("}\n");
        body.out
    }

    /// A fresh name in this scope.
    pub(super) fn fresh(&mut self, base: &str) -> String {
        self.namer.name(base)
    }

    /// The `let` lines that the expressions printed since the last call need.
    pub(super) fn take_lets(&mut self) -> Vec<String> {
        std::mem::take(&mut self.lets)
    }

    /// An expression at the top of a statement.
    pub(super) fn top(&mut self, e: &Expr) -> String {
        self.inner(e, 0)
    }

    fn line(&mut self, indent: usize, text: &str) {
        for _ in 0..indent {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    /// Writes the pending `let` lines.
    fn flush(&mut self, indent: usize) {
        for line in self.take_lets() {
            self.line(indent, &line);
        }
    }

    fn block(&mut self, block: &Block, indent: usize) {
        for stmt in block {
            self.stmt(stmt, indent);
        }
    }

    fn stmt(&mut self, stmt: &Stmt, indent: usize) {
        match stmt {
            Stmt::Local { id, init } => {
                let ty = self.p.ty(&self.local_types[*id]);
                let text = match init {
                    Some(e) => {
                        let value = self.top(e);
                        format!("var {}: {ty} = {value};", self.locals[*id])
                    }
                    None => format!("var {}: {ty};", self.locals[*id]),
                };
                self.flush(indent);
                self.line(indent, &text);
            }
            Stmt::Assign(place, value) => {
                let value = self.top(value);
                let place = self.place(place);
                self.flush(indent);
                self.line(indent, &format!("{place} = {value};"));
            }
            Stmt::Eval(e) => {
                let text = self.top(e);
                self.flush(indent);
                if matches!(e.kind, ExprKind::Call(..)) {
                    self.line(indent, &format!("{text};"));
                } else {
                    self.line(indent, &format!("_ = {text};"));
                }
            }
            Stmt::If(cond, accept, reject) => {
                let cond = self.top(cond);
                self.flush(indent);
                self.line(indent, &format!("if {cond} {{"));
                self.block(accept, indent + 1);
                if !reject.is_empty() {
                    self.line(indent, "} else {");
                    self.block(reject, indent + 1);
                }
                self.line(indent, "}");
            }
            Stmt::Loop {
                body,
                continuing,
                break_if,
            } => {
                self.line(indent, "loop {");
                if let Some((name, _)) = &self.p.names.loop_budget {
                    self.line(indent + 1, &format!("if {name} == 0u {{ break; }}"));
                    self.line(indent + 1, &format!("{name} -= 1u;"));
                }
                self.block(body, indent + 1);
                if !continuing.is_empty() || break_if.is_some() {
                    self.line(indent + 1, "continuing {");
                    self.block(continuing, indent + 2);
                    if let Some(cond) = break_if {
                        let cond = self.top(cond);
                        self.flush(indent + 2);
                        self.line(indent + 2, &format!("break if {cond};"));
                    }
                    self.line(indent + 1, "}");
                }
                self.line(indent, "}");
            }
            Stmt::Break => self.line(indent, "break;"),
            Stmt::Continue => self.line(indent, "continue;"),
            Stmt::Discard => self.line(indent, "discard;"),
            Stmt::Switch(selector, cases) => {
                let selector = self.top(selector);
                self.flush(indent);
                self.line(indent, &format!("switch {selector} {{"));
                let mut has_default = false;
                for (labels, body) in cases {
                    if labels.is_empty() {
                        continue;
                    }
                    let labels: Vec<String> = labels
                        .iter()
                        .map(|label| match label {
                            Some(value) => literal_text(Literal::Int(*value as i32)).0,
                            None => {
                                has_default = true;
                                "default".to_string()
                            }
                        })
                        .collect();
                    let head = if labels == ["default"] {
                        "default {".to_string()
                    } else {
                        format!("case {} {{", labels.join(", "))
                    };
                    self.line(indent + 1, &head);
                    self.block(body, indent + 2);
                    self.line(indent + 1, "}");
                }
                if !has_default {
                    self.line(indent + 1, "default {}");
                }
                self.line(indent, "}");
            }
            Stmt::Return(value) => {
                let text = match value {
                    Some(e) => format!("return {};", self.top(e)),
                    None => "return;".to_string(),
                };
                self.flush(indent);
                self.line(indent, &text);
            }
            Stmt::Block(block) => {
                self.line(indent, "{");
                self.block(block, indent + 1);
                self.line(indent, "}");
            }
        }
    }

    fn place(&mut self, place: &Place) -> String {
        let (mut text, mut ty) = match place.root {
            Root::Local(id) if self.pointer[id] => {
                if place.path.is_empty() {
                    return format!("*{}", self.locals[id]);
                }
                (
                    format!("(*{})", self.locals[id]),
                    self.local_types[id].clone(),
                )
            }
            Root::Local(id) => (self.locals[id].clone(), self.local_types[id].clone()),
            Root::Global(g) => (
                self.p.names.globals[g].clone(),
                self.p.module.globals[g].ty.clone(),
            ),
        };
        for access in &place.path {
            match access {
                Access::Field(f) => {
                    if let Type::Struct(s) = &ty {
                        let s = *s;
                        text = format!("{text}.{}", self.p.names.fields[s][*f]);
                        ty = self.p.module.structs[s].fields[*f].ty.clone();
                    }
                }
                Access::Index(index) => {
                    let index = self.inner(index, 1);
                    text = format!("{text}[{index}]");
                    ty = element(&ty);
                }
                Access::Component(c) => {
                    text = format!("{text}.{}", component(*c));
                    ty = element(&ty);
                }
            }
        }
        text
    }

    /// An expression without outer parentheses.
    fn inner(&mut self, e: &Expr, depth: usize) -> String {
        self.expr(e, depth).0
    }

    /// An expression that can be an operand or the base of a postfix.
    fn operand(&mut self, e: &Expr, depth: usize) -> String {
        match self.expr(e, depth) {
            (text, true) => text,
            (text, false) => format!("({text})"),
        }
    }

    fn list(&mut self, items: &[Expr], depth: usize) -> String {
        let parts: Vec<String> = items.iter().map(|e| self.inner(e, depth)).collect();
        parts.join(", ")
    }

    /// The text of an expression, and whether it is a primary expression.
    fn expr(&mut self, e: &Expr, depth: usize) -> (String, bool) {
        let leaf = matches!(e.kind, ExprKind::Literal(_) | ExprKind::Var(_));
        if depth > MAX_DEPTH && !leaf {
            let text = self.top(e);
            let name = self.namer.name("t");
            self.lets.push(format!("{} {name} = {text};", self.hoist));
            return (name, true);
        }
        if matches!(
            e.kind,
            ExprKind::Var(_) | ExprKind::Index(..) | ExprKind::Field(..)
        ) && let Some((text, repr)) = self.uniform_path(e, depth)
        {
            return self.p.uniforms.read(&repr, &text, &self.p.table());
        }
        let d = depth + 1;
        let p = self.p;
        match &e.kind {
            ExprKind::Literal(literal) => literal_text(*literal),
            ExprKind::Var(Root::Local(id)) if self.pointer[*id] => {
                (format!("(*{})", self.locals[*id]), true)
            }
            ExprKind::Var(Root::Local(id)) => (self.locals[*id].clone(), true),
            ExprKind::Var(Root::Global(g)) => (p.names.globals[*g].clone(), true),
            ExprKind::Field(base, f) => {
                let text = self.operand(base, d);
                let name = match base.ty {
                    Type::Struct(s) => p.names.fields[s][*f].as_str(),
                    _ => "x",
                };
                (format!("{text}.{name}"), true)
            }
            ExprKind::Index(base, index) => {
                let base = self.operand(base, d);
                let index = self.inner(index, d);
                (format!("{base}[{index}]"), true)
            }
            ExprKind::Swizzle(base, comps) => {
                let base = self.operand(base, d);
                let comps: String = comps.iter().map(|c| component(*c)).collect();
                (format!("{base}.{comps}"), true)
            }
            ExprKind::Unary(op, x) => {
                let op = match op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Not => "!",
                    UnaryOp::BitNot => "~",
                };
                (format!("{op}{}", self.operand(x, d)), false)
            }
            ExprKind::Binary(op, a, b) => {
                let a = self.operand(a, d);
                let b = self.operand(b, d);
                (format!("{a} {} {b}", binary_text(*op)), false)
            }
            ExprKind::Convert(x) | ExprKind::Splat(x) => {
                (format!("{}({})", p.ty(&e.ty), self.inner(x, d)), true)
            }
            ExprKind::Construct(parts) => {
                (format!("{}({})", p.ty(&e.ty), self.list(parts, d)), true)
            }
            ExprKind::Select {
                cond,
                accept,
                reject,
            } => {
                let reject = self.inner(reject, d);
                let accept = self.inner(accept, d);
                let cond = self.inner(cond, d);
                (format!("select({reject}, {accept}, {cond})"), true)
            }
            ExprKind::Call(f, args) => {
                let args: Vec<String> = args
                    .iter()
                    .map(|arg| match arg {
                        Arg::Value(e) => self.inner(e, d),
                        Arg::Ref(id) if self.pointer[*id] => self.locals[*id].clone(),
                        Arg::Ref(id) => format!("&{}", self.locals[*id]),
                    })
                    .collect();
                (
                    format!("{}({})", p.names.functions[*f], args.join(", ")),
                    true,
                )
            }
            ExprKind::Builtin(name, args) => (format!("{name}({})", self.list(args, d)), true),
            ExprKind::Sample {
                sampler,
                coords,
                kind,
            } => (self.sample(*sampler, coords, kind, d), true),
        }
    }

    fn sample(&mut self, sampler: usize, coords: &Expr, kind: &Sample, depth: usize) -> String {
        let texture = &self.p.names.globals[sampler];
        let smp = &self.p.names.samplers[sampler];
        let one_d = self.p.module.globals[sampler].ty == Type::Sampler(SamplerDim::D1);
        let coords = self.inner(coords, depth);
        match kind {
            Sample::Level(level) => {
                let level = self.inner(level, depth);
                format!("textureSampleLevel({texture}, {smp}, {coords}, {level})")
            }
            // A 1D texture has no mip levels, so a bias or gradients change nothing.
            Sample::Plain | Sample::Bias(_) | Sample::Grad(..) if one_d => {
                format!("textureSample({texture}, {smp}, {coords})")
            }
            Sample::Plain => format!("textureSample({texture}, {smp}, {coords})"),
            Sample::Bias(bias) => {
                let bias = self.inner(bias, depth);
                format!("textureSampleBias({texture}, {smp}, {coords}, {bias})")
            }
            Sample::Grad(dx, dy) => {
                let dx = self.inner(dx, depth);
                let dy = self.inner(dy, depth);
                format!("textureSampleGrad({texture}, {smp}, {coords}, {dx}, {dy})")
            }
        }
    }

    /// For a read of a uniform or a part of one: the stored value's text and form.
    fn uniform_path(&mut self, e: &Expr, depth: usize) -> Option<(String, Repr)> {
        match &e.kind {
            ExprKind::Var(Root::Global(g)) if self.p.module.globals[*g].space == Space::Uniform => {
                let repr = self.p.uniforms.globals.get(g)?.clone();
                let text = format!("{}.{}", self.p.names.uniform_var, self.p.names.globals[*g]);
                Some((text, repr))
            }
            ExprKind::Index(base, index) => {
                let (text, repr) = self.uniform_path(base, depth + 1)?;
                let index = self.inner(index, depth + 1);
                let (suffix, elem) = match repr {
                    Repr::Array { elem, wrap, .. } => {
                        (if wrap.is_some() { ".v" } else { "" }, *elem)
                    }
                    Repr::Value(ty) => ("", Repr::Value(element(&ty))),
                    Repr::Bool(_) => ("", Repr::Bool(1)),
                    other => ("", other),
                };
                Some((format!("{text}[{index}]{suffix}"), elem))
            }
            ExprKind::Field(base, f) => {
                let (text, repr) = self.uniform_path(base, depth + 1)?;
                let Repr::Struct(s) = repr else {
                    return None;
                };
                let text = format!("{text}.{}", self.p.names.fields[s][*f]);
                Some((text, self.p.uniforms.field(s, *f)))
            }
            _ => None,
        }
    }
}

/// The ids of the locals that a `Stmt::Local` in `block` declares.
fn declared_locals(block: &Block, out: &mut HashSet<usize>) {
    for stmt in block {
        match stmt {
            Stmt::Local { id, .. } => {
                out.insert(*id);
            }
            Stmt::If(_, a, b) => {
                declared_locals(a, out);
                declared_locals(b, out);
            }
            Stmt::Loop {
                body, continuing, ..
            } => {
                declared_locals(body, out);
                declared_locals(continuing, out);
            }
            Stmt::Switch(_, cases) => {
                for (_, body) in cases {
                    declared_locals(body, out);
                }
            }
            Stmt::Block(b) => declared_locals(b, out),
            _ => {}
        }
    }
}

/// The type of one element, row or component of `ty`.
fn element(ty: &Type) -> Type {
    match ty {
        Type::Array(elem, _) => (**elem).clone(),
        Type::Matrix { cols, .. } => Type::Vector(Scalar::Float, *cols),
        Type::Vector(s, _) => Type::Scalar(*s),
        other => other.clone(),
    }
}

fn component(c: u8) -> char {
    match c {
        0 => 'x',
        1 => 'y',
        2 => 'z',
        _ => 'w',
    }
}

fn binary_text(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul | BinaryOp::MatMul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Rem => "%",
        BinaryOp::Eq => "==",
        BinaryOp::Ne => "!=",
        BinaryOp::Lt => "<",
        BinaryOp::Le => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::Ge => ">=",
        BinaryOp::And => "&",
        BinaryOp::Or => "|",
        BinaryOp::Xor => "^",
        BinaryOp::Shl => "<<",
        BinaryOp::Shr => ">>",
    }
}

/// A literal with its type suffix, and whether it is a primary expression.
pub(super) fn literal_text(literal: Literal) -> (String, bool) {
    match literal {
        Literal::Bool(b) => (b.to_string(), true),
        Literal::Int(i32::MIN) => ("i32(-2147483648)".to_string(), true),
        Literal::Int(i) => (format!("{i}i"), i >= 0),
        Literal::Uint(u) => (format!("{u}u"), true),
        Literal::Float(f) => {
            // WGSL has no infinity or NaN literal: use the largest finite value or zero.
            let f = if f.is_nan() {
                0.0
            } else if f.is_infinite() {
                f.signum() * f32::MAX
            } else {
                f
            };
            let text = format!("{f:?}f");
            let primary = !text.starts_with('-');
            (text, primary)
        }
    }
}
