//! Statements and initializers.

use std::collections::HashMap;

use wgslift_syntax::Span;
use wgslift_syntax::ast::{self, Modifier, StmtKind as K};

use super::{Analyzer, R, is_const_expr};
use crate::ir::{self, Block, Expr, ExprKind, Root, Space, Stmt, UnaryOp};
use crate::types::Type;

impl Analyzer<'_> {
    /// Lowers statements in a new scope.
    pub fn block_stmts(&mut self, stmts: &[ast::Stmt]) -> Block {
        self.state().scopes.push(HashMap::new());
        let mut out = Vec::new();
        for s in stmts {
            self.stmt(s, &mut out);
        }
        self.state().scopes.pop();
        out
    }

    /// One statement as a block of its own.
    fn sub_block(&mut self, s: &ast::Stmt) -> Block {
        match &s.kind {
            K::Block(b) => self.block_stmts(&b.stmts),
            _ => self.block_stmts(std::slice::from_ref(s)),
        }
    }

    fn take_pre(&mut self) -> Vec<Stmt> {
        std::mem::take(&mut self.state().pre)
    }

    /// Lowers one statement into `out`. After an error the statement is left out.
    fn stmt(&mut self, s: &ast::Stmt, out: &mut Block) {
        let before = self.take_pre();
        debug_assert!(
            before.is_empty(),
            "statements start with no pending statements"
        );
        if self.stmt_inner(s, out).is_err() {
            self.take_pre();
        }
    }

    fn stmt_inner(&mut self, s: &ast::Stmt, out: &mut Block) -> R<()> {
        match &s.kind {
            K::Empty => {}
            K::Var(v) => self.local_decl(v, out)?,
            K::Expr(e) => {
                self.effect(e)?;
                out.extend(self.take_pre());
            }
            K::Block(b) => {
                let block = self.block_stmts(&b.stmts);
                out.push(Stmt::Block(block));
            }
            K::If {
                cond,
                then,
                otherwise,
            } => {
                let c = self.condition(cond)?;
                out.extend(self.take_pre());
                let then = self.sub_block(then);
                let otherwise = otherwise
                    .as_deref()
                    .map(|o| self.sub_block(o))
                    .unwrap_or_default();
                out.push(Stmt::If(c, then, otherwise));
            }
            K::For {
                init,
                cond,
                step,
                body,
            } => {
                let own_scope = !self.dialect.for_init_outer_scope;
                if own_scope {
                    self.state().scopes.push(HashMap::new());
                }
                let mut outer = Vec::new();
                if let Some(init) = init {
                    self.stmt(init, &mut outer);
                }
                let result = self.loop_stmt(cond.as_ref(), step.as_ref(), body, s.span);
                if own_scope {
                    self.state().scopes.pop();
                }
                outer.push(result?);
                if own_scope {
                    out.push(Stmt::Block(outer));
                } else {
                    out.extend(outer);
                }
            }
            K::While { cond, body } => {
                let l = self.loop_stmt(Some(cond), None, body, s.span)?;
                out.push(l);
            }
            K::DoWhile { body, cond } => {
                self.state().loops += 1;
                let saved_switches = std::mem::take(&mut self.state().switches);
                let body = self.sub_block(body);
                let c = self.condition(cond);
                self.state().switches = saved_switches;
                self.state().loops -= 1;
                let c = c?;
                let continuing = self.take_pre();
                out.push(Stmt::Loop {
                    body,
                    continuing,
                    break_if: Some(not(c)),
                });
            }
            K::Switch { value, cases } => self.switch(value, cases, s.span, out)?,
            K::Break => {
                let state = self.state();
                if state.loops == 0 && state.switches == 0 {
                    return Err(self.error(s.span, "`break` outside a loop or `switch`"));
                }
                out.push(Stmt::Break);
            }
            K::Continue => {
                if self.state().loops == 0 {
                    return Err(self.error(s.span, "`continue` outside a loop"));
                }
                out.push(Stmt::Continue);
            }
            K::Discard => out.push(Stmt::Discard),
            K::Return(value) => {
                let ret = self.state().ret.clone();
                let value = match (value, &ret) {
                    (None, Type::Void) => None,
                    (None, _) => {
                        return Err(self.error(s.span, "this function must return a value"));
                    }
                    (Some(_), Type::Void) => {
                        return Err(self.error(s.span, "this function returns nothing"));
                    }
                    (Some(e), _) => {
                        let v = self.expr(e)?;
                        Some(self.convert(v, &ret, e.span, false))
                    }
                };
                out.extend(self.take_pre());
                out.push(Stmt::Return(value));
            }
        }
        Ok(())
    }

    /// A loop: the condition, with its own statements, runs at the top of each
    /// pass, and the step in the `continuing` block.
    fn loop_stmt(
        &mut self,
        cond: Option<&ast::Expr>,
        step: Option<&ast::Expr>,
        body: &ast::Stmt,
        span: Span,
    ) -> R<Stmt> {
        let _ = span;
        let mut lowered = Vec::new();
        if let Some(c) = cond {
            let c = self.condition(c)?;
            lowered.extend(self.take_pre());
            lowered.push(Stmt::If(not(c), vec![Stmt::Break], Vec::new()));
        }
        self.state().loops += 1;
        let saved_switches = std::mem::take(&mut self.state().switches);
        lowered.extend(self.sub_block(body));
        let continuing = match step {
            Some(step) => self.effect(step).map(|_| self.take_pre()),
            None => Ok(Vec::new()),
        };
        self.state().switches = saved_switches;
        self.state().loops -= 1;
        Ok(Stmt::Loop {
            body: lowered,
            continuing: continuing?,
            break_if: None,
        })
    }

    fn switch(
        &mut self,
        value: &ast::Expr,
        cases: &[ast::SwitchCase],
        span: Span,
        out: &mut Block,
    ) -> R<()> {
        let v = self.expr(value)?;
        if v.ty.size() != Some(1) {
            return Err(self.error(value.span, "`switch` needs a number"));
        }
        let v = self.convert(v, &Type::INT, value.span, true);
        out.extend(self.take_pre());
        let mut lowered: Vec<(Vec<Option<i64>>, Block)> = Vec::new();
        let mut carried = Vec::new();
        self.state().switches += 1;
        for (i, case) in cases.iter().enumerate() {
            let mut labels = std::mem::take(&mut carried);
            for label in &case.labels {
                match label {
                    Some(e) => match self.const_int(e) {
                        Ok(n) => labels.push(Some(n)),
                        Err(_) => continue,
                    },
                    None => labels.push(None),
                }
            }
            let block = self.block_stmts(&case.stmts);
            let last = i + 1 == cases.len();
            if block.is_empty() && !last {
                // An empty case falls into the next one: they share a body.
                carried = labels;
                continue;
            }
            let ends = matches!(
                block.last(),
                Some(Stmt::Break | Stmt::Return(_) | Stmt::Discard | Stmt::Continue)
            );
            if !ends && !last {
                self.error(
                    case.span,
                    "a `case` that falls into the next one is not supported",
                );
            }
            lowered.push((labels, block));
        }
        self.state().switches -= 1;
        let _ = span;
        out.push(Stmt::Switch(v, lowered));
        Ok(())
    }

    fn local_decl(&mut self, v: &ast::VarDecl, out: &mut Block) -> R<()> {
        let base = self.resolve_type(&v.ty)?;
        if matches!(base, Type::Sampler(_)) {
            return Err(self.error(v.span, "local samplers are not supported"));
        }
        // A `static` local keeps its value between calls: it is a private global.
        let is_static =
            v.modifiers.contains(&Modifier::Static) && !v.modifiers.contains(&Modifier::Const);
        for d in &v.declarators {
            let ty = self.array_type(base.clone(), &d.array, d.span)?;
            let init = match &d.init {
                Some(init) => Some(self.initializer(init, &ty, d.span)?),
                None => None,
            };
            if is_static {
                if init
                    .as_ref()
                    .is_some_and(|v| !is_const_expr(v, &self.module.globals))
                {
                    return Err(
                        self.error(d.span, "the value of a `static` local must be a constant")
                    );
                }
                let g = self.module.globals.len();
                self.module.globals.push(ir::Global {
                    name: d.name.name.clone(),
                    ty,
                    space: Space::Private,
                    init,
                    entry_init: None,
                    span: d.span,
                });
                let scope = self.state().scopes.last_mut().expect("one scope at least");
                scope.insert(d.name.name.clone(), Root::Global(g));
                continue;
            }
            let id = self.new_local(&d.name.name, ty);
            out.extend(self.take_pre());
            out.push(Stmt::Local { id, init });
        }
        Ok(())
    }

    /// The value of an initializer of type `ty`. A `{}` list is flattened into
    /// scalars and rebuilt in the shape of `ty`, as HLSL does.
    pub fn initializer(&mut self, init: &ast::Initializer, ty: &Type, span: Span) -> R<Expr> {
        match init {
            ast::Initializer::Expr(e) => {
                let v = self.expr(e)?;
                Ok(self.convert(v, ty, e.span, false))
            }
            ast::Initializer::List(items, list_span) => {
                let mut scalars = Vec::new();
                self.flatten_list(items, &mut scalars)?;
                let total = scalars.len();
                let mut it = scalars.into_iter();
                let value = self.build(ty, &mut it, *list_span)?;
                let left = it.count();
                if left > 0 {
                    return Err(
                        self.error(*list_span, format!("{left} values too many, of {total}"))
                    );
                }
                Ok(value)
            }
            ast::Initializer::SamplerState(..) => {
                Err(self.error(span, "only a sampler takes `sampler_state`"))
            }
        }
    }

    fn flatten_list(&mut self, items: &[ast::Initializer], out: &mut Vec<Expr>) -> R<()> {
        for item in items {
            match item {
                ast::Initializer::List(inner, _) => self.flatten_list(inner, out)?,
                ast::Initializer::Expr(e) => {
                    let v = self.expr(e)?;
                    if !v.ty.is_numeric() {
                        return Err(self.error(e.span, "a `{}` list takes numbers only"));
                    }
                    out.extend(self.flatten(v, e.span));
                }
                ast::Initializer::SamplerState(_, span) => {
                    return Err(self.error(*span, "`sampler_state` is not a value"));
                }
            }
        }
        Ok(())
    }

    fn build(&mut self, ty: &Type, it: &mut std::vec::IntoIter<Expr>, span: Span) -> R<Expr> {
        let mut next = |a: &mut Self, scalar: &Type| match it.next() {
            Some(v) => Ok(a.convert(v, scalar, span, true)),
            None => Err(a.error(span, "too few values in the `{}` list")),
        };
        let parts = match ty {
            Type::Scalar(_) => return next(self, ty),
            Type::Vector(s, n) => (0..*n)
                .map(|_| next(self, &Type::Scalar(*s)))
                .collect::<R<Vec<_>>>()?,
            Type::Matrix { rows, cols } => (0..rows * cols)
                .map(|_| next(self, &Type::FLOAT))
                .collect::<R<Vec<_>>>()?,
            Type::Array(element, n) => (0..*n)
                .map(|_| self.build(element, it, span))
                .collect::<R<Vec<_>>>()?,
            Type::Struct(s) => {
                let fields: Vec<Type> = self.module.structs[*s]
                    .fields
                    .iter()
                    .map(|f| f.ty.clone())
                    .collect();
                fields
                    .iter()
                    .map(|f| self.build(f, it, span))
                    .collect::<R<Vec<_>>>()?
            }
            Type::Void | Type::Sampler(_) => {
                return Err(self.error(span, "this type has no `{}` value"));
            }
        };
        Ok(Expr {
            kind: ExprKind::Construct(parts),
            ty: ty.clone(),
        })
    }
}

/// `!c` for a bool condition.
fn not(c: Expr) -> Expr {
    Expr {
        kind: ExprKind::Unary(UnaryOp::Not, Box::new(c)),
        ty: Type::BOOL,
    }
}
