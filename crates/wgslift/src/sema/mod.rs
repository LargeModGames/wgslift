//! Semantic analysis: names, types and implicit conversions, from the syntax tree
//! to the typed IR.
//!
//! The passes run in this order: structs and typedefs, function signatures, a scan
//! for the globals that the shader writes, globals, function bodies, and last the
//! removal of everything the entry point does not reach.

mod expr;
mod intrinsics;
mod lowerer;
mod stmt;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use wgslift_syntax::ast::{self, Item, Modifier, TranslationUnit, TypeArg};
use wgslift_syntax::{Diagnostic, Span};

use crate::ir::{self, Expr, ExprKind, ParamDir, Root, Space, Stmt};
use crate::types::{Precision, SamplerDim, Scalar, Type};
use crate::{Dialect, Stage};

/// What the analysis needs from the options.
pub(crate) struct Settings<'a> {
    /// The stage of the entry point.
    pub stage: Stage,
    /// The HLSL name of the entry point.
    pub entry_point: &'a str,
    /// The dialect switches.
    pub dialect: &'a Dialect,
}

/// The names that a unit declares at the top level.
pub(crate) fn top_level_names(unit: &TranslationUnit) -> HashSet<&str> {
    let mut names = HashSet::new();
    for item in &unit.items {
        match item {
            Item::Variable(v) => names.extend(v.declarators.iter().map(|d| d.name.name.as_str())),
            Item::Function(f) => {
                names.insert(f.name.name.as_str());
            }
            Item::Struct(s) => {
                names.insert(s.name.name.as_str());
            }
            Item::Typedef(t) => names.extend(t.names.iter().map(|(n, _)| n.name.as_str())),
            Item::Buffer(b) => names.extend(
                b.members
                    .iter()
                    .flat_map(|m| m.declarators.iter().map(|d| d.name.name.as_str())),
            ),
        }
    }
    names
}

/// Checks `units` (the source first, then predeclared units) and lowers them to
/// the IR. Returns `None` after errors.
pub(crate) fn analyze(
    units: &[TranslationUnit],
    settings: &Settings<'_>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<ir::Module> {
    let mut a = Analyzer {
        dialect: settings.dialect,
        diagnostics,
        module: ir::Module::default(),
        types: HashMap::new(),
        globals: HashMap::new(),
        functions: HashMap::new(),
        signatures: Vec::new(),
        func: None,
        errors: false,
    };
    let _ = settings.stage;
    // Predeclared units come first: the source can use them anywhere, also in the
    // values of its globals.
    let items: Vec<&Item> = units[1..]
        .iter()
        .chain(&units[..1])
        .flat_map(|u| &u.items)
        .collect();
    for item in &items {
        match item {
            Item::Struct(s) => a.declare_struct(s),
            Item::Typedef(t) => a.declare_typedef(t),
            _ => {}
        }
    }
    let mut declared = Vec::new();
    for item in &items {
        if let Item::Function(f) = item {
            declared.push((a.declare_function(f), f));
        }
    }
    let written = a.written_names(&items);
    for item in &items {
        match item {
            Item::Variable(v) => a.declare_globals(v, &written, false),
            Item::Buffer(b) => b
                .members
                .iter()
                .for_each(|m| a.declare_globals(m, &written, true)),
            _ => {}
        }
    }
    for (index, f) in &declared {
        if let Some(index) = index {
            a.define_function(*index, f);
        }
    }
    let entry = a.find_entry(settings.entry_point);
    for global in &mut a.module.globals {
        if global.ty == Type::Sampler(SamplerDim::Any) {
            global.ty = Type::Sampler(SamplerDim::D2);
        }
    }
    let entry = entry?;
    if a.errors {
        return None;
    }
    a.module.entry = entry;
    Some(prune(a.module))
}

/// Marks that an error was reported; the caller recovers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reported;

/// The result of a step that reports its own errors.
pub(crate) type R<T> = Result<T, Reported>;

/// The signature of a user function.
#[derive(Debug, Clone)]
pub(crate) struct Signature {
    pub name: String,
    pub params: Vec<SigParam>,
    pub ret: Type,
    pub has_body: bool,
}

/// One parameter of a signature.
#[derive(Debug, Clone)]
pub(crate) struct SigParam {
    pub name: Option<String>,
    pub ty: Type,
    pub dir: ParamDir,
    pub semantic: Option<String>,
    pub default: Option<ast::Expr>,
}

/// The state of the function that is being analyzed.
pub(crate) struct FnState {
    pub locals: Vec<ir::Local>,
    /// Name scopes, innermost last.
    pub scopes: Vec<HashMap<String, usize>>,
    /// Statements to insert before the current statement.
    pub pre: Vec<Stmt>,
    pub ret: Type,
    /// The number of loops around the current statement.
    pub loops: usize,
    /// The number of `switch` statements around the current statement, inside the
    /// innermost loop.
    pub switches: usize,
}

pub(crate) struct Analyzer<'a> {
    pub dialect: &'a Dialect,
    pub diagnostics: &'a mut Vec<Diagnostic>,
    pub module: ir::Module,
    /// Struct and typedef names.
    pub types: HashMap<String, Type>,
    /// Global names, by index into `module.globals`.
    pub globals: HashMap<String, usize>,
    /// Function names, with the indexes of their overloads.
    pub functions: HashMap<String, Vec<usize>>,
    /// The signatures, by function index.
    pub signatures: Vec<Signature>,
    pub func: Option<FnState>,
    pub errors: bool,
}

impl Analyzer<'_> {
    pub fn error(&mut self, span: Span, message: impl Into<String>) -> Reported {
        self.errors = true;
        self.diagnostics.push(Diagnostic::error(span, message));
        Reported
    }

    pub fn warning(&mut self, span: Span, message: impl Into<String>) {
        self.diagnostics.push(Diagnostic::warning(span, message));
    }

    // Types.

    pub fn resolve_type(&mut self, t: &ast::TypeRef) -> R<Type> {
        let name = t.name.name.as_str();
        if let Some(ty) = self.types.get(name) {
            return Ok(ty.clone());
        }
        match name {
            "vector" | "matrix" => return self.template_type(t),
            _ => {}
        }
        match Type::from_hlsl_name(name) {
            Some((ty, precision)) => {
                if precision == Precision::Lowered {
                    self.warning(t.span, format!("`{name}` is 32-bit in WGSL"));
                }
                Ok(ty)
            }
            None => Err(self.error(t.span, format!("unknown type `{name}`"))),
        }
    }

    /// `vector<T, N>` and `matrix<T, R, C>`.
    fn template_type(&mut self, t: &ast::TypeRef) -> R<Type> {
        let mut scalar = Scalar::Float;
        let mut sizes = Vec::new();
        for (i, arg) in t.args.iter().enumerate() {
            match arg {
                TypeArg::Type(inner) if i == 0 => match self.resolve_type(inner)? {
                    Type::Scalar(s) => scalar = s,
                    _ => return Err(self.error(inner.span, "expected a scalar type")),
                },
                TypeArg::Expr(e) => sizes.push(self.const_int(e)?),
                TypeArg::Type(inner) => return Err(self.error(inner.span, "expected a size")),
            }
        }
        let dim = |n: i64| (1..=4).contains(&n).then_some(n as u8);
        let ty = match (t.name.name.as_str(), sizes.as_slice()) {
            ("vector", []) => Some(Type::Vector(Scalar::Float, 4)),
            ("vector", [n]) => dim(*n).map(|n| Type::vector(scalar, n)),
            ("matrix", []) => Some(Type::Matrix { rows: 4, cols: 4 }),
            ("matrix", [r, c]) if scalar == Scalar::Float => match (dim(*r), dim(*c)) {
                (Some(1), Some(n)) | (Some(n), Some(1)) => Some(Type::vector(scalar, n)),
                (Some(rows), Some(cols)) => Some(Type::Matrix { rows, cols }),
                _ => None,
            },
            _ => None,
        };
        ty.ok_or_else(|| self.error(t.span, "this vector or matrix type is not supported"))
    }

    /// Wraps `ty` in the array sizes of a declarator, innermost last.
    pub fn array_type(&mut self, mut ty: Type, dims: &[Option<ast::Expr>], span: Span) -> R<Type> {
        for dim in dims.iter().rev() {
            let Some(dim) = dim else {
                return Err(self.error(span, "an array needs a size"));
            };
            let n = self.const_int(dim)?;
            if !(1..=65536).contains(&n) {
                return Err(self.error(span, format!("the array size {n} is out of range")));
            }
            ty = Type::Array(Box::new(ty), n as u32);
        }
        Ok(ty)
    }

    /// Evaluates a constant integer: literals, integer constants and arithmetic.
    pub fn const_int(&mut self, e: &ast::Expr) -> R<i64> {
        let value = match &e.kind {
            ast::ExprKind::Literal(ast::Literal::Int(text)) => {
                let digits = text.trim_end_matches(['u', 'U', 'l', 'L']);
                let parsed = if let Some(hex) = digits
                    .strip_prefix("0x")
                    .or_else(|| digits.strip_prefix("0X"))
                {
                    i64::from_str_radix(hex, 16).ok()
                } else {
                    digits.parse().ok()
                };
                parsed.ok_or_else(|| self.error(e.span, "a bad integer"))?
            }
            ast::ExprKind::Unary(ast::UnaryOp::Neg, x) => -self.const_int(x)?,
            ast::ExprKind::Unary(ast::UnaryOp::Plus, x) => self.const_int(x)?,
            ast::ExprKind::Binary(op, a, b) => {
                let (a, b) = (self.const_int(a)?, self.const_int(b)?);
                match op {
                    ast::BinaryOp::Add => a.wrapping_add(b),
                    ast::BinaryOp::Sub => a.wrapping_sub(b),
                    ast::BinaryOp::Mul => a.wrapping_mul(b),
                    ast::BinaryOp::Div if b != 0 => a / b,
                    ast::BinaryOp::Rem if b != 0 => a % b,
                    ast::BinaryOp::Shl => a.wrapping_shl(b as u32),
                    ast::BinaryOp::Shr => a.wrapping_shr(b as u32),
                    _ => return Err(self.error(e.span, "expected a constant integer")),
                }
            }
            ast::ExprKind::Ident(name) => {
                let value = self
                    .globals
                    .get(name)
                    .map(|&g| &self.module.globals[g])
                    .filter(|g| g.space == Space::Const)
                    .and_then(|g| match g.init.as_ref().map(|e| &e.kind) {
                        Some(ExprKind::Literal(ir::Literal::Int(v))) => Some(i64::from(*v)),
                        Some(ExprKind::Literal(ir::Literal::Uint(v))) => Some(i64::from(*v)),
                        _ => None,
                    });
                value.ok_or_else(|| self.error(e.span, "expected a constant integer"))?
            }
            _ => return Err(self.error(e.span, "expected a constant integer")),
        };
        Ok(value)
    }

    // Declarations.

    fn declare_struct(&mut self, s: &ast::StructDef) {
        let mut fields = Vec::new();
        for decl in &s.fields {
            let Ok(base) = self.resolve_type(&decl.ty) else {
                continue;
            };
            for d in &decl.declarators {
                let Ok(ty) = self.array_type(base.clone(), &d.array, d.span) else {
                    continue;
                };
                fields.push(ir::Field {
                    name: d.name.name.clone(),
                    ty,
                    semantic: d.bindings.semantic.as_ref().map(|s| s.name.clone()),
                });
            }
        }
        let index = self.module.structs.len();
        self.module.structs.push(ir::StructDef {
            name: s.name.name.clone(),
            fields,
        });
        self.types.insert(s.name.name.clone(), Type::Struct(index));
    }

    fn declare_typedef(&mut self, t: &ast::Typedef) {
        let Ok(base) = self.resolve_type(&t.ty) else {
            return;
        };
        for (name, dims) in &t.names {
            if let Ok(ty) = self.array_type(base.clone(), dims, name.span) {
                self.types.insert(name.name.clone(), ty);
            }
        }
    }

    fn declare_function(&mut self, f: &ast::Function) -> Option<usize> {
        let ret = self.resolve_type(&f.return_type).ok()?;
        let mut params = Vec::new();
        for p in &f.params {
            let base = self.resolve_type(&p.ty).ok()?;
            let ty = self.array_type(base, &p.array, p.span).ok()?;
            let dir = if p.modifiers.contains(&Modifier::Inout)
                || (p.modifiers.contains(&Modifier::In) && p.modifiers.contains(&Modifier::Out))
            {
                ParamDir::InOut
            } else if p.modifiers.contains(&Modifier::Out) {
                ParamDir::Out
            } else {
                ParamDir::In
            };
            if p.modifiers.contains(&Modifier::Uniform) {
                self.error(
                    p.span,
                    "`uniform` parameters are not supported; declare a global",
                );
                return None;
            }
            if matches!(ty, Type::Sampler(_)) {
                self.error(p.span, "sampler parameters are not supported yet");
                return None;
            }
            params.push(SigParam {
                name: p.name.as_ref().map(|n| n.name.clone()),
                ty,
                dir,
                semantic: p.bindings.semantic.as_ref().map(|s| s.name.clone()),
                default: p.default.clone(),
            });
        }
        let name = f.name.name.clone();
        let index = self.signatures.len();
        self.signatures.push(Signature {
            name: name.clone(),
            params,
            ret,
            has_body: f.body.is_some(),
        });
        self.functions.entry(name).or_default().push(index);
        Some(index)
    }

    /// The names of variables that some function writes: assignment targets,
    /// incremented names and `out` arguments.
    fn written_names(&self, items: &[&Item]) -> HashSet<String> {
        let mut written = HashSet::new();
        for item in items {
            if let Item::Function(f) = item
                && let Some(body) = &f.body
            {
                for s in &body.stmts {
                    self.scan_stmt(s, &mut written);
                }
            }
        }
        written
    }

    fn scan_stmt(&self, s: &ast::Stmt, written: &mut HashSet<String>) {
        use ast::StmtKind as K;
        let mut scan = |e: &ast::Expr| self.scan_expr(e, written);
        match &s.kind {
            K::Var(v) => {
                for d in &v.declarators {
                    if let Some(ast::Initializer::Expr(e)) = &d.init {
                        scan(e);
                    }
                }
            }
            K::Expr(e) | K::Return(Some(e)) => scan(e),
            K::Block(b) => b.stmts.iter().for_each(|s| self.scan_stmt(s, written)),
            K::If {
                cond,
                then,
                otherwise,
            } => {
                scan(cond);
                self.scan_stmt(then, written);
                if let Some(o) = otherwise {
                    self.scan_stmt(o, written);
                }
            }
            K::For {
                init,
                cond,
                step,
                body,
            } => {
                if let Some(i) = init {
                    self.scan_stmt(i, written);
                }
                cond.iter()
                    .chain(step)
                    .for_each(|e| self.scan_expr(e, written));
                self.scan_stmt(body, written);
            }
            K::While { cond, body } | K::DoWhile { body, cond } => {
                scan(cond);
                self.scan_stmt(body, written);
            }
            K::Switch { value, cases } => {
                scan(value);
                for c in cases {
                    c.stmts.iter().for_each(|s| self.scan_stmt(s, written));
                }
            }
            _ => {}
        }
    }

    fn scan_expr(&self, e: &ast::Expr, written: &mut HashSet<String>) {
        use ast::ExprKind as K;
        let mut root = |e: &ast::Expr| {
            if let Some(name) = root_name(e) {
                written.insert(name.to_string());
            }
        };
        match &e.kind {
            K::Assign(_, target, value) => {
                root(target);
                self.scan_expr(target, written);
                self.scan_expr(value, written);
            }
            K::Unary(
                ast::UnaryOp::PreInc
                | ast::UnaryOp::PreDec
                | ast::UnaryOp::PostInc
                | ast::UnaryOp::PostDec,
                x,
            ) => {
                root(x);
                self.scan_expr(x, written);
            }
            K::Call(callee, args) => {
                if let K::Ident(name) = &callee.kind {
                    for (i, arg) in args.iter().enumerate() {
                        if self.may_write_arg(name, args.len(), i) {
                            root(arg);
                        }
                    }
                }
                self.scan_expr(callee, written);
                args.iter().for_each(|a| self.scan_expr(a, written));
            }
            K::Unary(_, x) | K::Cast(_, x) | K::Member(x, _) => self.scan_expr(x, written),
            K::Binary(_, a, b) | K::Index(a, b) => {
                self.scan_expr(a, written);
                self.scan_expr(b, written);
            }
            K::Ternary(c, a, b) => {
                for x in [c, a, b] {
                    self.scan_expr(x, written);
                }
            }
            K::Comma(list) | K::Construct(_, list) => {
                list.iter().for_each(|x| self.scan_expr(x, written))
            }
            K::Literal(_) | K::Ident(_) => {}
        }
    }

    /// Argument `index` of a call of `name` with `count` arguments can be written.
    fn may_write_arg(&self, name: &str, count: usize, index: usize) -> bool {
        if let Some(outs) = intrinsic_out_args(name, count) {
            return outs.contains(&index);
        }
        self.functions.get(name).is_some_and(|overloads| {
            overloads.iter().any(|&f| {
                self.signatures[f]
                    .params
                    .get(index)
                    .is_some_and(|p| p.dir != ParamDir::In)
            })
        })
    }

    fn declare_globals(&mut self, v: &ast::VarDecl, written: &HashSet<String>, in_buffer: bool) {
        let Ok(base) = self.resolve_type(&v.ty) else {
            return;
        };
        let is_static = v.modifiers.contains(&Modifier::Static);
        let is_const = v.modifiers.contains(&Modifier::Const);
        let is_uniform = in_buffer || v.modifiers.contains(&Modifier::Uniform);
        for d in &v.declarators {
            let Ok(ty) = self.array_type(base.clone(), &d.array, d.span) else {
                continue;
            };
            let name = d.name.name.clone();
            let is_written = written.contains(&name);
            let space = if matches!(ty, Type::Sampler(_)) {
                Space::Resource
            } else if is_uniform {
                Space::Uniform
            } else if is_const {
                Space::Const
            } else if is_static || d.init.is_some() || (self.dialect.writable_globals && is_written)
            {
                Space::Private
            } else {
                Space::Uniform
            };
            let mut global = ir::Global {
                name: name.clone(),
                ty: ty.clone(),
                space,
                init: None,
                entry_init: None,
                span: d.span,
            };
            match (&d.init, space) {
                (Some(ast::Initializer::SamplerState(..)), Space::Resource) | (None, _) => {}
                (Some(_), Space::Resource) => {
                    self.error(d.span, "a sampler cannot have a value");
                }
                (Some(_), Space::Uniform) => {
                    self.warning(
                        d.span,
                        "the value of a uniform is ignored; the host sets it",
                    );
                }
                (Some(init), _) => {
                    let Ok(value) = self.initializer(init, &ty, d.span) else {
                        continue;
                    };
                    if is_const_expr(&value, &self.module.globals) {
                        global.init = Some(value);
                    } else if space == Space::Const {
                        global.space = Space::Private;
                        global.entry_init = Some(value);
                    } else {
                        global.entry_init = Some(value);
                    }
                }
            }
            if global.space == Space::Const && global.init.is_none() {
                global.space = Space::Private;
            }
            if is_written && global.space == Space::Uniform && self.dialect.writable_globals {
                // Direct3D 9 lets a shader write a uniform: it writes its own copy.
                let uniform = self.module.globals.len();
                let ty = global.ty.clone();
                self.module.globals.push(global);
                let index = self.module.globals.len();
                self.module.globals.push(ir::Global {
                    name: name.clone(),
                    ty: ty.clone(),
                    space: Space::Private,
                    init: None,
                    entry_init: Some(Expr {
                        kind: ExprKind::Var(Root::Global(uniform)),
                        ty,
                    }),
                    span: d.span,
                });
                self.globals.insert(name, index);
                continue;
            }
            if is_written && matches!(global.space, Space::Uniform | Space::Const) {
                self.error(
                    d.span,
                    format!("`{name}` is written, but it is a uniform or a constant"),
                );
            }
            let index = self.module.globals.len();
            self.module.globals.push(global);
            self.globals.insert(name, index);
        }
    }

    // Functions.

    fn define_function(&mut self, index: usize, f: &ast::Function) {
        let Some(sig) = self.signatures.get(index).cloned() else {
            return;
        };
        let Some(body) = &f.body else {
            self.module.functions.push(placeholder(&sig, f.span));
            return;
        };
        let mut state = FnState {
            locals: Vec::new(),
            scopes: vec![HashMap::new()],
            pre: Vec::new(),
            ret: sig.ret.clone(),
            loops: 0,
            switches: 0,
        };
        let mut params = Vec::new();
        for (i, p) in sig.params.iter().enumerate() {
            let name = p.name.clone().unwrap_or_else(|| format!("_param{i}"));
            let local = state.locals.len();
            state.locals.push(ir::Local {
                name: name.clone(),
                ty: p.ty.clone(),
                written: false,
            });
            state.scopes[0].insert(name, local);
            params.push(ir::Param {
                local,
                dir: p.dir,
                semantic: p.semantic.clone(),
            });
        }
        self.func = Some(state);
        let mut stmts = self.block_stmts(&body.stmts);
        let state = self.func.take().expect("set above");
        if sig.ret != Type::Void && !matches!(stmts.last(), Some(Stmt::Return(_))) {
            stmts.push(Stmt::Return(Some(zero(&sig.ret))));
        }
        self.module.functions.push(ir::Function {
            name: sig.name.clone(),
            params,
            ret: sig.ret.clone(),
            ret_semantic: f.return_bindings.semantic.as_ref().map(|s| s.name.clone()),
            locals: state.locals,
            body: stmts,
            span: f.span,
        });
    }

    fn find_entry(&mut self, name: &str) -> Option<usize> {
        match self.functions.get(name).map(Vec::as_slice) {
            Some([index]) if self.signatures[*index].has_body => Some(*index),
            Some(_) => {
                let span = Span::new(wgslift_syntax::FileId(0), 0, 0);
                self.error(span, format!("the entry point `{name}` is overloaded"));
                None
            }
            None => {
                let span = Span::new(wgslift_syntax::FileId(0), 0, 0);
                self.error(span, format!("there is no entry point `{name}`"));
                None
            }
        }
    }

    // Locals.

    pub fn state(&mut self) -> &mut FnState {
        self.func.as_mut().expect("inside a function")
    }

    pub fn in_function(&self) -> bool {
        self.func.is_some()
    }

    /// Declares a local in the innermost scope and returns its index.
    pub fn new_local(&mut self, name: &str, ty: Type) -> usize {
        let state = self.state();
        let id = state.locals.len();
        state.locals.push(ir::Local {
            name: name.to_string(),
            ty,
            written: false,
        });
        if !name.is_empty() {
            state
                .scopes
                .last_mut()
                .expect("one scope at least")
                .insert(name.to_string(), id);
        }
        id
    }

    pub fn lookup_local(&self, name: &str) -> Option<usize> {
        let state = self.func.as_ref()?;
        state.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }
}

/// The name at the root of an assignment target: `a` in `a.b[1].x`.
fn root_name(e: &ast::Expr) -> Option<&str> {
    match &e.kind {
        ast::ExprKind::Ident(name) => Some(name),
        ast::ExprKind::Member(x, _) | ast::ExprKind::Index(x, _) => root_name(x),
        _ => None,
    }
}

/// The arguments of an intrinsic that it writes.
pub(crate) fn intrinsic_out_args(name: &str, count: usize) -> Option<&'static [usize]> {
    match (name, count) {
        ("modf", 2) => Some(&[1]),
        ("sincos", 3) => Some(&[1, 2]),
        _ => None,
    }
}

/// The zero value of a type: a WGSL constructor without arguments.
pub(crate) fn zero(ty: &Type) -> Expr {
    Expr {
        kind: ExprKind::Construct(Vec::new()),
        ty: ty.clone(),
    }
}

/// A function declared without a body: it keeps its signature, and a call of it
/// is an error later.
fn placeholder(sig: &Signature, span: Span) -> ir::Function {
    ir::Function {
        name: sig.name.clone(),
        params: Vec::new(),
        ret: sig.ret.clone(),
        ret_semantic: None,
        locals: Vec::new(),
        body: Vec::new(),
        span,
    }
}

/// The expression is a WGSL const-expression: it uses only literals, constants
/// and operations that WGSL can evaluate when it creates the shader.
pub(crate) fn is_const_expr(e: &Expr, globals: &[ir::Global]) -> bool {
    let all = |list: &[Expr]| list.iter().all(|x| is_const_expr(x, globals));
    match &e.kind {
        ExprKind::Literal(_) => true,
        ExprKind::Var(Root::Global(g)) => globals[*g].space == Space::Const,
        ExprKind::Var(Root::Local(_)) | ExprKind::Call(..) | ExprKind::Sample { .. } => false,
        ExprKind::Field(x, _)
        | ExprKind::Swizzle(x, _)
        | ExprKind::Unary(_, x)
        | ExprKind::Convert(x)
        | ExprKind::Splat(x) => is_const_expr(x, globals),
        ExprKind::Index(a, b) | ExprKind::Binary(_, a, b) => {
            is_const_expr(a, globals) && is_const_expr(b, globals)
        }
        ExprKind::Construct(list) => all(list),
        ExprKind::Select {
            cond,
            accept,
            reject,
        } => {
            is_const_expr(cond, globals)
                && is_const_expr(accept, globals)
                && is_const_expr(reject, globals)
        }
        // Derivatives need a running fragment; other math is const in WGSL.
        ExprKind::Builtin(name, args) => {
            !name.starts_with("dpd") && !name.starts_with("fwidth") && all(args)
        }
    }
}

/// Removes the globals and functions that the entry point does not reach, and
/// renumbers the rest.
fn prune(module: ir::Module) -> ir::Module {
    let mut used_functions = vec![false; module.functions.len()];
    let mut used_globals = vec![false; module.globals.len()];
    let mut stack = vec![module.entry];
    // Globals that are reached through other globals' values are found as well.
    while let Some(f) = stack.pop() {
        if std::mem::replace(&mut used_functions[f], true) {
            continue;
        }
        let function = &module.functions[f];
        record_uses(&function.body, &mut stack, &mut used_globals);
    }
    // Values of used globals can use other globals; repeat until nothing changes.
    loop {
        let mut changed = false;
        for g in 0..module.globals.len() {
            if !used_globals[g] {
                continue;
            }
            let global = &module.globals[g];
            for value in global.init.iter().chain(&global.entry_init) {
                visit_expr(value, &mut |e| {
                    if let ExprKind::Var(Root::Global(other)) = e.kind
                        && !used_globals[other]
                    {
                        used_globals[other] = true;
                        changed = true;
                    }
                });
            }
            for value in global.entry_init.iter() {
                visit_expr(value, &mut |e| {
                    if let ExprKind::Call(callee, _) = e.kind {
                        stack.push(callee);
                    }
                });
            }
        }
        while let Some(f) = stack.pop() {
            if std::mem::replace(&mut used_functions[f], true) {
                continue;
            }
            changed = true;
            record_uses(&module.functions[f].body, &mut stack, &mut used_globals);
        }
        if !changed {
            break;
        }
    }
    let renumber = |used: &[bool]| {
        let mut next = 0;
        used.iter()
            .map(|&u| {
                let index = next;
                next += usize::from(u);
                u.then_some(index)
            })
            .collect::<Vec<_>>()
    };
    let global_map = renumber(&used_globals);
    let function_map = renumber(&used_functions);
    let remap = |e: &mut Expr| match &mut e.kind {
        ExprKind::Var(Root::Global(g)) => *g = global_map[*g].expect("used"),
        ExprKind::Sample { sampler, .. } => *sampler = global_map[*sampler].expect("used"),
        ExprKind::Call(callee, _) => *callee = function_map[*callee].expect("used"),
        _ => {}
    };
    let ir::Module {
        structs,
        globals,
        functions,
        entry,
    } = module;
    let mut new_globals: Vec<ir::Global> = globals
        .into_iter()
        .zip(&used_globals)
        .filter(|(_, u)| **u)
        .map(|(g, _)| g)
        .collect();
    for g in &mut new_globals {
        for value in g.init.iter_mut().chain(g.entry_init.iter_mut()) {
            visit_expr_mut(value, &remap);
        }
    }
    let mut new_functions: Vec<ir::Function> = functions
        .into_iter()
        .zip(&used_functions)
        .filter(|(_, u)| **u)
        .map(|(f, _)| f)
        .collect();
    for f in &mut new_functions {
        visit_block_mut(&mut f.body, &remap, &|root: &mut Root| {
            if let Root::Global(g) = root {
                *g = global_map[*g].expect("used");
            }
        });
    }
    ir::Module {
        structs,
        globals: new_globals,
        functions: new_functions,
        entry: function_map[entry].expect("the entry is used"),
    }
}

/// Records the functions a body calls and the globals it uses.
fn record_uses(body: &[Stmt], calls: &mut Vec<usize>, used_globals: &mut [bool]) {
    let mut roots = Vec::new();
    visit_block(
        body,
        &mut |e| match &e.kind {
            ExprKind::Call(callee, _) => calls.push(*callee),
            ExprKind::Var(Root::Global(g)) => used_globals[*g] = true,
            ExprKind::Sample { sampler, .. } => used_globals[*sampler] = true,
            _ => {}
        },
        &mut |root| roots.push(root),
    );
    for root in roots {
        if let Root::Global(g) = root {
            used_globals[g] = true;
        }
    }
}
/// Calls `f` on every expression in `e`, outermost first.
pub(crate) fn visit_expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    match &e.kind {
        ExprKind::Literal(_) | ExprKind::Var(_) => {}
        ExprKind::Field(x, _)
        | ExprKind::Swizzle(x, _)
        | ExprKind::Unary(_, x)
        | ExprKind::Convert(x)
        | ExprKind::Splat(x) => visit_expr(x, f),
        ExprKind::Index(a, b) | ExprKind::Binary(_, a, b) => {
            visit_expr(a, f);
            visit_expr(b, f);
        }
        ExprKind::Construct(list) | ExprKind::Builtin(_, list) => {
            list.iter().for_each(|x| visit_expr(x, f))
        }
        ExprKind::Select {
            cond,
            accept,
            reject,
        } => {
            visit_expr(cond, f);
            visit_expr(accept, f);
            visit_expr(reject, f);
        }
        ExprKind::Call(_, args) => {
            for a in args {
                if let ir::Arg::Value(x) = a {
                    visit_expr(x, f);
                }
            }
        }
        ExprKind::Sample { coords, kind, .. } => {
            visit_expr(coords, f);
            match kind {
                ir::Sample::Plain => {}
                ir::Sample::Level(x) | ir::Sample::Bias(x) => visit_expr(x, f),
                ir::Sample::Grad(a, b) => {
                    visit_expr(a, f);
                    visit_expr(b, f);
                }
            }
        }
    }
}

fn visit_expr_mut(e: &mut Expr, f: &dyn Fn(&mut Expr)) {
    f(e);
    match &mut e.kind {
        ExprKind::Literal(_) | ExprKind::Var(_) => {}
        ExprKind::Field(x, _)
        | ExprKind::Swizzle(x, _)
        | ExprKind::Unary(_, x)
        | ExprKind::Convert(x)
        | ExprKind::Splat(x) => visit_expr_mut(x, f),
        ExprKind::Index(a, b) | ExprKind::Binary(_, a, b) => {
            visit_expr_mut(a, f);
            visit_expr_mut(b, f);
        }
        ExprKind::Construct(list) | ExprKind::Builtin(_, list) => {
            list.iter_mut().for_each(|x| visit_expr_mut(x, f))
        }
        ExprKind::Select {
            cond,
            accept,
            reject,
        } => {
            visit_expr_mut(cond, f);
            visit_expr_mut(accept, f);
            visit_expr_mut(reject, f);
        }
        ExprKind::Call(_, args) => {
            for a in args {
                if let ir::Arg::Value(x) = a {
                    visit_expr_mut(x, f);
                }
            }
        }
        ExprKind::Sample { coords, kind, .. } => {
            visit_expr_mut(coords, f);
            match kind {
                ir::Sample::Plain => {}
                ir::Sample::Level(x) | ir::Sample::Bias(x) => visit_expr_mut(x, f),
                ir::Sample::Grad(a, b) => {
                    visit_expr_mut(a, f);
                    visit_expr_mut(b, f);
                }
            }
        }
    }
}

/// Calls `f` on every expression and `root` on every assigned root in a block.
fn visit_block(block: &[Stmt], f: &mut dyn FnMut(&Expr), root: &mut dyn FnMut(Root)) {
    for s in block {
        match s {
            Stmt::Local { init, .. } => init.iter().for_each(|e| visit_expr(e, f)),
            Stmt::Assign(place, value) => {
                root(place.root);
                for a in &place.path {
                    if let ir::Access::Index(i) = a {
                        visit_expr(i, f);
                    }
                }
                visit_expr(value, f);
            }
            Stmt::Eval(e) => visit_expr(e, f),
            Stmt::If(c, a, b) => {
                visit_expr(c, f);
                visit_block(a, f, root);
                visit_block(b, f, root);
            }
            Stmt::Loop {
                body,
                continuing,
                break_if,
            } => {
                visit_block(body, f, root);
                visit_block(continuing, f, root);
                break_if.iter().for_each(|e| visit_expr(e, f));
            }
            Stmt::Switch(v, cases) => {
                visit_expr(v, f);
                cases.iter().for_each(|(_, b)| visit_block(b, f, root));
            }
            Stmt::Return(v) => v.iter().for_each(|e| visit_expr(e, f)),
            Stmt::Block(b) => visit_block(b, f, root),
            Stmt::Break | Stmt::Continue | Stmt::Discard => {}
        }
    }
}

fn visit_block_mut(block: &mut [Stmt], f: &dyn Fn(&mut Expr), root: &dyn Fn(&mut Root)) {
    for s in block {
        match s {
            Stmt::Local { init, .. } => init.iter_mut().for_each(|e| visit_expr_mut(e, f)),
            Stmt::Assign(place, value) => {
                root(&mut place.root);
                for a in &mut place.path {
                    if let ir::Access::Index(i) = a {
                        visit_expr_mut(i, f);
                    }
                }
                visit_expr_mut(value, f);
            }
            Stmt::Eval(e) => visit_expr_mut(e, f),
            Stmt::If(c, a, b) => {
                visit_expr_mut(c, f);
                visit_block_mut(a, f, root);
                visit_block_mut(b, f, root);
            }
            Stmt::Loop {
                body,
                continuing,
                break_if,
            } => {
                visit_block_mut(body, f, root);
                visit_block_mut(continuing, f, root);
                break_if.iter_mut().for_each(|e| visit_expr_mut(e, f));
            }
            Stmt::Switch(v, cases) => {
                visit_expr_mut(v, f);
                cases
                    .iter_mut()
                    .for_each(|(_, b)| visit_block_mut(b, f, root));
            }
            Stmt::Return(v) => v.iter_mut().for_each(|e| visit_expr_mut(e, f)),
            Stmt::Block(b) => visit_block_mut(b, f, root),
            Stmt::Break | Stmt::Continue | Stmt::Discard => {}
        }
    }
}
