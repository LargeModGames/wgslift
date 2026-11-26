//! The syntax tree. Every node has the span of the source text it came from.

use crate::span::Span;

/// A name with its place in the source.
#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    /// The name.
    pub name: String,
    /// Where it is.
    pub span: Span,
}

/// One parsed file after preprocessing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranslationUnit {
    /// The top-level items in source order.
    pub items: Vec<Item>,
}

/// A top-level item.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    /// Global variables, constants, textures and samplers.
    Variable(VarDecl),
    /// A function definition or declaration.
    Function(Function),
    /// A struct definition.
    Struct(StructDef),
    /// `typedef T name;`
    Typedef(Typedef),
    /// A `cbuffer` or `tbuffer` block.
    Buffer(BufferBlock),
}

/// A type as written, for example `float4`, `vector<float, 3>` or `MyStruct`.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeRef {
    /// The type's name.
    pub name: Ident,
    /// Arguments in `<>`, for example `float` and `3` in `vector<float, 3>`.
    pub args: Vec<TypeArg>,
    /// The whole type.
    pub span: Span,
}

/// One argument of a template type.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeArg {
    /// A type argument.
    Type(TypeRef),
    /// A constant argument.
    Expr(Expr),
}

/// A keyword before a declaration that changes its storage, layout or use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Modifier {
    Static,
    Uniform,
    Extern,
    Const,
    Shared,
    Volatile,
    Groupshared,
    Precise,
    Inline,
    RowMajor,
    ColumnMajor,
    Nointerpolation,
    Linear,
    Centroid,
    Noperspective,
    Sample,
    In,
    Out,
    Inout,
    Snorm,
    Unorm,
}

impl Modifier {
    /// The modifier with this keyword.
    pub fn from_word(word: &str) -> Option<Modifier> {
        Some(match word {
            "static" => Modifier::Static,
            "uniform" => Modifier::Uniform,
            "extern" => Modifier::Extern,
            "const" => Modifier::Const,
            "shared" => Modifier::Shared,
            "volatile" => Modifier::Volatile,
            "groupshared" => Modifier::Groupshared,
            "precise" => Modifier::Precise,
            "inline" => Modifier::Inline,
            "row_major" => Modifier::RowMajor,
            "column_major" => Modifier::ColumnMajor,
            "nointerpolation" => Modifier::Nointerpolation,
            "linear" => Modifier::Linear,
            "centroid" => Modifier::Centroid,
            "noperspective" => Modifier::Noperspective,
            "sample" => Modifier::Sample,
            "in" => Modifier::In,
            "out" => Modifier::Out,
            "inout" => Modifier::Inout,
            "snorm" => Modifier::Snorm,
            "unorm" => Modifier::Unorm,
            _ => return None,
        })
    }
}

/// `register(c0)` or `register(ps_3_0, s1)` after a declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct Register {
    /// The shader profile, if one is given.
    pub profile: Option<Ident>,
    /// The register, for example `c0` or `s1`.
    pub slot: Ident,
    /// The whole `register(...)`.
    pub span: Span,
}

/// `packoffset(c0.y)` after a member of a constant buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct PackOffset {
    /// The register, for example `c0`.
    pub slot: Ident,
    /// The component, for example `y`.
    pub component: Option<Ident>,
    /// The whole `packoffset(...)`.
    pub span: Span,
}

/// The things that can follow `:` after a declarator, a parameter or a function.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bindings {
    /// A semantic, for example `TEXCOORD0` or `SV_Target`.
    pub semantic: Option<Ident>,
    /// A `register(...)` binding.
    pub register: Option<Register>,
    /// A `packoffset(...)` binding.
    pub packoffset: Option<PackOffset>,
}

/// One declaration statement: modifiers, a type and one or more names.
#[derive(Debug, Clone, PartialEq)]
pub struct VarDecl {
    /// Keywords before the type.
    pub modifiers: Vec<Modifier>,
    /// The type of all declarators.
    pub ty: TypeRef,
    /// The declared names.
    pub declarators: Vec<Declarator>,
    /// The whole declaration.
    pub span: Span,
}

/// One declared name, with its array sizes, bindings and initializer.
#[derive(Debug, Clone, PartialEq)]
pub struct Declarator {
    /// The name.
    pub name: Ident,
    /// One entry for each `[...]`; `None` for `[]`.
    pub array: Vec<Option<Expr>>,
    /// Semantic, register and packoffset.
    pub bindings: Bindings,
    /// The span of an annotation block `<...>`, which is kept as text.
    pub annotations: Option<Span>,
    /// The value after `=`.
    pub init: Option<Initializer>,
    /// The whole declarator.
    pub span: Span,
}

/// The value of a declaration.
#[derive(Debug, Clone, PartialEq)]
pub enum Initializer {
    /// `= expr`
    Expr(Expr),
    /// `= { a, b, ... }`
    List(Vec<Initializer>, Span),
    /// `= sampler_state { Name = value; ... }`
    SamplerState(Vec<StateAssignment>, Span),
}

/// One `Name = value;` line in a `sampler_state` block.
#[derive(Debug, Clone, PartialEq)]
pub struct StateAssignment {
    /// The state, for example `MinFilter`.
    pub name: Ident,
    /// The value. A texture reference `<tex>` or `(tex)` is an identifier expression.
    pub value: Expr,
    /// The whole line.
    pub span: Span,
}

/// An attribute in `[]`, for example `[unroll]` or `[numthreads(8, 8, 1)]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// The attribute's name.
    pub name: Ident,
    /// Its arguments, if it has parentheses.
    pub args: Vec<Expr>,
    /// The whole attribute, with the brackets.
    pub span: Span,
}

/// A function.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    /// Attributes before the function.
    pub attributes: Vec<Attribute>,
    /// Keywords before the return type.
    pub modifiers: Vec<Modifier>,
    /// The return type.
    pub return_type: TypeRef,
    /// The function's name.
    pub name: Ident,
    /// The parameters.
    pub params: Vec<Param>,
    /// The semantic or register of the return value.
    pub return_bindings: Bindings,
    /// The body; `None` for a declaration that ends in `;`.
    pub body: Option<Block>,
    /// The whole function.
    pub span: Span,
}

/// A function parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    /// Keywords such as `in`, `out`, `inout` and `uniform`.
    pub modifiers: Vec<Modifier>,
    /// The type.
    pub ty: TypeRef,
    /// The name. HLSL lets a declaration leave it out.
    pub name: Option<Ident>,
    /// One entry for each `[...]`.
    pub array: Vec<Option<Expr>>,
    /// Semantic and register.
    pub bindings: Bindings,
    /// The default value after `=`.
    pub default: Option<Expr>,
    /// The whole parameter.
    pub span: Span,
}

/// `struct Name { fields };`
#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    /// The struct's name.
    pub name: Ident,
    /// The fields, as declarations.
    pub fields: Vec<VarDecl>,
    /// The whole definition.
    pub span: Span,
}

/// `typedef T name;`
#[derive(Debug, Clone, PartialEq)]
pub struct Typedef {
    /// Modifiers such as `const`.
    pub modifiers: Vec<Modifier>,
    /// The type that gets the new name.
    pub ty: TypeRef,
    /// The new names, with their array sizes.
    pub names: Vec<(Ident, Vec<Option<Expr>>)>,
    /// The whole typedef.
    pub span: Span,
}

/// The kind of a buffer block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferKind {
    /// `cbuffer`
    Constant,
    /// `tbuffer`
    Texture,
}

/// `cbuffer Name : register(b0) { members };`
#[derive(Debug, Clone, PartialEq)]
pub struct BufferBlock {
    /// `cbuffer` or `tbuffer`.
    pub kind: BufferKind,
    /// The block's name.
    pub name: Ident,
    /// The register binding.
    pub bindings: Bindings,
    /// The members.
    pub members: Vec<VarDecl>,
    /// The whole block.
    pub span: Span,
}

/// A block of statements in `{}`.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    /// The statements.
    pub stmts: Vec<Stmt>,
    /// The block, with the braces.
    pub span: Span,
}

/// A statement, with the attributes written before it.
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    /// Attributes such as `[unroll]` or `[branch]`.
    pub attributes: Vec<Attribute>,
    /// What the statement does.
    pub kind: StmtKind,
    /// The whole statement.
    pub span: Span,
}

/// The kinds of statements.
#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    /// `;`
    Empty,
    /// A local declaration.
    Var(VarDecl),
    /// An expression and `;`.
    Expr(Expr),
    /// `{ ... }`
    Block(Block),
    /// `if (cond) then else otherwise`
    If {
        /// The condition.
        cond: Expr,
        /// The statement for a true condition.
        then: Box<Stmt>,
        /// The statement after `else`.
        otherwise: Option<Box<Stmt>>,
    },
    /// `for (init; cond; step) body`
    For {
        /// The first part: a declaration or an expression.
        init: Option<Box<Stmt>>,
        /// The condition.
        cond: Option<Expr>,
        /// The expression after each pass.
        step: Option<Expr>,
        /// The loop body.
        body: Box<Stmt>,
    },
    /// `while (cond) body`
    While {
        /// The condition.
        cond: Expr,
        /// The loop body.
        body: Box<Stmt>,
    },
    /// `do body while (cond);`
    DoWhile {
        /// The loop body.
        body: Box<Stmt>,
        /// The condition.
        cond: Expr,
    },
    /// `switch (value) { cases }`
    Switch {
        /// The value to compare.
        value: Expr,
        /// The cases in order.
        cases: Vec<SwitchCase>,
    },
    /// `break;`
    Break,
    /// `continue;`
    Continue,
    /// `discard;`
    Discard,
    /// `return;` or `return value;`
    Return(Option<Expr>),
}

/// One `case`/`default` label group and its statements.
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchCase {
    /// The labels; `None` is `default`.
    pub labels: Vec<Option<Expr>>,
    /// The statements after the labels.
    pub stmts: Vec<Stmt>,
    /// The labels and statements.
    pub span: Span,
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    /// What the expression is.
    pub kind: ExprKind,
    /// The whole expression.
    pub span: Span,
}

/// A literal value. The text keeps the suffix, for example `1.5f` or `2u`.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// An integer literal.
    Int(String),
    /// A floating-point literal.
    Float(String),
    /// `true` or `false`.
    Bool(bool),
    /// A string literal, with its quotes.
    Str(String),
}

/// Prefix operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum UnaryOp {
    Neg,
    Plus,
    Not,
    BitNot,
    PreInc,
    PreDec,
    PostInc,
    PostDec,
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum BinaryOp {
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Shl,
    Shr,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    BitAnd,
    BitXor,
    BitOr,
    And,
    Or,
}

/// The kinds of expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    /// A literal.
    Literal(Literal),
    /// A name.
    Ident(String),
    /// A prefix or postfix operator.
    Unary(UnaryOp, Box<Expr>),
    /// A binary operator.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// `target = value`, or a compound form such as `+=` with its operator.
    Assign(Option<BinaryOp>, Box<Expr>, Box<Expr>),
    /// `cond ? a : b`
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    /// `a, b`
    Comma(Vec<Expr>),
    /// A call of a function or method: `f(x)` or `tex.Sample(s, uv)`.
    Call(Box<Expr>, Vec<Expr>),
    /// A constructor: `float4(a, b)`.
    Construct(Box<TypeRef>, Vec<Expr>),
    /// `(type)value`
    Cast(Box<TypeRef>, Box<Expr>),
    /// `value.name`, which includes swizzles.
    Member(Box<Expr>, Ident),
    /// `value[index]`
    Index(Box<Expr>, Box<Expr>),
}
