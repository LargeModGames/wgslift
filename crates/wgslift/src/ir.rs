//! The typed program after semantic analysis. Every conversion and splat is
//! explicit, every expression is free of side effects except calls, and loops are
//! WGSL `loop` blocks, so the printer only prints.

use wgslift_syntax::Span;

use crate::types::Type;

/// A whole translated program.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Module {
    /// Structs, by index.
    pub structs: Vec<StructDef>,
    /// Module-scope variables, constants and resources, by index.
    pub globals: Vec<Global>,
    /// Functions, by index, callees before callers.
    pub functions: Vec<Function>,
    /// The index of the entry-point function.
    pub entry: usize,
}

/// A struct.
#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    /// The HLSL name.
    pub name: String,
    /// The fields in order.
    pub fields: Vec<Field>,
}

/// A struct field.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// The HLSL name.
    pub name: String,
    /// The type.
    pub ty: Type,
    /// The HLSL semantic, for entry-point inputs and outputs.
    pub semantic: Option<String>,
}

/// Where a module-scope declaration lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    /// A member of the uniform buffer.
    Uniform,
    /// A `var<private>`: one copy per invocation, writable.
    Private,
    /// A module-scope `const`.
    Const,
    /// A texture and sampler pair.
    Resource,
}

/// A module-scope declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct Global {
    /// The HLSL name.
    pub name: String,
    /// The type.
    pub ty: Type,
    /// Where it lives.
    pub space: Space,
    /// The value: a const-expression for `Const` and `Private`.
    pub init: Option<Expr>,
    /// For a private variable whose HLSL initializer is not a const-expression:
    /// the value that the entry point assigns first.
    pub entry_init: Option<Expr>,
    /// Where it was declared.
    pub span: Span,
}

/// How a parameter passes its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamDir {
    /// By value.
    In,
    /// Written by the callee only.
    Out,
    /// Read and written by the callee.
    InOut,
}

/// A function parameter. Its value lives in local `local` of the function.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    /// The local that holds the parameter.
    pub local: usize,
    /// How it passes its value.
    pub dir: ParamDir,
    /// The HLSL semantic, for the entry point.
    pub semantic: Option<String>,
}

/// A local variable of a function. Parameters are locals too.
#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    /// The HLSL name.
    pub name: String,
    /// The type.
    pub ty: Type,
    /// The function writes it after its declaration.
    pub written: bool,
}

/// A function.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    /// The HLSL name.
    pub name: String,
    /// The parameters in order.
    pub params: Vec<Param>,
    /// The return type.
    pub ret: Type,
    /// The semantic of the return value, for the entry point.
    pub ret_semantic: Option<String>,
    /// All locals, parameters included, by index.
    pub locals: Vec<Local>,
    /// The statements.
    pub body: Block,
    /// Where it was declared.
    pub span: Span,
}

/// A list of statements.
pub type Block = Vec<Stmt>;

/// The variable at the start of a place or a read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Root {
    /// A local, by index.
    Local(usize),
    /// A global, by index.
    Global(usize),
}

/// One step from a variable into a part of it.
#[derive(Debug, Clone, PartialEq)]
pub enum Access {
    /// A struct field, by index.
    Field(usize),
    /// An array element, a matrix row or a vector component, by a value.
    Index(Expr),
    /// One vector component, 0 to 3.
    Component(u8),
}

/// Something that can be assigned: a variable or one part of it.
#[derive(Debug, Clone, PartialEq)]
pub struct Place {
    /// The variable.
    pub root: Root,
    /// The path into it.
    pub path: Vec<Access>,
}

/// A statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// Declares local `id`, with a value or zero.
    Local {
        /// The local.
        id: usize,
        /// The value; `None` gives zero.
        init: Option<Expr>,
    },
    /// Stores a value.
    Assign(Place, Expr),
    /// Evaluates a call for its effects.
    Eval(Expr),
    /// `if`
    If(Expr, Block, Block),
    /// A WGSL `loop`. `continuing` runs after each pass and after `continue`;
    /// `break_if` ends it.
    Loop {
        /// The body.
        body: Block,
        /// The statements after each pass.
        continuing: Block,
        /// The loop ends when this is true, checked at the end of `continuing`.
        break_if: Option<Expr>,
    },
    /// `break`
    Break,
    /// `continue`
    Continue,
    /// `switch` on an `i32`: cases with their labels; `None` is `default`.
    Switch(Expr, Vec<(Vec<Option<i64>>, Block)>),
    /// `return`
    Return(Option<Expr>),
    /// `discard`
    Discard,
    /// A nested block.
    Block(Block),
}

/// A literal value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Literal {
    /// A bool.
    Bool(bool),
    /// An `i32`.
    Int(i32),
    /// A `u32`.
    Uint(u32),
    /// An `f32`.
    Float(f32),
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// `-x`
    Neg,
    /// `!x` on bools.
    Not,
    /// `~x` on integers.
    BitNot,
}

/// Binary operators. Both operands have the same type, except shifts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    /// A matrix product in WGSL operand order.
    MatMul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// `&` on bools or integers. HLSL `&&` is not short-circuit, so it becomes this.
    And,
    /// `|` on bools or integers.
    Or,
    Xor,
    Shl,
    Shr,
}

/// How a texture is sampled.
#[derive(Debug, Clone, PartialEq)]
pub enum Sample {
    /// `textureSample`
    Plain,
    /// `textureSampleLevel` with this level.
    Level(Box<Expr>),
    /// `textureSampleBias` with this bias.
    Bias(Box<Expr>),
    /// `textureSampleGrad` with these gradients.
    Grad(Box<Expr>, Box<Expr>),
}

/// An argument of a user function call.
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    /// A value for an `in` parameter.
    Value(Expr),
    /// A local passed by pointer, for an `out` or `inout` parameter.
    Ref(usize),
}

/// An expression with its type.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    /// What it computes.
    pub kind: ExprKind,
    /// Its type.
    pub ty: Type,
}

/// The kinds of expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    /// A literal.
    Literal(Literal),
    /// The value of a variable.
    Var(Root),
    /// A struct field of a value.
    Field(Box<Expr>, usize),
    /// An element of an array, a matrix row or a vector component.
    Index(Box<Expr>, Box<Expr>),
    /// Vector components, 1 to 4 of them, each 0 to 3. One component gives a scalar.
    Swizzle(Box<Expr>, Vec<u8>),
    /// A unary operator.
    Unary(UnaryOp, Box<Expr>),
    /// A binary operator.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// A conversion to `ty`, which has the same shape with another component type.
    Convert(Box<Expr>),
    /// A scalar repeated into the vector `ty`.
    Splat(Box<Expr>),
    /// A vector, matrix, array or struct built from parts, as WGSL constructors take them.
    Construct(Vec<Expr>),
    /// `select(reject, accept, cond)`: both values are computed.
    Select {
        /// The condition, a bool or a bool vector.
        cond: Box<Expr>,
        /// The value where the condition is true.
        accept: Box<Expr>,
        /// The value where the condition is false.
        reject: Box<Expr>,
    },
    /// A call of a user function, by index.
    Call(usize, Vec<Arg>),
    /// A call of a WGSL built-in function, by its WGSL name.
    Builtin(&'static str, Vec<Expr>),
    /// A texture sample through a sampler global.
    Sample {
        /// The sampler global.
        sampler: usize,
        /// The coordinates.
        coords: Box<Expr>,
        /// How it samples.
        kind: Sample,
    },
}

impl Expr {
    /// A float literal.
    pub fn float(value: f32) -> Expr {
        Expr {
            kind: ExprKind::Literal(Literal::Float(value)),
            ty: Type::FLOAT,
        }
    }

    /// An `i32` literal.
    pub fn int(value: i32) -> Expr {
        Expr {
            kind: ExprKind::Literal(Literal::Int(value)),
            ty: Type::INT,
        }
    }

    /// A bool literal.
    pub fn bool(value: bool) -> Expr {
        Expr {
            kind: ExprKind::Literal(Literal::Bool(value)),
            ty: Type::BOOL,
        }
    }
}
