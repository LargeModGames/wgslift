//! The front end of wgslift: an HLSL preprocessor, lexer and parser.
//!
//! It turns HLSL source text into a syntax tree with source spans. It does no type
//! checking and no translation; the `wgslift` crate does that on top of this tree.

pub mod diagnostic;
pub mod lexer;
pub mod span;

pub use diagnostic::{Diagnostic, Severity};
pub use lexer::{Punct, Token, TokenKind};
pub use span::{FileId, SourceFile, SourceFiles, Span};
