//! Splits source text into tokens.

use crate::diagnostic::Diagnostic;
use crate::span::{FileId, Span};

/// The kind of a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    /// A name or a keyword. The parser tells keywords from names by their text.
    Ident,
    /// An integer literal, with its suffix.
    Int,
    /// A floating-point literal, with its suffix.
    Float,
    /// A string literal, with its quotes.
    Str,
    /// An operator or punctuation mark.
    Punct(Punct),
    /// A character that is not part of the language.
    Other,
}

/// Operators and punctuation marks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum Punct {
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Semi,
    Comma,
    Dot,
    Colon,
    Question,
    Tilde,
    Bang,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    Amp,
    Pipe,
    Eq,
    Lt,
    Gt,
    PlusPlus,
    MinusMinus,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    CaretEq,
    AmpEq,
    PipeEq,
    ShlEq,
    ShrEq,
    Shl,
    Shr,
    EqEq,
    NotEq,
    LtEq,
    GtEq,
    AndAnd,
    OrOr,
    Hash,
    HashHash,
}

/// Longest first, so that `<<=` wins over `<<` and `<`.
const PUNCTS: [(&str, Punct); 46] = [
    ("<<=", Punct::ShlEq),
    (">>=", Punct::ShrEq),
    ("##", Punct::HashHash),
    ("++", Punct::PlusPlus),
    ("--", Punct::MinusMinus),
    ("+=", Punct::PlusEq),
    ("-=", Punct::MinusEq),
    ("*=", Punct::StarEq),
    ("/=", Punct::SlashEq),
    ("%=", Punct::PercentEq),
    ("^=", Punct::CaretEq),
    ("&=", Punct::AmpEq),
    ("|=", Punct::PipeEq),
    ("<<", Punct::Shl),
    (">>", Punct::Shr),
    ("==", Punct::EqEq),
    ("!=", Punct::NotEq),
    ("<=", Punct::LtEq),
    (">=", Punct::GtEq),
    ("&&", Punct::AndAnd),
    ("||", Punct::OrOr),
    ("(", Punct::LParen),
    (")", Punct::RParen),
    ("[", Punct::LBracket),
    ("]", Punct::RBracket),
    ("{", Punct::LBrace),
    ("}", Punct::RBrace),
    (";", Punct::Semi),
    (",", Punct::Comma),
    (".", Punct::Dot),
    (":", Punct::Colon),
    ("?", Punct::Question),
    ("~", Punct::Tilde),
    ("!", Punct::Bang),
    ("+", Punct::Plus),
    ("-", Punct::Minus),
    ("*", Punct::Star),
    ("/", Punct::Slash),
    ("%", Punct::Percent),
    ("^", Punct::Caret),
    ("&", Punct::Amp),
    ("|", Punct::Pipe),
    ("=", Punct::Eq),
    ("<", Punct::Lt),
    (">", Punct::Gt),
    ("#", Punct::Hash),
];

impl Punct {
    /// The source text of the mark.
    pub fn as_str(self) -> &'static str {
        PUNCTS
            .iter()
            .find(|(_, p)| *p == self)
            .map_or("", |(s, _)| s)
    }
}

/// One token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// What the token is.
    pub kind: TokenKind,
    /// The token's text. A token made by macro pasting or `#` has text that is
    /// not in any file.
    pub text: String,
    /// Where the token's text is: in the file, or in the macro body it came from.
    pub span: Span,
    /// Space or a comment comes before the token on its line.
    pub space_before: bool,
    /// The token is the first one on its line.
    pub line_start: bool,
    /// For a token from a macro body: the macro call in the source it came from.
    pub expanded_from: Option<Span>,
}

impl Token {
    /// Where the token is in the text the author wrote: the macro call for a
    /// token from a macro body, else the token itself.
    pub fn site(&self) -> Span {
        self.expanded_from.unwrap_or(self.span)
    }

    /// The token is this punctuation mark.
    pub fn is(&self, punct: Punct) -> bool {
        self.kind == TokenKind::Punct(punct)
    }

    /// The token is a name or keyword with this text.
    pub fn is_word(&self, word: &str) -> bool {
        self.kind == TokenKind::Ident && self.text == word
    }
}

/// Splits `text` into tokens. Comments and line continuations become space.
pub fn lex(file: FileId, text: &str, diagnostics: &mut Vec<Diagnostic>) -> Vec<Token> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut space = false;
    let mut line_start = true;
    while i < bytes.len() {
        let b = bytes[i];
        let next = bytes.get(i + 1).copied();
        match b {
            b'\n' => {
                line_start = true;
                space = true;
                i += 1;
                continue;
            }
            b' ' | b'\t' | b'\r' | 0x0b | 0x0c => {
                space = true;
                i += 1;
                continue;
            }
            b'\\' if continuation_len(bytes, i) > 0 => {
                i += continuation_len(bytes, i);
                space = true;
                continue;
            }
            b'/' if next == Some(b'/') => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                space = true;
                continue;
            }
            b'/' if next == Some(b'*') => {
                match text[i + 2..].find("*/") {
                    Some(end) => i += end + 4,
                    None => {
                        diagnostics.push(Diagnostic::error(
                            Span::new(file, i, i + 2),
                            "this comment has no closing `*/`",
                        ));
                        i = bytes.len();
                    }
                }
                space = true;
                continue;
            }
            _ => {}
        }
        let start = i;
        let kind = if is_ident_start(b) {
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            TokenKind::Ident
        } else if b.is_ascii_digit() || (b == b'.' && next.is_some_and(|n| n.is_ascii_digit())) {
            let (end, float) = scan_number(bytes, i);
            i = end;
            if float {
                TokenKind::Float
            } else {
                TokenKind::Int
            }
        } else if b == b'"' {
            i = scan_string(bytes, i);
            if i <= start + 1 || bytes[i - 1] != b'"' {
                diagnostics.push(Diagnostic::error(
                    Span::new(file, start, i),
                    "this string has no closing quote",
                ));
            }
            TokenKind::Str
        } else if let Some((s, punct)) = PUNCTS
            .iter()
            .find(|(s, _)| bytes[i..].starts_with(s.as_bytes()))
        {
            i += s.len();
            TokenKind::Punct(*punct)
        } else {
            i += text[i..].chars().next().map_or(1, char::len_utf8);
            TokenKind::Other
        };
        tokens.push(Token {
            kind,
            text: text[start..i].to_string(),
            span: Span::new(file, start, i),
            space_before: space,
            line_start,
            expanded_from: None,
        });
        space = false;
        line_start = false;
    }
    tokens
}

/// The length of a backslash-newline at `i`, or 0.
fn continuation_len(bytes: &[u8], i: usize) -> usize {
    match (bytes.get(i + 1), bytes.get(i + 2)) {
        (Some(b'\n'), _) => 2,
        (Some(b'\r'), Some(b'\n')) => 3,
        _ => 0,
    }
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Scans a number from `i`. Returns its end and if it is a floating-point literal.
fn scan_number(b: &[u8], mut i: usize) -> (usize, bool) {
    let at = |i: usize| b.get(i).copied().unwrap_or(0);
    if at(i) == b'0' && matches!(at(i + 1), b'x' | b'X') && at(i + 2).is_ascii_hexdigit() {
        i += 2;
        while at(i).is_ascii_hexdigit() {
            i += 1;
        }
        while matches!(at(i), b'u' | b'U' | b'l' | b'L') {
            i += 1;
        }
        return (i, false);
    }
    let mut float = false;
    while at(i).is_ascii_digit() {
        i += 1;
    }
    if at(i) == b'.' {
        float = true;
        i += 1;
        while at(i).is_ascii_digit() {
            i += 1;
        }
    }
    if matches!(at(i), b'e' | b'E') {
        let digits = if matches!(at(i + 1), b'+' | b'-') {
            i + 2
        } else {
            i + 1
        };
        if at(digits).is_ascii_digit() {
            float = true;
            i = digits;
            while at(i).is_ascii_digit() {
                i += 1;
            }
        }
    }
    while matches!(at(i), b'f' | b'F' | b'h' | b'H' | b'l' | b'L' | b'u' | b'U') {
        float |= matches!(at(i), b'f' | b'F' | b'h' | b'H');
        i += 1;
    }
    (i, float)
}

/// Scans a string from the quote at `i` to its closing quote or the end of the line.
fn scan_string(b: &[u8], mut i: usize) -> usize {
    i += 1;
    while i < b.len() && b[i] != b'\n' {
        match b[i] {
            b'"' => return i + 1,
            b'\\' if i + 1 < b.len() && b[i + 1] != b'\n' => i += 2,
            _ => i += 1,
        }
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(text: &str) -> Vec<(TokenKind, String)> {
        let mut d = Vec::new();
        let tokens = lex(FileId(0), text, &mut d);
        assert!(d.is_empty(), "{d:?}");
        tokens.into_iter().map(|t| (t.kind, t.text)).collect()
    }

    #[test]
    fn numbers_keep_their_suffixes_and_kinds() {
        let k = kinds("1 1.0 .5 2. 1e3 1e-3f 0x1Fu 10u 1.0h 3f 1.5L");
        let expected = [
            (TokenKind::Int, "1"),
            (TokenKind::Float, "1.0"),
            (TokenKind::Float, ".5"),
            (TokenKind::Float, "2."),
            (TokenKind::Float, "1e3"),
            (TokenKind::Float, "1e-3f"),
            (TokenKind::Int, "0x1Fu"),
            (TokenKind::Int, "10u"),
            (TokenKind::Float, "1.0h"),
            (TokenKind::Float, "3f"),
            (TokenKind::Float, "1.5L"),
        ];
        let expected: Vec<_> = expected.iter().map(|(k, s)| (*k, s.to_string())).collect();
        assert_eq!(k, expected);
    }

    #[test]
    fn longest_punctuation_wins() {
        let k = kinds("a<<=b>>c##d->e");
        let texts: Vec<_> = k.iter().map(|(_, s)| s.as_str()).collect();
        assert_eq!(
            texts,
            ["a", "<<=", "b", ">>", "c", "##", "d", "-", ">", "e"]
        );
    }

    #[test]
    fn comments_are_space_and_block_comments_do_not_end_a_line() {
        let mut d = Vec::new();
        let t = lex(FileId(0), "a // x\nb/* y\n z */c", &mut d);
        assert!(d.is_empty());
        let flags: Vec<_> = t
            .iter()
            .map(|t| (t.text.as_str(), t.line_start, t.space_before))
            .collect();
        assert_eq!(
            flags,
            [("a", true, false), ("b", true, true), ("c", false, true)]
        );
    }

    #[test]
    fn a_continuation_joins_lines() {
        let mut d = Vec::new();
        let t = lex(FileId(0), "#define A 1 \\\r\n + 2\nB", &mut d);
        let starts: Vec<_> = t
            .iter()
            .filter(|t| t.line_start)
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(starts, ["#", "B"]);
    }

    #[test]
    fn strings_and_unknown_characters() {
        let k = kinds(r#""a\"b" @ é"#);
        assert_eq!(k[0], (TokenKind::Str, r#""a\"b""#.to_string()));
        assert_eq!(k[1], (TokenKind::Other, "@".to_string()));
        assert_eq!(k[2], (TokenKind::Other, "é".to_string()));
    }

    #[test]
    fn unterminated_comment_and_string_are_errors() {
        let mut d = Vec::new();
        lex(FileId(0), "\"abc\n/* x", &mut d);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn spans_cover_the_token_text() {
        let mut d = Vec::new();
        let t = lex(FileId(3), "  float4", &mut d);
        assert_eq!(t[0].span, Span::new(FileId(3), 2, 8));
        assert_eq!(Punct::ShlEq.as_str(), "<<=");
    }
}
