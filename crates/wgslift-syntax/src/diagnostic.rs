//! Errors and warnings with the source span they refer to.

use std::fmt;

use crate::span::{SourceFiles, Span};

/// How serious a diagnostic is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// The input is wrong; the result is not complete.
    Error,
    /// The input is accepted, but probably not what the author meant.
    Warning,
}

/// A message about the input, with the place it refers to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// Error or warning.
    pub severity: Severity,
    /// What is wrong, in one sentence.
    pub message: String,
    /// Where it is wrong.
    pub span: Span,
}

impl Diagnostic {
    /// An error at `span`.
    pub fn error(span: Span, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            severity: Severity::Error,
            message: message.into(),
            span,
        }
    }

    /// A warning at `span`.
    pub fn warning(span: Span, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            severity: Severity::Warning,
            message: message.into(),
            span,
        }
    }

    /// The diagnostic as `file:line:column: error: message`.
    pub fn display<'a>(&'a self, files: &'a SourceFiles) -> impl fmt::Display + 'a {
        DisplayDiagnostic {
            diagnostic: self,
            files,
        }
    }
}

struct DisplayDiagnostic<'a> {
    diagnostic: &'a Diagnostic,
    files: &'a SourceFiles,
}

impl fmt::Display for DisplayDiagnostic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.diagnostic.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        write!(
            f,
            "{}: {kind}: {}",
            self.files.location(self.diagnostic.span),
            self.diagnostic.message
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_location_severity_and_message() {
        let mut files = SourceFiles::default();
        let id = files.add("main.hlsl", "float x\n;");
        let d = Diagnostic::error(Span::new(id, 8, 9), "expected a name");
        assert_eq!(
            d.display(&files).to_string(),
            "main.hlsl:2:1: error: expected a name"
        );
        let w = Diagnostic::warning(Span::new(id, 0, 5), "unused");
        assert_eq!(
            w.display(&files).to_string(),
            "main.hlsl:1:1: warning: unused"
        );
    }
}
