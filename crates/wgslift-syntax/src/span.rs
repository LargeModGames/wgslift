//! Source files and byte spans into them.

/// Identifies one source file: the main file or a file it included.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileId(pub u32);

/// A byte range in one source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    /// The file the range is in.
    pub file: FileId,
    /// Offset of the first byte.
    pub start: u32,
    /// Offset after the last byte.
    pub end: u32,
}

impl Span {
    /// A span from `start` to `end` in `file`.
    pub fn new(file: FileId, start: usize, end: usize) -> Span {
        Span {
            file,
            start: start.min(u32::MAX as usize) as u32,
            end: end.min(u32::MAX as usize) as u32,
        }
    }

    /// The smallest span that covers `self` and `other`. Spans from different files
    /// keep `self`.
    pub fn to(self, other: Span) -> Span {
        if self.file != other.file {
            return self;
        }
        Span {
            file: self.file,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// One source file: its name and text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    /// The name the file was opened with, for diagnostics.
    pub name: String,
    /// The full text.
    pub text: String,
    line_starts: Vec<u32>,
}

impl SourceFile {
    fn new(name: String, text: String) -> SourceFile {
        let line_starts = std::iter::once(0)
            .chain(text.match_indices('\n').map(|(i, _)| (i + 1) as u32))
            .collect();
        SourceFile {
            name,
            text,
            line_starts,
        }
    }

    /// The 1-based line and column of a byte offset.
    pub fn line_column(&self, offset: u32) -> (usize, usize) {
        let line = self
            .line_starts
            .partition_point(|start| *start <= offset)
            .max(1);
        let column = offset - self.line_starts[line - 1] + 1;
        (line, column as usize)
    }
}

/// All files that took part in one parse.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceFiles {
    files: Vec<SourceFile>,
}

impl SourceFiles {
    /// Adds a file and returns its id.
    pub fn add(&mut self, name: impl Into<String>, text: impl Into<String>) -> FileId {
        self.files.push(SourceFile::new(name.into(), text.into()));
        FileId((self.files.len() - 1) as u32)
    }

    /// The file with this id.
    pub fn get(&self, id: FileId) -> Option<&SourceFile> {
        self.files.get(id.0 as usize)
    }

    /// The text a span covers, or `""` for a span that is out of range.
    pub fn text(&self, span: Span) -> &str {
        self.get(span.file)
            .and_then(|f| f.text.get(span.start as usize..span.end as usize))
            .unwrap_or("")
    }

    /// `name:line:column` for the start of a span.
    pub fn location(&self, span: Span) -> String {
        match self.get(span.file) {
            Some(file) => {
                let (line, column) = file.line_column(span.start);
                format!("{}:{line}:{column}", file.name)
            }
            None => "<unknown>".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_and_column_are_one_based() {
        let mut files = SourceFiles::default();
        let id = files.add("a.hlsl", "ab\ncd\n\nef");
        let file = files.get(id).unwrap();
        assert_eq!(file.line_column(0), (1, 1));
        assert_eq!(file.line_column(1), (1, 2));
        assert_eq!(file.line_column(3), (2, 1));
        assert_eq!(file.line_column(7), (4, 1));
        assert_eq!(files.location(Span::new(id, 4, 5)), "a.hlsl:2:2");
    }

    #[test]
    fn text_of_a_span_and_out_of_range_spans() {
        let mut files = SourceFiles::default();
        let id = files.add("a", "float4 x;");
        assert_eq!(files.text(Span::new(id, 7, 8)), "x");
        assert_eq!(files.text(Span::new(id, 7, 99)), "");
        assert_eq!(files.text(Span::new(FileId(9), 0, 1)), "");
    }

    #[test]
    fn joining_spans_covers_both() {
        let id = FileId(0);
        assert_eq!(
            Span::new(id, 5, 7).to(Span::new(id, 1, 3)),
            Span::new(id, 1, 7)
        );
        assert_eq!(
            Span::new(id, 5, 7).to(Span::new(FileId(1), 1, 3)),
            Span::new(id, 5, 7)
        );
    }
}
