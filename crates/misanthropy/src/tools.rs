//! Type definitions for built-in tool use
use serde::{Deserialize, Serialize};

/// Commands for the built-in text editor tool.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum TextEditor {
    /// View the contents of a file at the specified path.
    View {
        /// Path to the file to view.
        path: String,
        /// Optional range of lines to view [start, end].
        #[serde(skip_serializing_if = "Option::is_none")]
        view_range: Option<[i32; 2]>,
    },
    /// Replace occurrences of a string in a file.
    StrReplace {
        /// Path to the file to modify.
        path: String,
        /// String to search for and replace.
        old_str: String,
        /// String to replace with.
        new_str: String,
    },
    /// Create a new file with the specified content.
    Create {
        /// Path where the file will be created.
        path: String,
        /// Content to write to the new file.
        file_text: String,
    },
    /// Insert text at a specific line in a file.
    Insert {
        /// Path to the file to modify.
        path: String,
        /// Line number where text will be inserted.
        insert_line: usize,
        /// Text to insert at the specified line.
        new_str: String,
    },
    /// Undo the last edit made to a file.
    UndoEdit {
        /// Path to the file to undo edits on.
        path: String,
    },
}
