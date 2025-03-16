//! Type definitions for built-in tool use
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum TextEditor {
    View {
        path: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        view_range: Option<[i32; 2]>,
    },
    StrReplace {
        path: String,
        old_str: String,
        new_str: String,
    },
    Create {
        path: String,
        file_text: String,
    },
    Insert {
        path: String,
        insert_line: usize,
        new_str: String,
    },
    UndoEdit {
        path: String,
    },
}
