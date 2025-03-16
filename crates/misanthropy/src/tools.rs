//! Type definitions for built-in tool use
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum TextEditor {
    View {},
    StrReplace {},
    Create {},
    Insert {},
    UndoEdit {},
}
