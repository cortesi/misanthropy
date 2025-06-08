//! Rust client for the Anthropic API.
use std::{env, fs, path::Path};

use base64::prelude::*;
use futures_util::StreamExt;
use log::trace;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest_eventsource::{Event, EventSource};
use schemars::{schema_for, JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Default Anthropic AI model identifier used for API requests.
pub const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";

/// Default maximum number of tokens for AI model responses.
pub const DEFAULT_MAX_TOKENS: u32 = 1024;

/// Environment variable name for the Anthropic API key.
pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";

/// Version of the Anthropic API used in requests.
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";

const DEFAULT_API_DOMAIN: &str = "api.anthropic.com";

/// Name of the built-in text editor tool for Claude 4
pub const TEXT_EDITOR_4: &str = "text_editor_20250429";

/// Name of the built-in text editor tool for Claude 3.7
pub const TEXT_EDITOR_37: &str = "text_editor_20250124";

/// Name of the built-in text editor tool for Claude 3.5
pub const TEXT_EDITOR_35: &str = "text_editor_20241022";

/// Name of the built-in text editor tool for Claude 3.x
pub const TEXT_EDITOR_NAME_3: &str = "str_replace_editor";

/// Name of the built-in text editor tool for Claude 4.x
pub const TEXT_EDITOR_NAME_4: &str = "str_replace_based_edit_tool";

mod error;
pub mod tools;

/// Represents cache control options for conversation blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CacheControl {
    /// Indicates that the content is ephemeral and should not be cached.
    #[serde(rename = "ephemeral")]
    Ephemeral,
}

impl CacheControl {
    /// Serializes the CacheControl enum to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({"type": "ephemeral"})
    }
}

pub use error::*;

/// Specifies how the AI model should choose and use tools in a conversation.
/// Can be set to automatic, any tool, or a specific tool.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolChoice {
    /// Let the model automatically decide whether to use tools.
    #[default]
    Auto,
    /// Allow the model to use any available tool.
    Any,
    /// Instruct the model to use a specific tool.
    Tool {
        /// The name of the specific tool to use.
        name: String,
    },
    /// Instruct the model not to use any tools.
    None,
}

/// Represents a tool that can be used by the AI model in a conversation.
///
/// This enum allows for different types of tools to be defined:
/// - `Custom`: A user-defined tool with a name, description, and input schema.
///
/// # Usage
///
/// To create a new custom tool, define a struct that represents the tool's input,
/// implement `JsonSchema` for it, and use the `Tool::custom()` method:
///
/// ```ignore
/// use schemars::JsonSchema;
/// use your_crate::Tool;
///
/// /// Get the current weather for a location.
/// #[derive(JsonSchema)]
/// struct GetWeather {
///     /// The city and country, e.g., "London, UK"
///     location: String,
///     /// Temperature unit: "celsius" or "fahrenheit"
///     unit: Option<String>,
/// }
///
/// let weather_tool = Tool::custom::<GetWeather>();
/// ```
///
/// The resulting tool will have its name set to "GetWeather", its description
/// set to "Get the current weather for a location.", and its input schema derived
/// from the `GetWeather` struct.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Tool {
    /// A custom tool with a name, description, and input schema.
    Custom {
        /// The name of the tool.
        name: String,

        /// A description of the tool's purpose and functionality.
        description: String,

        /// The JSON schema defining the structure of the tool's input.
        input_schema: Schema,

        /// Optional cache control settings for the tool.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    TextEditor {
        /// Must be equal to `TEXT_EDITOR_NAME_3` or `TEXT_EDITOR_NAME_4`.
        name: String,

        /// The type of the text editor tool. This must match the model, and should be either
        /// equal to the constants in `TEXT_EDITOR_35`, `TEXT_EDITOR_37`, `TEXT_EDITOR_4`.
        #[serde(rename = "type")]
        typ: String,

        /// Optional cache control settings for the tool.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl Tool {
    /// Creates a new custom tool from a type implementing JsonSchema.
    ///
    /// The tool's name and description are automatically derived from the input type.
    pub fn custom<T: JsonSchema>(name: &str) -> Result<Self> {
        let input_schema = schema_for!(T);
        let description = if let Some(d) = input_schema.get("description") {
            d.as_str().unwrap_or("").to_string()
        } else {
            "".to_string()
        };

        Ok(Self::Custom {
            name: name.into(),
            description,
            input_schema,
            cache_control: None,
        })
    }
}

/// An event in the streaming response from the Anthropic API.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// A stream error.
    Error { error: StreamError },

    /// Indicates the start of a new message.
    MessageStart {
        /// The initial message response.
        message: MessagesResponse,
    },
    /// Marks the beginning of a new content block.
    ContentBlockStart {
        /// The index of the content block.
        index: usize,
        /// The initial content of the block.
        content_block: Content,
    },
    /// A keep-alive event to maintain the connection.
    Ping,
    /// Represents an update to an existing content block.
    ContentBlockDelta {
        /// The index of the content block being updated.
        index: usize,
        /// The incremental change to the content block.
        delta: ContentBlockDelta,
    },
    /// Signals the end of a content block.
    ContentBlockStop {
        /// The index of the completed content block.
        index: usize,
    },
    /// Represents an update to the overall message.
    MessageDelta {
        /// The incremental change to the message.
        delta: MessageDelta,
        /// Updated token usage information.
        usage: Usage,
    },
    /// Indicates the completion of the entire message.
    MessageStop,
}

/// An error that has occurred as part of a stream.
#[derive(Debug, Deserialize)]
pub struct StreamError {
    /// The type of stream error.
    #[serde(rename = "type")]
    pub type_: String,
    /// The error message.
    pub message: String,
}

/// An incremental update to a content block in a streaming response.
/// Can be either a text delta or a partial JSON update.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    /// An update to a text content block.
    TextDelta {
        /// The new text to be appended.
        text: String,
    },
    /// An update to a JSON content block.
    InputJsonDelta {
        /// The partial JSON string to be appended or merged.
        partial_json: String,
    },
    /// An update to thinking content in a streaming response.
    ThinkingDelta {
        /// The thinking text to be appended.
        thinking: String,
    },
}

impl ContentBlockDelta {
    /// Returns a string representation of the delta type.
    pub fn typ(&self) -> &'static str {
        match self {
            Self::TextDelta { .. } => "text_delta",
            Self::InputJsonDelta { .. } => "input_json_delta",
            Self::ThinkingDelta { .. } => "thinking_delta",
        }
    }
}

/// Incremental update to a message in a streaming response.
/// Contains changes to the stop reason and stop sequence.
#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    /// The updated reason for why the model stopped generating, if any.
    pub stop_reason: Option<StopReason>,
    /// The updated stop sequence that caused the model to stop, if any.
    pub stop_sequence: Option<String>,
}

/// Manages a streamed response from the Anthropic API for message generation.
///
/// This struct handles the incremental updates to message content and metadata
/// received as streaming events. It encapsulates an `EventSource` for receiving
/// events and maintains the current state of the response.
///
/// The stream is considered complete when `next()` returns `None` or a `MessageStop`
/// event is received, at which point the `event_source` is dropped.
pub struct StreamedResponse {
    /// The accumulated response data from the stream.
    pub response: MessagesResponse,
    /// The underlying event source for the stream, if active.
    event_source: Option<EventSource>,
}

impl StreamedResponse {
    pub fn new(event_source: EventSource) -> Self {
        Self {
            response: MessagesResponse::default(),
            event_source: Some(event_source),
        }
    }

    /// Retrieves the next event from the stream.
    ///
    /// This method asynchronously fetches the next `StreamEvent` from the underlying
    /// `EventSource`. It handles the internal state of the stream, including merging
    /// events into the response and managing stream completion.
    ///
    /// # Returns
    ///
    /// - `Some(Ok(StreamEvent))` if an event was successfully retrieved.
    /// - `Some(Err(...))` if an error occurred while fetching or parsing an event.
    /// - `None` if the stream has been completed or a `MessageStop` event was previously received.
    ///
    /// # Note
    ///
    /// After receiving a `MessageStop` event or when the stream is otherwise completed,
    /// this method will drop the internal `EventSource` and return `None` on subsequent calls.
    pub async fn next(&mut self) -> Option<Result<StreamEvent>> {
        let event_source = self.event_source.as_mut()?;

        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Open) => continue,
                Ok(Event::Message(message)) => match serde_json::from_str(&message.data) {
                    Ok(stream_event) => {
                        trace!("stream event: {stream_event:#?}");
                        self.merge_event(&stream_event);

                        if matches!(stream_event, StreamEvent::MessageStop) {
                            // Drop the event_source when we receive MessageStop
                            self.event_source = None;
                        }

                        return Some(Ok(stream_event));
                    }

                    Err(e) => return Some(Err(Error::ResponseParseError(e))),
                },
                Err(e) => {
                    // Check if this is a transport error that might have status code info
                    if let reqwest_eventsource::Error::Transport(transport_err) = &e {
                        if let Some(status) = transport_err.status() {
                            if status.as_u16() == 429 {
                                return Some(Err(Error::RateLimitExceeded(format!(
                                    "Rate limit exceeded: {e}"
                                ))));
                            }
                        }
                    }
                    return Some(Err(Error::StreamError(e.to_string())));
                }
            }
        }

        // If we've reached this point, the event_source has been exhausted
        self.event_source = None;
        None
    }

    fn merge_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::MessageStart { message } => {
                self.response.id = message.id.clone();
                self.response.model = message.model.clone();
                self.response.role = message.role.clone();
                self.response.content = message.content.clone();
                self.response.stop_reason = message.stop_reason.clone();
                self.response.stop_sequence = message.stop_sequence.clone();
                self.response.usage = self.response.usage.merge(&message.usage);
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                if self.response.content.len() <= *index {
                    self.response.content.push(content_block.clone());
                }
            }
            StreamEvent::ContentBlockDelta { index, delta } => {
                if let Some(block) = self.response.content.get_mut(*index) {
                    match (block, delta) {
                        (
                            Content::Text(text),
                            ContentBlockDelta::TextDelta { text: delta_text },
                        ) => {
                            text.text.push_str(delta_text);
                        }
                        (
                            Content::Thinking(thinking_content),
                            ContentBlockDelta::ThinkingDelta {
                                thinking: delta_thinking,
                            },
                        ) => {
                            thinking_content.thinking.push_str(delta_thinking);
                        }
                        (Content::ToolUse(_), _) => {
                            unimplemented!(
                                "Partial updates for ToolUse blocks are not supported yet"
                            );
                        }
                        (block, delta) => {
                            log::warn!(
                                "Received {} for {} content block at index {index}",
                                delta.typ(),
                                block.typ()
                            );
                        }
                    }
                }
            }
            StreamEvent::MessageDelta { delta, usage } => {
                self.response.stop_reason = delta.stop_reason.clone();
                self.response.stop_sequence = delta.stop_sequence.clone();
                self.response.usage = self.response.usage.merge(usage);
            }
            StreamEvent::ContentBlockStop { .. }
            | StreamEvent::Ping
            | StreamEvent::MessageStop
            | StreamEvent::Error { .. } => {}
        }
    }

    pub fn content_text(&self) -> String {
        self.response
            .content
            .iter()
            .filter_map(|block| {
                if let Content::Text(text) = block {
                    Some(text.text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<String>>()
            .join("")
    }
}

/// Represents the types of errors that can be returned by the Anthropic API.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiErrorType {
    /// There was an issue with the format or content of the request (HTTP 400).
    InvalidRequestError,
    /// There's an issue with the API key (HTTP 401).
    AuthenticationError,
    /// The API key does not have permission to use the specified resource (HTTP 403).
    PermissionError,
    /// The requested resource was not found (HTTP 404).
    NotFoundError,
    /// The account has hit a rate limit (HTTP 429).
    RateLimitError,
    /// An unexpected error has occurred internal to Anthropic's systems (HTTP 500).
    ApiError,
    /// Anthropic's API is temporarily overloaded (HTTP 529).
    OverloadedError,
    /// An error type not explicitly handled (other HTTP status codes).
    #[serde(other)]
    Other,
}

/// The top-level error structure returned by the Anthropic API.
/// Includes a nested ApiError with more details.
#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    /// The type of the response; will always be "error".
    /// See `error.error_type` for more specific information.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Detailed information about the error.
    pub error: ApiError,
}

/// Contains detailed information about an error returned by the Anthropic API.
#[derive(Debug, Deserialize)]
pub struct ApiError {
    /// A string identifier for the error type.
    #[serde(rename = "type")]
    pub error_type: ApiErrorType,
    /// A human-readable description of the error.
    pub message: String,
}

/// The role of a participant in a conversation. Can be either a user or an assistant.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// The user or client interacting with the AI.
    #[default]
    User,
    /// The AI assistant responding to the user.
    Assistant,
}

/// Represents the reason why the model stopped generating content.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model reached a natural stopping point in its generation.
    EndTurn,

    /// The generation was stopped because it reached the maximum number of tokens
    /// specified in the request or the model's maximum limit.
    MaxTokens,

    /// The generation was stopped because it produced one of the custom stop sequences
    /// provided in the request.
    StopSequence,

    /// The model invoked one or more tools, which terminated its generation.
    ToolUse,
}

/// The response from the Anthropic API for a message request. Contains generated content, message
/// metadata, and usage statistics.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct MessagesResponse {
    /// The generated content of the message.
    pub content: Vec<Content>,
    /// Unique identifier for this message.
    pub id: String,
    /// The AI model used to generate this response.
    pub model: String,
    /// The role of the entity that produced this message.
    pub role: Role,
    /// The reason why the AI stopped generating content, if applicable.
    pub stop_reason: Option<StopReason>,
    /// The sequence that caused the AI to stop generating, if applicable.
    pub stop_sequence: Option<String>,
    /// Always "message" for this type of response.
    #[serde(rename = "type")]
    pub message_type: String,
    /// Token usage statistics for this response. For streaming responses, this is cumulative over
    /// all streamed messages.
    pub usage: Usage,
}

impl MessagesResponse {
    pub fn format_content(&self) -> String {
        let mut output = String::new();
        let mut has_user_messages = false;

        for content in &self.content {
            match content {
                Content::Text(text) => {
                    if self.role == Role::User {
                        has_user_messages = true;
                        output.push_str(&format!("user: {}\n", text.text));
                    } else {
                        output.push_str(&format!("assistant: {}\n", text.text));
                    }
                }
                Content::Image(image) => {
                    output.push_str(&format!(
                        "{}: [Image: {} {}]\n",
                        if self.role == Role::User {
                            "user"
                        } else {
                            "assistant"
                        },
                        image.source.source_type,
                        image.source.media_type
                    ));
                    has_user_messages = true; // Always show roles if there are images
                }
                Content::ToolUse(tool_use) => {
                    output.push_str(&format!(
                        "{}: [Tool({}): {} {}]\n",
                        if self.role == Role::User {
                            "user"
                        } else {
                            "assistant"
                        },
                        tool_use.id,
                        tool_use.name,
                        tool_use.input
                    ));
                    has_user_messages = true; // Always show roles if there are tools
                }
                Content::ToolResult(tool_result) => {
                    output.push_str(&format!(
                        "{}: [Tool Result({}): {}]\n",
                        if self.role == Role::User {
                            "user"
                        } else {
                            "assistant"
                        },
                        tool_result.tool_use_id,
                        tool_result.content
                    ));
                    has_user_messages = true; // Always show roles if there are tool results
                }
                Content::Thinking(thinking) => {
                    output.push_str(&format!(
                        "{}: [Thinking: {}]\n",
                        if self.role == Role::User {
                            "user"
                        } else {
                            "assistant"
                        },
                        thinking.thinking
                    ));
                }
            }
        }

        if !has_user_messages {
            // If there are only assistant responses, remove the "assistant: " prefix
            output = output
                .lines()
                .map(|line| line.trim_start_matches("assistant: "))
                .collect::<Vec<&str>>()
                .join("\n");
        }

        output.trim().to_string()
    }
}

/// A tool used by the AI model during a conversation.
/// Should be supplied in an assistant message.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolUse {
    /// Unique identifier for this tool use instance.
    pub id: String,
    /// The name of the tool that was used.
    pub name: String,
    /// The input provided to the tool, in a flexible JSON format.
    pub input: Value,
    /// Optional cache control settings for the tool use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl ToolUse {
    pub fn new(id: String, name: String, input: Value) -> Self {
        Self {
            id,
            name,
            input,
            cache_control: None,
        }
    }
}

/// A response to a tool used by the AI model during a conversation.
/// Should be supplied in a user message.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolResult {
    /// The unique identifier of the tool use instance.
    pub tool_use_id: String,
    /// The output of the tool. Arbitrary format, but should be intelligible to the assistant.
    pub content: String,
    /// Is the response an error?
    #[serde(skip_serializing_if = "is_false")]
    pub is_error: bool,
}

impl ToolResult {
    pub fn new(tool_use_id: String, content: String) -> Self {
        Self {
            tool_use_id,
            content,
            is_error: false,
        }
    }
}

/// Textual content in a message.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Text {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl Text {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            cache_control: None,
        }
    }
}

/// Thinking content in a message.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ThinkingContent {
    pub thinking: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl ThinkingContent {
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            thinking: thinking.into(),
            signature: None,
        }
    }
}

/// An image with its source information in a message.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Image {
    pub source: Source,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl Image {
    pub fn new(source: Source) -> Self {
        Self {
            source,
            cache_control: None,
        }
    }
}

/// A piece of content in a message.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// Textual content.
    Text(Text),
    /// An image with its source information.
    Image(Image),
    /// Details of a tool used by the AI.
    ToolUse(ToolUse),
    /// The result to a tool used by the AI.
    ToolResult(ToolResult),
    /// Thinking content from the AI.
    Thinking(ThinkingContent),
}

impl Content {
    /// Returns a string representation of the content type.
    pub fn typ(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::Image(_) => "image",
            Self::ToolUse(_) => "tool_use",
            Self::ToolResult(_) => "tool_result",
            Self::Thinking(_) => "thinking",
        }
    }

    /// Creates a new text content block.
    pub fn text(text: impl Into<String>) -> Self {
        Content::Text(Text::new(text))
    }

    /// Creates a new image content block from a file path by reading the image
    /// data and encoding it as Base64.
    pub fn image(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        let image_data = fs::read(path)?;
        let base64_image = BASE64_STANDARD.encode(image_data);

        Ok(Content::Image(Image::new(Source {
            source_type: "base64".to_string(),
            media_type: Self::detect_media_type(path),
            data: base64_image,
        })))
    }

    /// Creates a tool result block given a tool use and some content.
    pub fn tool_result(tool_use: &ToolUse, content: impl Into<String>) -> Self {
        Content::ToolResult(ToolResult {
            tool_use_id: tool_use.id.clone(),
            content: content.into(),
            is_error: false,
        })
    }

    fn detect_media_type(path: &Path) -> String {
        match path.extension().and_then(std::ffi::OsStr::to_str) {
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            _ => "application/octet-stream", // Default to binary data if unknown
        }
        .to_string()
    }
}

/// Metadata for an image in a message.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Source {
    /// The type of image source (e.g., "base64").
    #[serde(rename = "type")]
    pub source_type: String,
    /// MIME type of the image (e.g., "image/jpeg").
    pub media_type: String,
    /// The image data, typically base64-encoded.
    pub data: String,
}

/// Token usage statistics for a message.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Usage {
    /// Number of tokens in the input message
    pub input_tokens: Option<u32>,
    /// Number of tokens in the output message
    pub output_tokens: Option<u32>,
    /// Number of input tokens that went to cache creation
    pub cache_creation_input_tokens: Option<u32>,
    /// Number of input tokens that resulted in a cache read
    pub cache_read_input_tokens: Option<u32>,
}

impl Usage {
    pub fn merge(&self, other: &Usage) -> Self {
        Usage {
            input_tokens: Some(self.input_tokens.unwrap_or(0) + other.input_tokens.unwrap_or(0)),
            output_tokens: Some(self.output_tokens.unwrap_or(0) + other.output_tokens.unwrap_or(0)),
            cache_creation_input_tokens: Some(
                self.cache_creation_input_tokens.unwrap_or(0)
                    + other.cache_creation_input_tokens.unwrap_or(0),
            ),
            cache_read_input_tokens: Some(
                self.cache_read_input_tokens.unwrap_or(0)
                    + other.cache_read_input_tokens.unwrap_or(0),
            ),
        }
    }
}

fn is_default_tool_choice(choice: &ToolChoice) -> bool {
    *choice == ToolChoice::Auto
}

fn is_false(b: &bool) -> bool {
    !(*b)
}

/// Configuration for enabling Claude's extended thinking mode.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Thinking {
    /// The type of thinking mode. Must be "enabled".
    #[serde(rename = "type")]
    pub thinking_type: String,
    /// The number of tokens allocated for the thinking process.
    /// Must be at least 1024 and less than max_tokens.
    pub budget_tokens: u32,
}

impl Thinking {
    /// Creates a new Thinking configuration with the specified token budget.
    pub fn new(budget_tokens: u32) -> Self {
        Self {
            thinking_type: "enabled".to_string(),
            budget_tokens,
        }
    }
}

/// A request to the Anthropic API for message generation.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MessagesRequest {
    /// The AI model to use for generating the response.
    pub model: String,
    /// The maximum number of tokens to generate.
    pub max_tokens: u32,
    /// The conversation history and any new messages to process.
    pub messages: Vec<Message>,
    /// Optional system messages to guide the AI's behavior.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub system: Vec<Content>,
    /// Optional temperature setting for response randomness.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Whether to return a streaming response.
    pub stream: bool,
    /// List of tools available for the AI to use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    /// How the AI should choose which tool to use, if any.
    #[serde(default, skip_serializing_if = "is_default_tool_choice")]
    pub tool_choice: ToolChoice,
    /// Optional list of stop sequences to end generation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    /// Optional thinking configuration for extended reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
}

impl Default for MessagesRequest {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            messages: Vec::new(),
            system: Vec::new(),
            temperature: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: ToolChoice::default(),
            stop_sequences: Vec::new(),
            thinking: None,
        }
    }
}

impl MessagesRequest {
    /// Merges a MessagesResponse into the current MessagesRequest.
    ///
    /// Adds the entire content of the given response as a new assistant message
    /// to the conversation history, preserving all content types.
    ///
    /// Useful for maintaining context in ongoing conversations by incorporating
    /// AI responses into the history for subsequent requests.
    pub fn merge_response(&mut self, response: &MessagesResponse) {
        let new_message = Message {
            role: Role::Assistant,
            content: response.content.clone(),
        };
        self.messages.push(new_message);
    }

    /// Merges a StreamedResponse into the current MessagesRequest.
    ///
    /// Adds the entire content of the given streamed response as a new assistant message
    /// to the conversation history, preserving all content types.
    ///
    /// Useful for maintaining context in ongoing conversations by incorporating
    /// streamed AI responses into the history for subsequent requests.
    pub fn merge_streamed_response(&mut self, response: &StreamedResponse) {
        let new_message = Message {
            role: Role::Assistant,
            content: response.response.content.clone(),
        };
        self.messages.push(new_message);
    }

    /// Adds a text editor tool to the request.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the text editor tool, e.g. `text_editor`
    /// * `typ` - The type of the text editor. This must match the model, and should be either
    ///   `TEXT_EDITOR_35`, `TEXT_EDITOR_37`, `TEXT_EDITOR_4".
    pub fn with_text_editor(mut self, typ: impl Into<String>) -> Self {
        let typ = typ.into();
        let name = if typ == TEXT_EDITOR_4 {
            TEXT_EDITOR_NAME_4
        } else {
            TEXT_EDITOR_NAME_3
        }
        .into();

        self.tools.push(Tool::TextEditor {
            name,
            typ,
            cache_control: None,
        });
        self
    }

    /// Adds a custom tool to the request.
    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Sets the tool choice for the AI model.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Sets the AI model to use for generating the response.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the temperature for the AI model's response generation.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the maximum number of tokens to generate in the response.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets whether the request should return a streaming response.
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Sets the messages for the request.
    pub fn with_system(mut self, system: Vec<Content>) -> Self {
        self.system = system;
        self
    }

    /// Adds a system message to the request, appending it to the existing system messages.
    pub fn add_system(&mut self, content: Content) {
        self.system.push(content);
    }

    /// Sets the stop sequences for the request.
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = stop_sequences;
        self
    }

    /// Adds a stop sequence to the request.
    pub fn add_stop_sequence(&mut self, stop_sequence: &str) {
        self.stop_sequences.push(stop_sequence.into());
    }

    /// Enables thinking mode with the specified token budget.
    pub fn with_thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(Thinking::new(budget_tokens));
        self
    }

    /// Add a user message to the conversation history. Appends the content to the last user
    /// message if we're still in the same role, otherwise creates a new user message.
    pub fn add_user(&mut self, content: Content) {
        self.add_content(Role::User, content);
    }

    /// Add an assistant message to the conversation history. Appends the content to the last
    /// assistant message if we're still in the same role, otherwise creates a new assistant
    /// message.
    pub fn add_assistant(&mut self, content: Content) {
        self.add_content(Role::Assistant, content);
    }

    /// Add a message with a specific role and content to the conversation history. Appends the
    /// content to the last message of that role if it exists, otherwise creates a new message.
    fn add_content(&mut self, role: Role, content: Content) {
        if let Some(last_message) = self.messages.last_mut() {
            if last_message.role == role {
                last_message.content.push(content);
                return;
            }
        }

        let mut new_message = Message::new(role);
        new_message.content.push(content);
        self.messages.push(new_message);
    }
}

/// A single message in a conversation, with a role and content.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    /// The role of the message sender (user or assistant).
    pub role: Role,
    /// The content blocks within this message.
    pub content: Vec<Content>,
}

impl Message {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            content: Vec::new(),
        }
    }

    pub fn format_content(&self) -> String {
        let role_prefix = match self.role {
            Role::User => "User: ",
            Role::Assistant => "Assistant: ",
        };

        let content_strings: Vec<String> = self
            .content
            .iter()
            .map(|content| match content {
                Content::Text(text) => text.text.clone(),
                Content::Image(image) => {
                    format!(
                        "[Image: {} {}]",
                        image.source.source_type, image.source.media_type
                    )
                }
                Content::ToolUse(tool_use) => {
                    format!(
                        "[Tool({}): {} {}]",
                        tool_use.id, tool_use.name, tool_use.input
                    )
                }
                Content::ToolResult(tool_result) => {
                    format!(
                        "[Tool Result({}): {}]",
                        tool_result.tool_use_id, tool_result.content
                    )
                }
                Content::Thinking(thinking) => {
                    format!("[Thinking: {}]", thinking.thinking)
                }
            })
            .collect();

        let formatted_content = content_strings.join("\n");

        format!("{role_prefix}{formatted_content}")
    }
}

/// Client for interacting with the Anthropic API.
/// Manages authentication and default parameters for requests.
#[derive(Debug)]
pub struct Anthropic {
    api_key: String,
    base_url: String,
}

impl Anthropic {
    /// Creates a new Anthropic client with an optional API key.
    /// Uses default values for model and max_tokens.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: format!("https://{DEFAULT_API_DOMAIN}"),
        }
    }

    /// Creates an Anthropic client using the API key from the environment.
    /// Reads the key from the ANTHROPIC_API_KEY environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key = env::var(ANTHROPIC_API_KEY_ENV)?;
        Ok(Self::new(&api_key))
    }

    // Creates an Anthropic client using a provided API key or the environment.
    /// If the provided string is empty, falls back to the ANTHROPIC_API_KEY environment variable.
    pub fn from_string_or_env(api_key: &str) -> Result<Self> {
        if !api_key.is_empty() {
            Ok(Self::new(api_key))
        } else {
            Self::from_env()
        }
    }

    /// Creates the headers for API requests, including the beta header if enabled.
    fn create_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_str(&self.api_key)?);
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_API_VERSION),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(headers)
    }

    /// Sends a message request to the Anthropic API and returns a streaming response.
    /// Allows processing of incremental updates as they arrive from the API.
    ///
    /// It is an error to pass a `MessagesRequest` with `stream` set to `false`.
    pub fn messages_stream(&self, request: &MessagesRequest) -> Result<StreamedResponse> {
        if !request.stream {
            return Err(Error::BadRequest(
                "Streaming requests must have stream set to true".to_string(),
            ));
        }
        let event_source = EventSource::new(
            reqwest::Client::new()
                .post(format!("{}/v1/messages", self.base_url))
                .headers(self.create_headers()?)
                .json(&request),
        )
        .map_err(|e| Error::EventSourceError(e.to_string()))?;

        Ok(StreamedResponse::new(event_source))
    }

    /// Sends a message request to the Anthropic API and returns the response.
    /// Uses client defaults for model and max_tokens if not specified in the request.
    pub async fn messages(&self, request: &MessagesRequest) -> Result<MessagesResponse> {
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/v1/messages", self.base_url))
            .headers(self.create_headers()?)
            .json(&request)
            .send()
            .await?;

        let status = response.status();

        if status.is_success() {
            let messages_response: MessagesResponse = response.json().await?;
            Ok(messages_response)
        } else {
            let error_response: ApiErrorResponse = response.json().await?;
            Err(error_response.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// This is a test description
    #[allow(dead_code)]
    #[derive(JsonSchema)]
    struct TestInput {
        required_field: String,
        optional_field: Option<i32>,
        enum_field: TestEnum,
    }

    #[allow(dead_code)]
    #[derive(JsonSchema)]
    enum TestEnum {
        OptionA,
        OptionB,
    }

    #[test]
    fn test_tool_creation_and_serialization() {
        // Create a tool
        let tool = Tool::custom::<TestInput>("testtool").unwrap();

        // Serialize the tool to JSON
        let json = serde_json::to_value(&tool).expect("Failed to serialize Tool to JSON");

        // Check the structure of the serialized JSON
        assert!(json.is_object());
        let json_obj = json.as_object().unwrap();

        // Check the basic properties
        let (name, description) = match &tool {
            Tool::Custom {
                name, description, ..
            } => (name, description),
            Tool::TextEditor { name, .. } => (name, &"Text editor tool".to_string()),
        };
        assert_eq!(name, "testtool");
        assert_eq!(description, "This is a test description");

        assert!(json_obj.contains_key("name"));
        assert!(json_obj.contains_key("description"));
        assert!(json_obj.contains_key("input_schema"));

        // Check the input_schema
        let input_schema = &json_obj["input_schema"];
        assert!(input_schema.is_object());
        let schema_obj = input_schema.as_object().unwrap();

        // Check for required fields in the schema
        assert!(schema_obj.contains_key("type"));
        assert!(schema_obj.contains_key("properties"));
        assert!(schema_obj.contains_key("required"));

        // Check the properties in the schema
        let properties = &schema_obj["properties"];
        assert!(properties.is_object());
        let props_obj = properties.as_object().unwrap();

        assert!(props_obj.contains_key("required_field"));
        assert!(props_obj.contains_key("optional_field"));
        assert!(props_obj.contains_key("enum_field"));

        // Check the enum field
        let enum_field = &props_obj["enum_field"];
        assert!(enum_field.is_object());

        // Check the definitions for the enum
        let definitions = &schema_obj["$defs"];
        assert!(definitions.is_object());
        let defs_obj = definitions.as_object().unwrap();

        assert!(defs_obj.contains_key("TestEnum"));
        let test_enum_def = &defs_obj["TestEnum"];
        assert!(test_enum_def.is_object());

        let test_enum_obj = test_enum_def.as_object().unwrap();

        assert_eq!(test_enum_obj["type"], "string");
        let enum_values = test_enum_obj["enum"].as_array().unwrap();
        assert!(enum_values.contains(&Value::String("OptionA".to_string())));
        assert!(enum_values.contains(&Value::String("OptionB".to_string())));
    }

    #[test]
    fn test_cache_control_serialization() {
        let cache_control = CacheControl::Ephemeral;
        let json = cache_control.to_json();
        assert_eq!(json, json!({"type": "ephemeral"}));

        let serialized = serde_json::to_string(&cache_control).unwrap();
        assert_eq!(serialized, r#"{"type":"ephemeral"}"#);

        let deserialized: CacheControl = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, CacheControl::Ephemeral);
    }

    #[test]
    fn test_messages_request_thinking_field() {
        // Test default value
        let request = MessagesRequest::default();
        assert_eq!(request.thinking, None);

        // Test with thinking enabled
        let request_with_thinking = MessagesRequest::default().with_thinking(1024);
        assert!(request_with_thinking.thinking.is_some());
        let thinking = request_with_thinking.thinking.as_ref().unwrap();
        assert_eq!(thinking.thinking_type, "enabled");
        assert_eq!(thinking.budget_tokens, 1024);

        // Test serialization - thinking field should be omitted when None
        let request_json = serde_json::to_value(&request).unwrap();
        assert!(!request_json.as_object().unwrap().contains_key("thinking"));

        // Test serialization - thinking field should be present when Some
        let request_with_thinking_json = serde_json::to_value(&request_with_thinking).unwrap();
        assert!(request_with_thinking_json["thinking"].is_object());
        assert_eq!(
            request_with_thinking_json["thinking"]["type"],
            json!("enabled")
        );
        assert_eq!(
            request_with_thinking_json["thinking"]["budget_tokens"],
            json!(1024)
        );

        // Test deserialization
        let json_str = r#"{"model":"claude-sonnet-4-20250514","max_tokens":2048,"messages":[],"stream":false,"thinking":{"type":"enabled","budget_tokens":1024}}"#;
        let deserialized: MessagesRequest = serde_json::from_str(json_str).unwrap();
        assert!(deserialized.thinking.is_some());
        let thinking = deserialized.thinking.as_ref().unwrap();
        assert_eq!(thinking.thinking_type, "enabled");
        assert_eq!(thinking.budget_tokens, 1024);
    }

    #[test]
    fn test_content_block_delta_thinking() {
        // Test deserialization of ThinkingDelta as part of ContentBlockDelta
        let json_str = r#"{
            "type": "thinking_delta",
            "thinking": "\n2. 453 = 400 + 50 + 3"
        }"#;

        let delta: ContentBlockDelta = serde_json::from_str(json_str).unwrap();
        match delta {
            ContentBlockDelta::ThinkingDelta { thinking } => {
                assert_eq!(thinking, "\n2. 453 = 400 + 50 + 3");
            }
            _ => panic!("Expected ThinkingDelta"),
        }

        // Test deserialization of a complete content_block_delta event with thinking
        let event_json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "thinking_delta",
                "thinking": "\n2. 453 = 400 + 50 + 3"
            }
        }"#;

        let event: StreamEvent = serde_json::from_str(event_json).unwrap();
        match event {
            StreamEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    ContentBlockDelta::ThinkingDelta { thinking } => {
                        assert_eq!(thinking, "\n2. 453 = 400 + 50 + 3");
                    }
                    _ => panic!("Expected ThinkingDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDelta event"),
        }
    }

    #[test]
    fn test_content_thinking() {
        // Test Content::Thinking serialization and deserialization
        let thinking_content = Content::Thinking(ThinkingContent::new("Let me analyze this..."));

        let json = serde_json::to_value(&thinking_content).unwrap();
        assert_eq!(json["type"], "thinking");
        assert_eq!(json["thinking"], "Let me analyze this...");
        assert!(json.get("signature").is_none()); // Should be omitted when None

        // Test deserialization without signature
        let json_str = r#"{"type": "thinking", "thinking": "Let me analyze this..."}"#;
        let deserialized: Content = serde_json::from_str(json_str).unwrap();
        match deserialized {
            Content::Thinking(thinking) => {
                assert_eq!(thinking.thinking, "Let me analyze this...");
                assert_eq!(thinking.signature, None);
            }
            _ => panic!("Expected Content::Thinking"),
        }

        // Test deserialization with signature
        let json_str_with_sig = r#"{
            "type": "thinking",
            "thinking": "Let me analyze this step by step...",
            "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3URve/op91XWHOEBLLqIOMfFG/UvLEczmEsUjavL...."
        }"#;
        let deserialized: Content = serde_json::from_str(json_str_with_sig).unwrap();
        match deserialized {
            Content::Thinking(thinking) => {
                assert_eq!(thinking.thinking, "Let me analyze this step by step...");
                assert_eq!(thinking.signature, Some("WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3URve/op91XWHOEBLLqIOMfFG/UvLEczmEsUjavL....".to_string()));
            }
            _ => panic!("Expected Content::Thinking"),
        }
    }
}
