//! Rust client for the Anthropic API.
use std::any::type_name;
use std::{env, fs, path::Path};

use base64::prelude::*;
use futures_util::StreamExt;
use log::trace;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest_eventsource::{Event, EventSource};
use schemars::{schema::RootSchema, schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const DEFAULT_MODEL: &str = "claude-3-opus-20240229";
pub const DEFAULT_MAX_TOKENS: u32 = 1024;
pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const DEFAULT_API_DOMAIN: &str = "api.anthropic.com";

mod error;

pub use error::*;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    #[default]
    Auto,
    Any,
    #[serde(rename = "tool")]
    SpecificTool(String),
}

/// Represents a tool that can be used by the AI model in a conversation.
///
/// A `Tool` defines a capability that the AI can use to perform specific tasks
/// or retrieve information. It consists of a name, description, and an input schema
/// that defines the structure of the data the tool expects.
///
/// The name and description are automatically derived from the input type used
/// to create the tool, leveraging Rust's type system and documentation features.
///
/// # Fields
///
/// * `name`: The name of the tool, automatically derived from the input type's name.
/// * `description`: A description of what the tool does, taken from the input type's doc comment.
/// * `input_schema`: A `RootSchema` that defines the structure of the input the tool expects.
///
/// # Usage
///
/// To create a new `Tool`, define a struct that represents the tool's input,
/// implement `JsonSchema` for it, and use the `Tool::new()` method:
///
/// ```
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
/// let weather_tool = Tool::new::<GetWeather>();
/// ```
///
/// The resulting `Tool` will have its name set to "GetWeather", its description
/// set to "Get the current weather for a location.", and its input schema derived
/// from the `GetWeather` struct.
///
/// # Note
///
/// Ensure that the struct used to create the tool has a meaningful name and
/// is well-documented, as this information is used directly in the created `Tool`.
#[derive(Debug, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: RootSchema,
}

impl Tool {
    pub fn new<T: JsonSchema>() -> Self {
        let schema = schema_for!(T);
        let description = schema
            .schema
            .metadata
            .as_ref()
            .and_then(|m| m.description.clone())
            .unwrap_or_else(|| "No description provided".to_string());

        let name = type_name::<T>()
            .split("::")
            .last()
            .unwrap_or("UnknownTool")
            .to_string();

        Self {
            name,
            description,
            input_schema: schema,
        }
    }

    pub fn with_name<T: JsonSchema>(name: String) -> Self {
        let mut tool = Self::new::<T>();
        tool.name = name;
        tool
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: MessagesResponse,
    },
    ContentBlockStart {
        index: usize,
        content_block: Content,
    },
    Ping,
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: Usage,
    },
    MessageStop,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<StopReason>,
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
    pub response: MessagesResponse,
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
                        trace!("stream event: {:#?}", stream_event);
                        self.merge_event(&stream_event);

                        if matches!(stream_event, StreamEvent::MessageStop) {
                            // Drop the event_source when we receive MessageStop
                            self.event_source = None;
                        }

                        return Some(Ok(stream_event));
                    }

                    Err(e) => return Some(Err(Error::ResponseParseError(e))),
                },
                Err(e) => return Some(Err(Error::StreamError(e.to_string()))),
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
                self.response.usage = message.usage.clone();
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
                    match block {
                        Content::Text { text } => match delta {
                            ContentBlockDelta::TextDelta { text: delta_text } => {
                                text.push_str(delta_text);
                            }
                            ContentBlockDelta::InputJsonDelta { .. } => {}
                        },
                        Content::Image { .. } => {}
                        Content::ToolUse { .. } => {}
                    }
                }
            }
            StreamEvent::MessageDelta { delta, usage } => {
                self.response.stop_reason = delta.stop_reason.clone();
                self.response.stop_sequence = delta.stop_sequence.clone();
                self.response.usage = usage.clone();
            }
            StreamEvent::ContentBlockStop { .. } | StreamEvent::Ping | StreamEvent::MessageStop => {
            }
        }
    }

    pub fn content_text(&self) -> String {
        self.response
            .content
            .iter()
            .filter_map(|block| {
                if let Content::Text { text } = block {
                    Some(text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<String>>()
            .join("")
    }
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: ApiError,
}

#[derive(Debug, Deserialize)]
pub struct ApiError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// The role of a participant in a conversation. Can be either a user or an assistant.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    #[default]
    User,
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
    pub content: Vec<Content>,
    pub id: String,
    pub model: String,
    pub role: Role,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,

    /// Always "message" for this type of response.
    #[serde(rename = "type")]
    pub message_type: String,

    pub usage: Usage,
}

impl MessagesResponse {
    pub fn format_nicely(&self) -> String {
        let mut output = String::new();
        let mut has_user_messages = false;

        for content in &self.content {
            match content {
                Content::Text { text } => {
                    if self.role == Role::User {
                        has_user_messages = true;
                        output.push_str(&format!("user: {}\n", text));
                    } else {
                        output.push_str(&format!("assistant: {}\n", text));
                    }
                }
                Content::Image { source } => {
                    output.push_str(&format!(
                        "{}: [Image: {} {}]\n",
                        if self.role == Role::User {
                            "user"
                        } else {
                            "assistant"
                        },
                        source.source_type,
                        source.media_type
                    ));
                    has_user_messages = true; // Always show roles if there are images
                }
                Content::ToolUse(tool_use) => {
                    output.push_str(&format!(
                        "{}: [Tool: {} {}]\n",
                        if self.role == Role::User {
                            "user"
                        } else {
                            "assistant"
                        },
                        tool_use.name,
                        tool_use.input
                    ));
                    has_user_messages = true; // Always show roles if there are tools
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// A piece of content in a message, either text or an image.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text { text: String },
    Image { source: Source },
    ToolUse(ToolUse),
}

impl Content {
    pub fn text(text: impl Into<String>) -> Self {
        Content::Text { text: text.into() }
    }

    pub fn image(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        let image_data = fs::read(path)?;
        let base64_image = BASE64_STANDARD.encode(image_data);

        Ok(Content::Image {
            source: Source {
                source_type: "base64".to_string(),
                media_type: Self::detect_media_type(path),
                data: base64_image,
            },
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
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// Token usage statistics for a message.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

fn is_default_tool_choice(choice: &ToolChoice) -> bool {
    *choice == ToolChoice::Auto
}

/// A request to the Anthropic API for message generation.
#[derive(Debug, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "is_default_tool_choice")]
    pub tool_choice: ToolChoice,
}

impl Default for MessagesRequest {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            messages: Vec::new(),
            system: None,
            temperature: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: ToolChoice::default(),
        }
    }
}

impl MessagesRequest {
    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn add_user(&mut self, content: Content) {
        self.add_content(Role::User, content);
    }

    pub fn add_assistant(&mut self, content: Content) {
        self.add_content(Role::Assistant, content);
    }

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
    pub role: Role,
    pub content: Vec<Content>,
}

impl Message {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            content: Vec::new(),
        }
    }

    pub fn format_nicely(&self) -> String {
        let role_prefix = match self.role {
            Role::User => "User: ",
            Role::Assistant => "Assistant: ",
        };

        let content_strings: Vec<String> = self
            .content
            .iter()
            .map(|content| match content {
                Content::Text { text } => text.clone(),
                Content::Image { source } => {
                    format!("[Image: {} {}]", source.source_type, source.media_type)
                }
                Content::ToolUse(tool_use) => {
                    format!("[Tool: {} {}]", tool_use.name, tool_use.input)
                }
            })
            .collect();

        let formatted_content = content_strings.join("\n");

        format!("{}{}", role_prefix, formatted_content)
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
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: format!("https://{}", DEFAULT_API_DOMAIN),
        }
    }

    pub fn from_env() -> Result<Self> {
        let api_key = env::var(ANTHROPIC_API_KEY_ENV)?;
        Ok(Self::new(api_key))
    }

    pub fn with_string_or_env(api_key: &str) -> Result<Self> {
        if !api_key.is_empty() {
            Ok(Self::new(api_key.to_string()))
        } else {
            Self::from_env()
        }
    }

    pub fn messages_stream(&self, request: MessagesRequest) -> Result<StreamedResponse> {
        let mut headers = HeaderMap::new();
        let request = request.with_stream(true);
        headers.insert("x-api-key", HeaderValue::from_str(&self.api_key)?);
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_API_VERSION),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let url = format!("{}/v1/messages", self.base_url);

        let event_source = EventSource::new(
            reqwest::Client::new()
                .post(&url)
                .headers(headers)
                .json(&request),
        )
        .map_err(|e| Error::EventSourceError(e.to_string()))?;

        Ok(StreamedResponse::new(event_source))
    }

    /// Sends a message request to the Anthropic API and returns the response.
    /// Uses client defaults for model and max_tokens if not specified in the request.
    pub async fn messages(&self, request: MessagesRequest) -> Result<MessagesResponse> {
        let client = reqwest::Client::new();

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_str(&self.api_key)?);
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_API_VERSION),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let url = format!("{}/v1/messages", self.base_url);
        trace!("Full request:");
        trace!("URL: {}", url);
        trace!("Headers: {:#?}", headers);
        trace!("Body: {:#?}", request);

        let response = client
            .post(url)
            .headers(headers)
            .json(&request)
            .send()
            .await?;

        let status = response.status();

        // Debug print the full response, including status and headers
        trace!("Full response:");
        trace!("Status: {}", status);
        trace!("Headers: {:#?}", response.headers());

        if status.is_success() {
            let messages_response: MessagesResponse = response.json().await?;
            trace!("Body: {:#?}", messages_response);
            Ok(messages_response)
        } else {
            let error_response: ApiErrorResponse = response.json().await?;
            trace!("Error: {:#?}", error_response);
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
        let tool = Tool::new::<TestInput>();

        // Check the basic properties
        assert_eq!(tool.name, "TestInput");
        assert_eq!(tool.description, "This is a test description");

        // Serialize the tool to JSON
        let json = serde_json::to_value(&tool).expect("Failed to serialize Tool to JSON");

        // Check the structure of the serialized JSON
        assert!(json.is_object());
        let json_obj = json.as_object().unwrap();

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
        let definitions = &schema_obj["definitions"];
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
}
