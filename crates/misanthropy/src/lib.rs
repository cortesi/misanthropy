//! Rust client for the Anthropic API.
use std::{env, error::Error, fs, path::Path};

use base64::prelude::*;
use futures_util::StreamExt;
use log::trace;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};

pub const DEFAULT_MODEL: &str = "claude-3-opus-20240229";
pub const DEFAULT_MAX_TOKENS: u32 = 1024;
pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const DEFAULT_API_DOMAIN: &str = "api.anthropic.com";

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

pub struct StreamedResponse {
    pub inner: MessagesResponse,
    event_source: Option<EventSource>,
}

impl StreamedResponse {
    pub fn new(event_source: EventSource) -> Self {
        Self {
            inner: MessagesResponse::default(),
            event_source: Some(event_source),
        }
    }

    pub async fn next(&mut self) -> Option<Result<StreamEvent, Box<dyn Error>>> {
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
                    Err(e) => return Some(Err(Box::new(e))),
                },
                Err(e) => return Some(Err(Box::new(e))),
            }
        }

        // If we've reached this point, the event_source has been exhausted
        self.event_source = None;
        None
    }

    fn merge_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::MessageStart { message } => {
                self.inner.id = message.id.clone();
                self.inner.model = message.model.clone();
                self.inner.role = message.role.clone();
                self.inner.content = message.content.clone();
                self.inner.stop_reason = message.stop_reason.clone();
                self.inner.stop_sequence = message.stop_sequence.clone();
                self.inner.usage = message.usage.clone();
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                if self.inner.content.len() <= *index {
                    self.inner.content.push(content_block.clone());
                }
            }
            StreamEvent::ContentBlockDelta { index, delta } => {
                if let Some(block) = self.inner.content.get_mut(*index) {
                    match block {
                        Content::Text { text } => match delta {
                            ContentBlockDelta::TextDelta { text: delta_text } => {
                                text.push_str(delta_text);
                            }
                            ContentBlockDelta::InputJsonDelta { .. } => {}
                        },
                        Content::Image { .. } => {}
                    }
                }
            }
            StreamEvent::MessageDelta { delta, usage } => {
                self.inner.stop_reason = delta.stop_reason.clone();
                self.inner.stop_sequence = delta.stop_sequence.clone();
                self.inner.usage = usage.clone();
            }
            StreamEvent::ContentBlockStop { .. } | StreamEvent::Ping | StreamEvent::MessageStop => {
            }
        }
    }

    pub fn is_complete(&self) -> bool {
        self.inner.stop_reason.is_some()
    }

    pub fn content_text(&self) -> String {
        self.inner
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

/// A piece of content in a message, either text or an image.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Content {
    Text { text: String },
    Image { source: Source },
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
        }
    }
}

impl MessagesRequest {
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

    pub fn from_env() -> Result<Self, env::VarError> {
        let api_key = env::var(ANTHROPIC_API_KEY_ENV)?;
        Ok(Self::new(api_key))
    }

    pub fn with_string_or_env(api_key: &str) -> Result<Self, env::VarError> {
        if !api_key.is_empty() {
            Ok(Self::new(api_key.to_string()))
        } else {
            Self::from_env()
        }
    }

    pub fn messages_stream(
        &self,
        request: MessagesRequest,
    ) -> Result<StreamedResponse, Box<dyn std::error::Error>> {
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
        )?;

        Ok(StreamedResponse::new(event_source))
    }

    /// Sends a message request to the Anthropic API and returns the response.
    /// Uses client defaults for model and max_tokens if not specified in the request.
    pub async fn messages(
        &self,
        request: MessagesRequest,
    ) -> Result<MessagesResponse, Box<dyn std::error::Error>> {
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
            Err(format!("API request failed with status: {}", status).into())
        }
    }
}
