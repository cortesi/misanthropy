//! Rust client for the Anthropic API.
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;

pub const DEFAULT_MODEL: &str = "claude-3-opus-20240229";
pub const DEFAULT_MAX_TOKENS: u32 = 1024;
pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const DEFAULT_API_DOMAIN: &str = "api.anthropic.com";

/// The role of a participant in a conversation. Can be either a user or an assistant.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// The response from the Anthropic API for a message request. Contains generated content, message
/// metadata, and usage statistics.
#[derive(Debug, Serialize, Deserialize)]
pub struct MessagesResponse {
    pub content: Vec<Content>,
    pub id: String,
    pub model: String,
    pub role: Role,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
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
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Content {
    Text { text: String },
    Image { source: Source },
}

/// Metadata for an image in a message.
#[derive(Debug, Serialize, Deserialize)]
pub struct Source {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// Token usage statistics for a message.
#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// A request to the Anthropic API for message generation.
#[derive(Debug, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
}

/// A single message in a conversation, with a role and content.
#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug)]
pub struct Anthropic {
    api_key: String,
    base_url: String,
    model: String,
    max_tokens: u32,
}

/// Client for interacting with the Anthropic API.
/// Manages authentication and default parameters for requests.
impl Anthropic {
    /// Creates a new Anthropic client with an optional API key.
    /// Uses default values for model and max_tokens.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: format!("https://{}", DEFAULT_API_DOMAIN),
            model: DEFAULT_MODEL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
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

    /// Sends a message request to the Anthropic API and returns the response.
    /// Uses client defaults for model and max_tokens if not specified in the request.
    pub async fn messages(
        &self,
        request: MessagesRequest,
    ) -> Result<MessagesResponse, Box<dyn std::error::Error>> {
        let request = MessagesRequest {
            model: if request.model.is_empty() {
                self.model.clone()
            } else {
                request.model
            },
            max_tokens: if request.max_tokens == 0 {
                self.max_tokens
            } else {
                request.max_tokens
            },
            messages: request.messages,
        };

        let client = reqwest::Client::new();

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_str(&self.api_key)?);
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_API_VERSION),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let response = client
            .post(format!("{}/v1/messages", self.base_url))
            .headers(headers)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if status.is_success() {
            let messages_response: MessagesResponse = response.json().await?;
            Ok(messages_response)
        } else {
            Err(format!("API request failed with status: {}", status).into())
        }
    }
}
