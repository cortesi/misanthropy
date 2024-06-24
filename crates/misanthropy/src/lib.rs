use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;

pub const DEFAULT_MODEL: &str = "claude-3-opus-20240229";
pub const DEFAULT_MAX_TOKENS: u32 = 1024;
pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const ANTHROPIC_API_DOMAIN_ENV: &str = "ANTHROPIC_API_DOMAIN";
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const DEFAULT_API_DOMAIN: &str = "api.anthropic.com";

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

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

#[derive(Debug, Serialize, Deserialize)]
pub struct Content {
    pub text: String,
    #[serde(rename = "type")]
    pub content_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
}

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

impl Anthropic {
    pub fn new(api_key: String) -> Self {
        let domain =
            env::var(ANTHROPIC_API_DOMAIN_ENV).unwrap_or_else(|_| DEFAULT_API_DOMAIN.to_string());
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
