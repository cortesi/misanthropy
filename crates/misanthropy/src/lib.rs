use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;

pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const ANTHROPIC_API_DOMAIN_ENV: &str = "ANTHROPIC_API_DOMAIN";
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const DEFAULT_API_DOMAIN: &str = "api.anthropic.com";

#[derive(Debug, Serialize, Deserialize)]
pub struct MessagesResponse {
    pub content: Vec<Content>,
    pub id: String,
    pub model: String,
    pub role: String,
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
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub struct Anthropic {
    api_key: String,
    base_url: String,
}

impl Anthropic {
    pub fn new(api_key: String) -> Self {
        let domain =
            env::var(ANTHROPIC_API_DOMAIN_ENV).unwrap_or_else(|_| DEFAULT_API_DOMAIN.to_string());
        let base_url = format!("https://{}", domain);
        Self { api_key, base_url }
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
