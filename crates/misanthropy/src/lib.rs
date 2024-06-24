use std::env;

pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const ANTHROPIC_API_DOMAIN_ENV: &str = "ANTHROPIC_API_DOMAIN";
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const DEFAULT_API_DOMAIN: &str = "api.anthropic.com";

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

    pub async fn messages(&self) -> Result<(), Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .send()
            .await?;

        println!("Response status: {}", response.status());

        Ok(())
    }
}
