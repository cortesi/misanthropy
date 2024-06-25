use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("API request failed: {0}")]
    ApiError(String),

    #[error("Failed to parse API response: {0}")]
    ResponseParseError(#[from] serde_json::Error),

    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Environment variable not found: {0}")]
    EnvVarError(#[from] std::env::VarError),

    #[error("Failed to create event source: {0}")]
    EventSourceError(String),

    #[error("Stream error: {0}")]
    StreamError(String),

    #[error("Invalid header value: {0}")]
    InvalidHeaderValue(#[from] reqwest::header::InvalidHeaderValue),
}

impl From<crate::ApiErrorResponse> for Error {
    fn from(error: crate::ApiErrorResponse) -> Self {
        match error.error_type.as_str() {
            "rate_limit_error" => Error::RateLimitExceeded,
            "invalid_request_error" => Error::BadRequest(error.error.message),
            "authentication_error" => Error::Unauthorized(error.error.message),
            _ => Error::ApiError(error.error.message),
        }
    }
}
