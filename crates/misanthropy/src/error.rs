use thiserror::Error;

use crate::{ApiErrorResponse, ApiErrorType};

/// Convenience type alias for Results using the crate's Error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors encountered during Anthropic API interactions.
/// Covers API responses, HTTP issues, parsing failures, and client-side problems.
#[derive(Error, Debug)]
pub enum Error {
    /// API request failed with an error message.
    #[error("API request failed: {0}")]
    ApiError(String),

    /// Failed to parse the API response JSON.
    #[error("Failed to parse API response: {0}")]
    ResponseParseError(#[from] serde_json::Error),

    /// HTTP request failed.
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    /// Rate limit exceeded for the API.
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// API is temporarily overloaded.
    #[error("API overloaded: {0}")]
    ApiOverloaded(String),

    /// An unknown error occurred.
    #[error("Unknown error: {0}")]
    UnknownError(String),

    /// Authentication or permission error.
    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    /// Invalid request format or content.
    #[error("Bad request: {0}")]
    BadRequest(String),

    /// Environment variable not found.
    #[error("Environment variable not found: {0}")]
    EnvVarError(#[from] std::env::VarError),

    /// Failed to create an event source for streaming.
    #[error("Failed to create event source: {0}")]
    EventSourceError(String),

    /// Error occurred during stream processing.
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Invalid HTTP header value.
    #[error("Invalid header value: {0}")]
    InvalidHeaderValue(#[from] reqwest::header::InvalidHeaderValue),
}

impl From<ApiErrorResponse> for Error {
    fn from(error: ApiErrorResponse) -> Self {
        match error.error.error_type {
            ApiErrorType::InvalidRequestError => Error::BadRequest(error.error.message),
            ApiErrorType::AuthenticationError => Error::Unauthorized(error.error.message),
            ApiErrorType::PermissionError => Error::Unauthorized(error.error.message),
            ApiErrorType::RateLimitError => Error::RateLimitExceeded(error.error.message),
            ApiErrorType::NotFoundError => {
                Error::UnknownError(format!("Not found: {}", error.error.message))
            }
            ApiErrorType::ApiError => Error::ApiError(error.error.message),
            ApiErrorType::OverloadedError => Error::ApiOverloaded(error.error.message),
            ApiErrorType::Other => Error::UnknownError(error.error.message),
        }
    }
}
