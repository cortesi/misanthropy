use thiserror::Error;

use crate::{ApiErrorResponse, ApiErrorType};

pub type Result<T> = std::result::Result<T, Error>;

/// Errors encountered during Anthropic API interactions.
/// Covers API responses, HTTP issues, parsing failures, and client-side problems.
#[derive(Error, Debug)]
pub enum Error {
    #[error("API request failed: {0}")]
    ApiError(String),

    #[error("Failed to parse API response: {0}")]
    ResponseParseError(#[from] serde_json::Error),

    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("API overloaded: {0}")]
    ApiOverloaded(String),

    #[error("Unknown error: {0}")]
    UnknownError(String),

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
