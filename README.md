# Misanthropy

Misanthropy is a Rust client library for the Anthropic API, providing easy
access to Claude and other Anthropic models.

## Features

- Simple, idiomatic Rust interface for the Anthropic API
- Support for text and image content in messages
- Configurable client with defaults for model and token limits
- CLI tool for quick interactions with the API

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
misanthropy = "0.1.0"
```

## Usage

### Library

```rust
use misanthropy::{Anthropic, MessagesRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Anthropic::new(Some("your-api-key".to_string()));
    
    let request = MessagesRequest::new(
        "",  // Use default model
        0,   // Use default max_tokens
        "Hello, Claude!",
    );

    let response = client.messages(request).await?;
    println!("{}", response.format_nicely());

    Ok(())
}
```

### CLI

```bash
# Set your API key as an environment variable
export ANTHROPIC_API_KEY=your-api-key

# Send a message
cargo run -- message -c "Hello, Claude!"

# Use a specific model and max tokens
cargo run -- --model claude-3-opus-20240229 --max-tokens 2048 message -c "Tell me a joke"
```

## Configuration

- `ANTHROPIC_API_KEY`: Set this environment variable with your Anthropic API
  key.
- Default model and max tokens can be set when creating the `Anthropic` client
  or overridden per request.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
