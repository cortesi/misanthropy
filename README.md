# Misanthropy

[![Crates.io](https://img.shields.io/crates/v/misanthropy)](https://crates.io/crates/misanthropy)
[![docs.rs](https://img.shields.io/docsrs/misanthropy)](https://docs.rs/misanthropy)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-blue.svg?logo=rust)](https://www.rust-lang.org)

Misanthropy is a Rust project for interacting with the Anthropic API, providing easy access to Claude and other Anthropic models. It consists of two main components:

1. `misanthropy`: A Rust client library for the Anthropic API
2. `misan`: A command-line interface (CLI) tool for quick interactions with the API


## Features

- Simple, idiomatic Rust interface for the Anthropic API
- Support for text and image content in messages
- Support for streaming real-time responses
- Configurable client with defaults for model and token limits
- CLI tool for quick interactions with the API from the command line


## Usage

### Library

Here's a basic example of using the `misanthropy` library:

```rust
use misanthropy::{Anthropic, MessagesRequest, Content};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Anthropic::from_env()?;
    
    let mut request = MessagesRequest::default();
    request.add_user(Content::text("Hello, Claude!"));

    let response = client.messages(request).await?;
    println!("{}", response.format_nicely());

    Ok(())
}
```

For more examples, please check the [`examples`](./crates/misanthropy/examples)
directory in the `misanthropy` crate. These examples demonstrate various
features and use cases of the library. 


### CLI

The `misan` CLI tool provides a command-line interface for interacting with the
Anthropic API. For usage instructions, run:

```bash
misan --help
```

## Configuration

- `ANTHROPIC_API_KEY`: Set this environment variable with your Anthropic API key.
- Default model and max tokens can be set when creating the `Anthropic` client or overridden per request.

## Advanced Features

### Streaming Responses

The library supports streaming responses for real-time interactions:

```rust
let mut stream = client.messages_stream(request)?;

while let Some(event) = stream.next().await {
    match event {
        Ok(event) => {
            // Handle the streaming event
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### Using Tools

The library supports defining and using tools in conversations:

```rust
let weather_tool = Tool::new::<GetWeather>();
let request = MessagesRequest::default()
    .with_tool(weather_tool)
    .with_system("You can use the GetWeather tool to check the weather.");
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




