# Misanthropy

[![Crates.io](https://img.shields.io/crates/v/misanthropy)](https://crates.io/crates/misanthropy)
[![docs.rs](https://img.shields.io/docsrs/misanthropy)](https://docs.rs/misanthropy)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-blue.svg?logo=rust)](https://www.rust-lang.org)

Misanthropy is set of Rust bindings for Anthropic API, providing easy access to
Claude and other Anthropic models. It consists of two main components:

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

## Advanced Features

### Using Tools

The library supports defining and using tools in conversations. Tools are
defined using the `schemars` crate to generate JSON schemas for the tool
inputs.

1. First, add `schemars` to your dependencies:

```toml
[dependencies]
schemars = "0.8"
```

2. Define your tool input structure and derive `JsonSchema`:

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Get the current weather for a location.
#[derive(JsonSchema, Serialize, Deserialize)]
struct GetWeather {
    /// The city and country, e.g., "London, UK"
    location: String,
    /// Temperature unit: "celsius" or "fahrenheit"
    unit: Option<String>,
}
```

3. Create a `Tool` from your input structure:

```rust
use misanthropy::{Anthropic, MessagesRequest, Tool};

let weather_tool = Tool::new::<GetWeather>();
```

4. Add the tool to your request:

```rust
let request = MessagesRequest::default()
    .with_tool(weather_tool)
    .with_system(vec![Content::text("You can use the GetWeather tool to check the weather.")]);
```

5. When the AI uses the tool, you can deserialize the input:

```rust
if let Some(tool_use) = response.content.iter().find_map(|content| {
    if let Content::ToolUse(tool_use) = content {
        Some(tool_use)
    } else {
        None
    }
}) {
    if tool_use.name == "GetWeather" {
        let weather_input: GetWeather = serde_json::from_value(tool_use.input.clone())?;
        println!("Weather requested for: {}", weather_input.location);
        // Here you would typically call an actual weather API
    }
}
```

This approach allows you to define strongly-typed tool inputs that the AI can
use, while also providing a way to handle the tool usage in your code.




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




