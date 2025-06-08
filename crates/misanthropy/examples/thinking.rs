use misanthropy::{Anthropic, Content, ContentBlockDelta, MessagesRequest, StreamEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the Anthropic client
    let client = Anthropic::from_env()?;

    // Create a new MessagesRequest with thinking enabled and streaming
    // We allocate 1024 tokens for the thinking process (minimum required)
    let mut request = MessagesRequest::default()
        .with_thinking(1024)
        .with_max_tokens(2048)
        .with_stream(true);

    let msg = "Can you solve this step by step? What is 453 + 897? Please show your work.";

    // Add a user message
    request.add_user(Content::text(msg));

    println!("Sending streaming request with thinking enabled: {msg}");
    println!("\n--- Thinking Process ---");

    // Send the request and get the streaming response
    let mut stream = client.messages_stream(&request)?;

    let mut thinking_started = false;
    let mut response_started = false;

    // Process the stream
    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::ContentBlockStart {
                index: _,
                content_block,
            }) => match &content_block {
                Content::Thinking(_) => {
                    thinking_started = true;
                }
                Content::Text(_) => {
                    if thinking_started && !response_started {
                        println!("\n\n--- Response ---");
                        response_started = true;
                    }
                }
                _ => {}
            },
            Ok(StreamEvent::ContentBlockDelta { index: _, delta }) => match &delta {
                ContentBlockDelta::ThinkingDelta { thinking } => {
                    print!("{}", thinking);
                    use std::io::{self, Write};
                    io::stdout().flush()?;
                }
                ContentBlockDelta::TextDelta { text } => {
                    print!("{}", text);
                    use std::io::{self, Write};
                    io::stdout().flush()?;
                }
                _ => {}
            },
            Ok(StreamEvent::MessageStop) => {
                println!("\n");
                break;
            }
            Err(e) => {
                eprintln!("Error in stream: {}", e);
                break;
            }
            _ => {}
        }
    }

    // Print the complete assembled message
    println!("\n--- Complete Assembled Message ---");
    println!("{}", stream.response.format_content());

    Ok(())
}
