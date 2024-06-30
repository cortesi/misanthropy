use misanthropy::{Anthropic, Content, MessagesRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the Anthropic client
    let client = Anthropic::from_env()?;

    // Create a new MessagesRequest
    let mut request = MessagesRequest::default();

    let msg = "Hello, Claude! How are you today?";
    // Add a user message
    request.add_user(Content::text(msg));

    println!("Sending request: {}", msg);
    // Send the request and get the response
    let response = client.messages(&request).await?;

    // Print the formatted response
    println!("Claude's response:");
    println!("{}", response.format_nicely());

    Ok(())
}
