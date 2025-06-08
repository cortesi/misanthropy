use misanthropy::{Anthropic, Content, MessagesRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the Anthropic client
    let client = Anthropic::from_env()?;

    // Create a new MessagesRequest with advanced sampling parameters
    let mut request = MessagesRequest::default()
        .with_max_tokens(2048)
        .with_metadata("user-12345-demo") // Track user interactions
        .with_temperature(0.7) // Base randomness
        .with_top_k(40) // Only consider top 40 tokens
        .with_top_p(0.9); // Nucleus sampling at 90%

    let msg = "Write a creative story about a robot discovering emotions. Be imaginative!";

    // Add a user message
    request.add_user(Content::text(msg));

    println!("Sending request with advanced sampling parameters:");
    println!("- Temperature: 0.7");
    println!("- Top-k: 40");
    println!("- Top-p: 0.9");
    println!("- User ID: user-12345-demo");
    println!("\nPrompt: {msg}\n");

    // Send the request and get the response
    let response = client.messages(&request).await?;

    // Print the response
    println!("Claude's creative response:");
    println!("{}", response.format_content());

    Ok(())
}
