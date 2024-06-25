use misanthropy::{Anthropic, Content, MessagesRequest, Tool, DEFAULT_MODEL};
use schemars::JsonSchema;

/// Get the current stock price for a given ticker symbol.
#[derive(JsonSchema)]
struct GetStockPrice {
    /// The stock ticker symbol, e.g. AAPL for Apple Inc.
    ticker: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the Anthropic client
    let anthropic = Anthropic::from_env()?;

    // Create the tool
    let get_stock_price_tool = Tool::new::<GetStockPrice>();

    // Create the request
    let request = MessagesRequest::default()
        .with_model(DEFAULT_MODEL.to_string())
        .with_max_tokens(1000)
        .with_tool(get_stock_price_tool)
        .with_system("You are a helpful assistant that can look up stock prices.".to_string());

    // Add the user's question
    let mut request = request;
    request.add_user(Content::text("What is Apple's stock price today?"));

    // Send the request
    let response = anthropic.messages(request).await?;

    // Print the response
    println!("Claude's response:");
    println!("{}", response.format_nicely());

    Ok(())
}
