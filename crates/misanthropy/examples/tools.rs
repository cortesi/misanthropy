use misanthropy::{Anthropic, Content, MessagesRequest, Tool, DEFAULT_MODEL};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Get the current stock price for a given ticker symbol.
#[derive(JsonSchema, Serialize, Deserialize, Debug)]
struct GetStockPrice {
    /// The stock ticker symbol, e.g. AAPL for Apple Inc.
    ticker: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let anthropic = Anthropic::from_env()?;
    let get_stock_price_tool = Tool::new::<GetStockPrice>();
    let mut request = MessagesRequest::default()
        .with_model(DEFAULT_MODEL.to_string())
        .with_max_tokens(1000)
        .with_tool(get_stock_price_tool)
        .with_system("You are a helpful assistant that can look up stock prices.".to_string());

    request.add_user(Content::text("What is Apple's stock price today?"));

    println!("Making request...");
    let response = anthropic.messages(&request).await?;
    println!("Claude's response:");
    println!("{}", response.format_nicely());

    // Check for tool use in the response
    println!("------------------------------------");
    for content in response.content {
        request.add_assistant(content.clone());
        if let Content::ToolUse(tool_use) = content {
            println!("Tool Use Detected:");
            println!("Tool Name: {}", tool_use.name);
            println!("Tool ID: {}", tool_use.id);

            match serde_json::from_value::<GetStockPrice>(tool_use.input.clone()) {
                Ok(get_stock_price) => {
                    println!("{:#?}", get_stock_price.ticker);
                    request.add_user(Content::tool_result(
                        &tool_use,
                        "The stock price is $150.00",
                    ));
                }
                Err(e) => {
                    eprintln!("Failed to parse tool input: {}", e);
                    return Ok(());
                }
            }
        }
    }

    // Continue the conversation and get the next response
    println!("------------------------------------");
    println!("Responded to tool, making request...");
    let response = anthropic.messages(&request).await?;
    println!("Claude's response:");
    println!("{}", response.format_nicely());

    Ok(())
}
