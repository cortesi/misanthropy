use misanthropy::{Anthropic, Content, MessagesRequest, Tool, ToolChoice, DEFAULT_MODEL};
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

    if std::env::args().any(|arg| arg == "--with-tool-choice") {
        make_request(&anthropic, true).await?;
    } else {
        make_request(&anthropic, false).await?;
    }

    Ok(())
}

async fn make_request(
    anthropic: &Anthropic,
    with_tool_choice: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let get_stock_price_tool = Tool::custom::<GetStockPrice>("stockprice")?;
    let get_stock_price_tool_name = match &get_stock_price_tool {
        Tool::Custom { name, .. } => name.clone(),
        Tool::TextEditor { name, .. } => name.clone(),
    };
    let request = MessagesRequest::default()
        .with_model(DEFAULT_MODEL.to_string())
        .with_max_tokens(1000)
        .with_tool(get_stock_price_tool)
        .with_system(vec![Content::text(
            "You are a helpful assistant that can look up stock prices.",
        )]);

    let mut request = if with_tool_choice {
        request.with_tool_choice(ToolChoice::Tool {
            name: get_stock_price_tool_name,
        })
    } else {
        request
    };
    request.add_user(Content::text("What is Apple's stock price today?"));

    println!("{}", serde_json::to_string_pretty(&request)?);

    println!("Making request...");
    let response = anthropic.messages(&request).await?;
    println!("Claude's response:");
    println!("{}", response.format_content());

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
                    eprintln!("Failed to parse tool input: {e}");
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
    println!("{}", response.format_content());

    Ok(())
}
