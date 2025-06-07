use misanthropy::{tools, Anthropic, Content, MessagesRequest, DEFAULT_MODEL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let anthropic = Anthropic::from_env()?;
    make_request(&anthropic).await?;
    Ok(())
}

async fn make_request(anthropic: &Anthropic) -> Result<(), Box<dyn std::error::Error>> {
    let mut request = MessagesRequest::default()
        .with_model(DEFAULT_MODEL.to_string())
        .with_max_tokens(1000)
        .with_text_editor(misanthropy::TEXT_EDITOR_37)
        .with_system(vec![Content::text("You are a helpful editor assistant.")]);

    request.add_user(Content::text("Please view the file /bar.py"));

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

            match serde_json::from_value::<tools::TextEditor>(tool_use.input.clone()) {
                Ok(ed) => {
                    println!("{ed:#?}");
                }
                Err(e) => {
                    eprintln!("Failed to parse tool input: {e}");
                    return Ok(());
                }
            }
        }
    }

    Ok(())
}
