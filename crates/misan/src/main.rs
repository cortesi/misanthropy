use std::{
    env,
    io::{self, Write},
    path::PathBuf,
};

use clap::{Args, Parser, Subcommand};
use colored::*;
use env_logger::Builder;
use log::{debug, error, info, LevelFilter};

use misanthropy::{
    Anthropic, Content, MessagesRequest, ANTHROPIC_API_KEY_ENV, ANTHROPIC_API_VERSION,
    DEFAULT_MAX_TOKENS, DEFAULT_MODEL,
};

fn setup_logger(verbose: u8, quiet: bool) {
    let mut builder = Builder::new();

    let log_level = if quiet {
        LevelFilter::Error
    } else {
        match verbose {
            0 => LevelFilter::Warn,
            1 => LevelFilter::Info,
            2 => LevelFilter::Debug,
            _ => LevelFilter::Trace,
        }
    };

    builder
        .format_timestamp(None)
        .filter(None, log_level)
        .init();
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[arg(long)]
    api_key: Option<String>,

    #[arg(long, default_value = misanthropy::DEFAULT_MODEL)]
    model: String,

    #[arg(long, default_value_t = misanthropy::DEFAULT_MAX_TOKENS)]
    max_tokens: u32,

    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[arg(short, long)]
    quiet: bool,

    #[arg(long, help = "Output the response as JSON")]
    json: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session
    Chat(ChatArgs),
    /// Send a message to the API
    Message(MessageArgs),
    /// Stream a message from the API
    Stream(MessageArgs),
    /// Display information about the tool and API
    Info,
}

#[derive(Args)]
struct ChatArgs {
    #[arg(short = 's', long = "system", help = "System prompt")]
    system: Option<String>,

    #[arg(long, help = "Set the temperature for the model's output")]
    temperature: Option<f32>,

    #[arg(long = "stop", help = "Set stop sequences for the model")]
    stop_sequences: Vec<String>,
}

#[derive(Args)]
struct MessageArgs {
    #[arg(short = 'u', long = "user", help = "User text message")]
    user_messages: Vec<String>,

    #[arg(short = 'a', long = "assistant", help = "Assistant text message")]
    assistant_messages: Vec<String>,

    #[arg(long = "uimg", help = "User image file path")]
    user_images: Vec<PathBuf>,

    #[arg(long = "aimg", help = "Assistant image file path")]
    assistant_images: Vec<PathBuf>,

    #[arg(short = 's', long = "system", help = "System prompt")]
    system: Option<String>,

    #[arg(long, help = "Set the temperature for the model's output")]
    temperature: Option<f32>,

    #[arg(long = "stop", help = "Set stop sequences for the model")]
    stop_sequences: Vec<String>,
}

impl MessageArgs {
    fn is_empty(&self) -> bool {
        self.user_messages.is_empty()
            && self.assistant_messages.is_empty()
            && self.user_images.is_empty()
            && self.assistant_images.is_empty()
            && self.system.is_none()
    }
}

#[derive(Clone)]
enum MessageContent {
    UserText(String),
    AssistantText(String),
    UserImage(PathBuf),
    AssistantImage(PathBuf),
}

async fn handle_chat(
    anthropic: &Anthropic,
    args: &ChatArgs,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut request = MessagesRequest::default()
        .with_model(cli.model.clone())
        .with_max_tokens(cli.max_tokens)
        .with_stream(true);

    if let Some(system) = &args.system {
        request.add_system(Content::text(system));
    }

    if let Some(temp) = &args.temperature {
        request = request.with_temperature(*temp);
    }

    if !args.stop_sequences.is_empty() {
        request = request.with_stop_sequences(args.stop_sequences.clone());
    }

    println!(
        "{}",
        "Starting chat session. Type 'exit' or ctrl-c to end the conversation.".blue()
    );

    loop {
        print!("{} ", "You:".green().bold());
        io::stdout().flush()?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;

        user_input = user_input.trim().to_string();

        if user_input.is_empty() {
            continue; // Skip this iteration if the input is empty
        }

        if user_input.to_lowercase() == "exit" {
            println!("Ending chat session.");
            break;
        }

        request.add_user(Content::text(user_input));

        match anthropic.messages_stream(&request) {
            Ok(mut streamed_response) => {
                print!("{} ", "AI:".blue().bold());
                io::stdout().flush()?;

                let mut response_content = String::new();

                while let Some(event) = streamed_response.next().await {
                    match event {
                        Ok(event) => {
                            match event {
                                misanthropy::StreamEvent::ContentBlockDelta {
                                    delta: misanthropy::ContentBlockDelta::TextDelta { text },
                                    ..
                                } => {
                                    print!("{text}");
                                    io::stdout().flush()?;
                                    response_content.push_str(&text);
                                }
                                misanthropy::StreamEvent::MessageStop => {
                                    println!(); // End the line after the full response
                                }
                                _ => {} // Ignore other event types
                            }
                        }
                        Err(e) => {
                            eprintln!("{}", "Error in stream:".red().bold());
                            eprintln!("{e}");
                            println!("\nAn error occurred. Please try again.");
                            break;
                        }
                    }
                }

                // Merge the streamed response into the request for context
                request.merge_streamed_response(&streamed_response);
            }
            Err(e) => {
                eprintln!("{}", "Failed to start stream:".red().bold());
                eprintln!("{e}");
                println!("An error occurred. Please try again.");
            }
        }
    }

    Ok(())
}

async fn handle_message(
    anthropic: &Anthropic,
    args: &MessageArgs,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        return Err("No message content provided. Please provide at least one user or assistant message, or an image.".into());
    }

    info!("Running Message command");
    let request = build_request(args, cli)?;

    debug!("Constructed request: {request:#?}");

    match anthropic.messages(&request).await {
        Ok(response) => {
            info!("Message sent successfully");
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&response)?);
            } else {
                println!("{}", response.format_content());
            }

            if cli.verbose >= 2 {
                debug!("Full response: {response:#?}");
            }
        }
        Err(e) => error!("Failed to send message: {e}"),
    }

    Ok(())
}

async fn handle_stream(
    anthropic: &Anthropic,
    args: &MessageArgs,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        return Err("No message content provided. Please provide at least one user or assistant message, or an image.".into());
    }

    info!("Running Stream command");
    let request = build_request(args, cli)?.with_stream(true);

    debug!("Constructed request: {request:#?}");

    match anthropic.messages_stream(&request) {
        Ok(mut streamed_response) => {
            info!("Stream started successfully");
            while (streamed_response.next().await).is_some() {}
            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&streamed_response.response)?
                );
            } else {
                println!("{}", streamed_response.response.format_content());
            }
        }
        Err(e) => error!("Failed to start stream: {e}"),
    }

    Ok(())
}

fn build_request(
    args: &MessageArgs,
    cli: &Cli,
) -> Result<MessagesRequest, Box<dyn std::error::Error>> {
    let mut request = MessagesRequest::default()
        .with_model(cli.model.clone())
        .with_max_tokens(cli.max_tokens);

    if let Some(system) = &args.system {
        request.add_system(Content::text(system));
    }

    if let Some(temp) = &args.temperature {
        request = request.with_temperature(*temp);
    }

    if !args.stop_sequences.is_empty() {
        request = request.with_stop_sequences(args.stop_sequences.clone());
    }

    // Collect all messages and images with their indices
    let mut messages: Vec<(usize, MessageContent)> = Vec::new();

    for (index, value) in std::env::args().enumerate() {
        match value.as_str() {
            "-u" | "--user" => {
                if let Some(text) = std::env::args().nth(index + 1) {
                    messages.push((index, MessageContent::UserText(text)));
                }
            }
            "-a" | "--assistant" => {
                if let Some(text) = std::env::args().nth(index + 1) {
                    messages.push((index, MessageContent::AssistantText(text)));
                }
            }
            "--uimg" => {
                if let Some(path) = std::env::args().nth(index + 1) {
                    messages.push((index, MessageContent::UserImage(PathBuf::from(path))));
                }
            }
            "--aimg" => {
                if let Some(path) = std::env::args().nth(index + 1) {
                    messages.push((index, MessageContent::AssistantImage(PathBuf::from(path))));
                }
            }
            _ => {}
        }
    }

    // Sort messages by their original order
    messages.sort_by_key(|&(index, _)| index);

    // Process messages in order
    for (_, content) in messages {
        match content {
            MessageContent::UserText(text) => request.add_user(Content::text(text)),
            MessageContent::AssistantText(text) => request.add_assistant(Content::text(text)),
            MessageContent::UserImage(path) => match Content::image(&path) {
                Ok(content) => request.add_user(content),
                Err(e) => {
                    error!("Failed to read user image file {}: {}", path.display(), e);
                    return Err(e.into());
                }
            },
            MessageContent::AssistantImage(path) => match Content::image(&path) {
                Ok(content) => request.add_assistant(content),
                Err(e) => {
                    error!(
                        "Failed to read assistant image file {}: {}",
                        path.display(),
                        e
                    );
                    return Err(e.into());
                }
            },
        }
    }

    Ok(request)
}

fn handle_info(cli: &Cli) {
    println!("Misan:");
    println!("\tVersion: {}", env!("CARGO_PKG_VERSION"));
    println!("\tDefault Model: {DEFAULT_MODEL}");
    println!("\tDefault Max Tokens: {DEFAULT_MAX_TOKENS}");
    println!("\tAnthropic API Version: {ANTHROPIC_API_VERSION}");
    if cli.api_key.is_some() {
        println!("\tAPI Key: Provided via command line argument");
    } else if env::var(ANTHROPIC_API_KEY_ENV).is_ok() {
        println!("\tAPI Key: Detected in environment variable {ANTHROPIC_API_KEY_ENV}");
    } else {
        println!("\tAPI Key: Not detected");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    setup_logger(cli.verbose, cli.quiet);

    let anthropic = Anthropic::from_string_or_env(cli.api_key.as_deref().unwrap_or(""))?;

    match &cli.command {
        Commands::Chat(args) => {
            handle_chat(&anthropic, args, &cli).await?;
        }
        Commands::Message(args) => {
            handle_message(&anthropic, args, &cli).await?;
        }
        Commands::Stream(args) => {
            handle_stream(&anthropic, args, &cli).await?;
        }
        Commands::Info => {
            handle_info(&cli);
        }
    }

    Ok(())
}
