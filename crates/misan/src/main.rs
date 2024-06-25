use std::{env, path::PathBuf};

use clap::{Args, Parser, Subcommand};
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

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Send a message to the API
    Message(MessageArgs),
    /// Stream a message from the API
    Stream(MessageArgs),
    /// Display information about the tool and API
    Info,
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
}

#[derive(Clone)]
enum MessageContent {
    UserText(String),
    AssistantText(String),
    UserImage(PathBuf),
    AssistantImage(PathBuf),
}

async fn handle_message(
    anthropic: &Anthropic,
    args: &MessageArgs,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Running Message command");
    let request = build_request(args, cli)?;

    debug!("Constructed request: {:#?}", request);

    match anthropic.messages(request).await {
        Ok(response) => {
            info!("Message sent successfully");
            println!("{}", response.format_nicely());

            if cli.verbose >= 2 {
                debug!("Full response: {:#?}", response);
            }
        }
        Err(e) => error!("Failed to send message: {}", e),
    }

    Ok(())
}

async fn handle_stream(
    anthropic: &Anthropic,
    args: &MessageArgs,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Running Stream command");
    let request = build_request(args, cli)?;

    debug!("Constructed request: {:#?}", request);

    match anthropic.messages_stream(request) {
        Ok(mut streamed_response) => {
            info!("Stream started successfully");
            while (streamed_response.next().await).is_some() {}
            println!("{}", streamed_response.response.format_nicely());
        }
        Err(e) => error!("Failed to start stream: {}", e),
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
        request = request.with_system(system);
    }

    if let Some(temp) = &args.temperature {
        request = request.with_temperature(*temp);
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
    println!("\tDefault Model: {}", DEFAULT_MODEL);
    println!("\tDefault Max Tokens: {}", DEFAULT_MAX_TOKENS);
    println!("\tAnthropic API Version: {}", ANTHROPIC_API_VERSION);
    if cli.api_key.is_some() {
        println!("\tAPI Key: Provided via command line argument");
    } else if env::var(ANTHROPIC_API_KEY_ENV).is_ok() {
        println!(
            "\tAPI Key: Detected in environment variable {}",
            ANTHROPIC_API_KEY_ENV
        );
    } else {
        println!("\tAPI Key: Not detected");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    setup_logger(cli.verbose, cli.quiet);

    let anthropic = Anthropic::with_string_or_env(cli.api_key.as_deref().unwrap_or(""))?;

    match &cli.command {
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
