use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use env_logger::Builder;
use log::{debug, error, info, LevelFilter};

use misanthropy::{Anthropic, Content, Message, MessagesRequest, Role};

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
    builder.filter(None, log_level).init();
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
}

#[derive(Clone)]
enum MessageContent {
    UserText(String),
    AssistantText(String),
    UserImage(PathBuf),
    AssistantImage(PathBuf),
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Setup logger (assuming you have a setup_logger function)
    setup_logger(cli.verbose, cli.quiet);

    let anthropic =
        Anthropic::with_string_or_env(cli.api_key.as_deref().unwrap_or("")).map(|a| {
            a.with_model(cli.model.clone())
                .with_max_tokens(cli.max_tokens)
        })?;

    match &cli.command {
        Commands::Message(_args) => {
            info!("Running Message command");
            let mut request = MessagesRequest::new(cli.model.clone(), cli.max_tokens);

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
                            messages
                                .push((index, MessageContent::AssistantImage(PathBuf::from(path))));
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
                    MessageContent::AssistantText(text) => {
                        request.add_assistant(Content::text(text))
                    }
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

            debug!("Constructed request: {:#?}", request);

            match anthropic.messages(request).await {
                Ok(response) => {
                    info!("Message sent successfully");
                    println!("{}", response.format_nicely());

                    // Print full response if verbosity is high enough
                    if cli.verbose >= 2 {
                        debug!("Full response: {:#?}", response);
                    }
                }
                Err(e) => error!("Failed to send message: {}", e),
            }
        }
    }

    Ok(())
}
