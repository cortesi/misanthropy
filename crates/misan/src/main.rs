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
    #[arg(short = 'p', long = "prompt")]
    user_prompts: Vec<String>,

    #[arg(short = 'i', long = "img")]
    user_images: Vec<PathBuf>,

    #[arg(long = "assistant-prompt")]
    assistant_prompts: Vec<String>,

    #[arg(long = "assistant-img")]
    assistant_images: Vec<PathBuf>,
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
        Commands::Message(args) => {
            info!("Running Message command");
            let mut request = MessagesRequest::new(cli.model.clone(), cli.max_tokens);

            // Process user prompts and images
            for prompt in &args.user_prompts {
                request.add_user(Content::text(prompt));
            }

            for image_path in &args.user_images {
                match Content::image(image_path) {
                    Ok(content) => request.add_user(content),
                    Err(e) => {
                        error!("Failed to read image file {}: {}", image_path.display(), e);
                        return Err(e.into());
                    }
                }
            }

            // Process assistant prompts and images
            for prompt in &args.assistant_prompts {
                request.add_assistant(Content::text(prompt));
            }

            for image_path in &args.assistant_images {
                match Content::image(image_path) {
                    Ok(content) => request.add_assistant(content),
                    Err(e) => {
                        error!("Failed to read image file {}: {}", image_path.display(), e);
                        return Err(e.into());
                    }
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
