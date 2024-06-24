use clap::{Args, Parser, Subcommand};
use env_logger::Builder;
use log::{debug, error, info, LevelFilter};
use misanthropy::*;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[arg(long)]
    api_key: Option<String>,

    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Silence all output
    #[arg(short, long)]
    quiet: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Checks status and prints basic info
    Info(Info),
    /// Send a message to the API
    Message(Message),
}

#[derive(Args)]
struct Info {}

#[derive(Args)]
struct Message {}

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    setup_logger(cli.verbose, cli.quiet);

    let anthropic = match Anthropic::with_string_or_env(cli.api_key.as_deref().unwrap_or("")) {
        Ok(client) => client,
        Err(_) => {
            error!(
                "No API key provided and {} environment variable not set.",
                ANTHROPIC_API_KEY_ENV
            );
            std::process::exit(1);
        }
    };

    debug!("Anthropic client initialized: {:?}", anthropic);

    match &cli.command {
        Commands::Info(_) => {
            info!("Running Info command");
            println!("Anthropic client: {:?}", anthropic);
        }
        Commands::Message(_) => {
            info!("Running Message command");
            match anthropic.messages().await {
                Ok(_) => info!("Message sent successfully"),
                Err(e) => error!("Failed to send message: {}", e),
            }
        }
    }

    Ok(())
}
