use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use smahties::{
    Result,
    embedding::OpenAiEmbedder,
    indexer::Indexer,
    mcp,
    model::Priority,
    parser::ParserRegistry,
    scanner::{EXCLUDED_DIR_NAMES, EXCLUDED_FILE_NAMES, Scanner, ensure_state_dir},
    service::AppState,
    store::Store,
    watcher,
};
use wickedsmaht_config::{Config, ResolvableSetting};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Repository or local coding directory to index.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// Deprecated; smahties always runs as an MCP stdio server.
    #[arg(long = "mcp", hide = true)]
    _mcp: bool,

    /// OpenAI-compatible API base URL.
    #[arg(long)]
    base_url: Option<String>,

    /// Coding embedding model to use.
    #[arg(long)]
    coding_embedding_model: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing();
    let (state, _watcher) = build_state(&args).await?;

    tracing::info!("smahties MCP server is listening on stdio");
    mcp::serve(state).await
}

async fn build_state(args: &Args) -> Result<(AppState, notify::RecommendedWatcher)> {
    let root = args.root.canonicalize()?;
    let state_dir = ensure_state_dir(&root)?;
    let (base_url, model) = resolve_api_settings(args)?;

    let store = Arc::new(Store::open(state_dir.join("smahties.sqlite"))?);
    for excluded_dir in EXCLUDED_DIR_NAMES {
        store.delete_path_prefix(excluded_dir)?;
    }
    for excluded_file in EXCLUDED_FILE_NAMES {
        store.delete_file_name(excluded_file)?;
    }
    let scanner = Scanner::new(root.clone());
    let parser = ParserRegistry;
    let embedder = OpenAiEmbedder::new(&base_url, model);
    let indexer = Indexer::new(scanner, parser, Arc::clone(&store), embedder.clone());
    indexer.spawn_worker();
    indexer.enqueue_path(root.clone(), Priority::Low).await;
    let watcher = watcher::start(&root, indexer.clone())?;

    Ok((
        AppState {
            store,
            indexer,
            embedder,
        },
        watcher,
    ))
}

fn resolve_api_settings(args: &Args) -> Result<(String, String)> {
    let config = if args.base_url.is_some() && args.coding_embedding_model.is_some() {
        Config::default()
    } else {
        Config::load()?
    };

    let base_url = String::resolve(
        args.base_url.clone(),
        config.base_url,
        "--base-url",
        "base_url",
    )?;
    let model = String::resolve(
        args.coding_embedding_model.clone(),
        config.coding_embedding_model,
        "--coding-embedding-model",
        "coding_embedding_model",
    )?;

    Ok((base_url, model))
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();
}
