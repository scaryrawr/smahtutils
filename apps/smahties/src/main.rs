use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::{Parser, Subcommand};
use smahties::{
    Result,
    embedding::OpenAiEmbedder,
    indexer::{IndexRunOutcome, IndexRunSummary, Indexer},
    mcp,
    model::{
        IndexedListResponse, Priority, QueryMatch, QueryMatchKind, QueryMode, QueryResponse,
        ServiceStatus,
    },
    parser::ParserRegistry,
    scanner::{EXCLUDED_DIR_NAMES, EXCLUDED_FILE_NAMES, Scanner, ensure_state_dir},
    service::{self, AppState, IndexRequest, ListIndexedRequest, QueryRequest},
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

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Search indexed code with semantic, keyword, or hybrid ranking.
    Query(QueryArgs),
    /// Block until indexing completes for a path, or press Ctrl+C to resume later.
    Index(IndexArgs),
    /// Show indexing status, queue counts, and recent errors.
    Status(OutputArgs),
    /// List indexed code units.
    #[command(alias = "list")]
    ListIndexed(ListIndexedArgs),
}

#[derive(Parser)]
struct QueryArgs {
    /// Natural-language or keyword query.
    query: String,
    /// Maximum number of matches to return.
    #[arg(long)]
    limit: Option<usize>,
    /// Ranking mode to use: hybrid, semantic, or keyword.
    #[arg(long, default_value = "hybrid", value_parser = parse_query_mode)]
    mode: QueryMode,
    /// Restrict matches to paths with this prefix.
    #[arg(long)]
    path_prefix: Option<String>,
    /// Restrict matches to a language name.
    #[arg(long)]
    language: Option<String>,
    /// Emit JSON instead of human-readable output.
    #[arg(long)]
    json: bool,
}

#[derive(Parser)]
struct IndexArgs {
    /// File or directory to index. Defaults to the repository root.
    #[arg(default_value = ".")]
    path: PathBuf,
    /// Emit JSON instead of human-readable output.
    #[arg(long)]
    json: bool,
}

#[derive(Parser)]
struct OutputArgs {
    /// Emit JSON instead of human-readable output.
    #[arg(long)]
    json: bool,
}

#[derive(Parser)]
struct ListIndexedArgs {
    /// Restrict items to paths with this prefix.
    #[arg(long)]
    path_prefix: Option<String>,
    /// Restrict items to a language name.
    #[arg(long)]
    language: Option<String>,
    /// Maximum number of items to return.
    #[arg(long)]
    limit: Option<usize>,
    /// Offset for pagination.
    #[arg(long)]
    offset: Option<usize>,
    /// Include source snippets when limit is 20 or lower.
    #[arg(long)]
    include_source: bool,
    /// Emit JSON instead of human-readable output.
    #[arg(long)]
    json: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing();

    match &args.command {
        Some(Command::Query(command)) => {
            let api_requirement = if matches!(command.mode, QueryMode::Keyword) {
                ApiRequirement::Optional
            } else {
                ApiRequirement::Required
            };
            let state = build_cli_state(&args, api_requirement)?;
            let response = service::query_code(
                &state,
                QueryRequest {
                    query: command.query.clone(),
                    limit: command.limit,
                    mode: Some(command.mode),
                    path_prefix: command.path_prefix.clone(),
                    language: command.language.clone(),
                },
            )
            .await?;
            print_query_response(&response, command.json)?;
            Ok(())
        }
        Some(Command::Index(command)) => {
            let state = build_cli_state(&args, ApiRequirement::Required)?;
            run_blocking_index(&state, command).await
        }
        Some(Command::Status(command)) => {
            let state = build_cli_state(&args, ApiRequirement::Optional)?;
            let response = service::status(&state).await?;
            print_status(&response, command.json)?;
            Ok(())
        }
        Some(Command::ListIndexed(command)) => {
            let state = build_cli_state(&args, ApiRequirement::Optional)?;
            let response = service::list_indexed(
                &state,
                ListIndexedRequest {
                    path_prefix: command.path_prefix.clone(),
                    language: command.language.clone(),
                    limit: command.limit,
                    offset: command.offset,
                    include_source: Some(command.include_source),
                },
            )?;
            print_list_indexed(&response, command.json)?;
            Ok(())
        }
        None => {
            let (state, _watcher) = build_mcp_state(&args).await?;

            tracing::info!("smahties MCP server is listening on stdio");
            mcp::serve(state).await
        }
    }
}

async fn build_mcp_state(args: &Args) -> Result<(AppState, notify::RecommendedWatcher)> {
    let state = build_cli_state(args, ApiRequirement::Required)?;
    state.indexer.spawn_worker();
    state
        .indexer
        .enqueue_path(state.indexer.root().to_path_buf(), Priority::Low)
        .await;
    let watcher = watcher::start(state.indexer.root(), state.indexer.clone())?;

    Ok((state, watcher))
}

#[derive(Clone, Copy)]
enum ApiRequirement {
    Required,
    Optional,
}

fn build_cli_state(args: &Args, api_requirement: ApiRequirement) -> Result<AppState> {
    let root = args.root.canonicalize()?;
    let state_dir = ensure_state_dir(&root)?;
    let (base_url, model) = resolve_api_settings(args, api_requirement)?;

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

    Ok(AppState {
        store,
        indexer,
        embedder,
    })
}

fn resolve_api_settings(args: &Args, requirement: ApiRequirement) -> Result<(String, String)> {
    let config = if args.base_url.is_some() && args.coding_embedding_model.is_some() {
        Config::default()
    } else {
        Config::load()?
    };

    if matches!(requirement, ApiRequirement::Optional) {
        return Ok((
            args.base_url
                .clone()
                .or(config.base_url)
                .unwrap_or_default(),
            args.coding_embedding_model
                .clone()
                .or(config.coding_embedding_model)
                .unwrap_or_else(|| "not configured".into()),
        ));
    }

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

async fn run_blocking_index(state: &AppState, command: &IndexArgs) -> Result<()> {
    let path = path_for_request(&command.path);
    service::index_path(state, IndexRequest { path: path.clone() }).await?;
    eprintln!("Indexing {path}. Press Ctrl+C to stop and resume later.");

    let outcome = state
        .indexer
        .run_until_idle_or_interrupt(tokio::signal::ctrl_c())
        .await?;
    let (status, summary) = index_outcome_parts(outcome);
    let queue = state.indexer.queue_stats().await;
    let store = state.store.stats()?;
    if command.json {
        println!(
            "{}",
            serde_json::json!({
                "status": status,
                "completed": summary.completed,
                "requeued": summary.requeued,
                "failed": summary.failed,
                "queue": queue,
                "store": store,
            })
        );
    } else {
        println!(
            "{status}: {} completed, {} requeued, {} failed. Queue: {} high, {} low, {} in progress. Indexed: {} files, {} units.",
            summary.completed,
            summary.requeued,
            summary.failed,
            queue.high_priority,
            queue.low_priority,
            queue.in_progress,
            store.indexed_files,
            store.indexed_units
        );
    }
    Ok(())
}

fn index_outcome_parts(outcome: IndexRunOutcome) -> (&'static str, IndexRunSummary) {
    match outcome {
        IndexRunOutcome::Complete(summary) => ("complete", summary),
        IndexRunOutcome::Interrupted(summary) => ("interrupted", summary),
    }
}

fn path_for_request(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn print_query_response(response: &QueryResponse, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(response)?);
        return Ok(());
    }

    if response.matches.is_empty() {
        println!("No matches.");
        return Ok(());
    }

    for (index, item) in response.matches.iter().enumerate() {
        println!("{}", format_query_match(index + 1, item));
    }
    Ok(())
}

fn format_query_match(index: usize, item: &QueryMatch) -> String {
    let mut output = String::new();
    let name = item
        .unit
        .name
        .as_deref()
        .map(|value| format!(" {value}"))
        .unwrap_or_default();
    let _ = writeln!(
        output,
        "{index}. {}:{}-{}{} [{} {} {} score {:.3}]",
        item.unit.file_path,
        item.unit.start_line,
        item.unit.end_line,
        name,
        item.unit.language,
        item.unit.unit_type,
        match_kind_label(item.match_kind),
        item.score
    );
    for line in item.unit.source.lines().take(12) {
        let _ = writeln!(output, "    {line}");
    }
    output
}

fn print_status(response: &ServiceStatus, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(response)?);
        return Ok(());
    }

    println!("Root: {}", response.root);
    println!("Model: {}", response.model);
    println!(
        "Queue: {} high, {} low, {} in progress",
        response.queue.high_priority, response.queue.low_priority, response.queue.in_progress
    );
    println!(
        "Indexed: {} files, {} units, {} embeddings, {} lexical units",
        response.store.indexed_files,
        response.store.indexed_units,
        response.store.embedded_units,
        response.store.lexical_units
    );
    if let Some(owner) = &response.lease.owner {
        println!(
            "Lease: {owner} until {}{}",
            response.lease.expires_at_unix.unwrap_or_default(),
            if response.lease.held_by_this_process {
                " (this process)"
            } else {
                ""
            }
        );
    } else {
        println!("Lease: none");
    }
    if !response.store.recent_errors.is_empty() {
        println!("Recent errors:");
        for error in &response.store.recent_errors {
            println!("  {}: {}", error.path, error.error);
        }
    }
    Ok(())
}

fn print_list_indexed(response: &IndexedListResponse, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(response)?);
        return Ok(());
    }

    if response.items.is_empty() {
        println!("No indexed units.");
        return Ok(());
    }

    for item in &response.items {
        let name = item
            .name
            .as_deref()
            .map(|value| format!(" {value}"))
            .unwrap_or_default();
        println!(
            "{}:{}-{}{} [{} {}]",
            item.file_path, item.start_line, item.end_line, name, item.language, item.unit_type
        );
        if let Some(source) = &item.source {
            for line in source.lines().take(12) {
                println!("    {line}");
            }
        }
    }
    println!(
        "Showing {} items from offset {}.",
        response.items.len(),
        response.offset
    );
    Ok(())
}

fn match_kind_label(kind: QueryMatchKind) -> &'static str {
    match kind {
        QueryMatchKind::Semantic => "semantic",
        QueryMatchKind::Keyword => "keyword",
        QueryMatchKind::Hybrid => "hybrid",
    }
}

fn parse_query_mode(value: &str) -> std::result::Result<QueryMode, String> {
    match value {
        "semantic" => Ok(QueryMode::Semantic),
        "keyword" => Ok(QueryMode::Keyword),
        "hybrid" => Ok(QueryMode::Hybrid),
        _ => Err("expected one of: hybrid, semantic, keyword".into()),
    }
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();
}
