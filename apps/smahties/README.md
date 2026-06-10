# smahties

`smahties` is a local semantic code-search and retrieval service. It runs as an MCP stdio server by default, indexes source files under the active root, stores state in `.smahties/smahties.sqlite`, and exposes tools for querying indexed code.

This is a vibe-coded personal dev utility, not a polished product. Your mileage may vary.

## Usage

Run the MCP server for the current directory:

```bash
cargo run -p smahties -- --base-url http://127.0.0.1:14892/v1 --coding-embedding-model <embedding-model>
```

The same settings can come from `$HOME/.wickedsmaht/config.json`:

```json
{
  "base_url": "http://127.0.0.1:14892/v1",
  "coding_embedding_model": "my-embedding-model"
}
```

The repository `.mcp.json` starts the server with:

```bash
cargo run --bin smahties
```

## CLI commands

`smahties` also includes direct CLI commands for indexing and inspection:

```bash
cargo run -p smahties -- index .
cargo run -p smahties -- status
cargo run -p smahties -- query "where is config loaded?"
cargo run -p smahties -- query "where is config loaded?" --mode keyword --json
cargo run -p smahties -- list-indexed --language rust --limit 20 --include-source
```

Use `--root <path>` to serve or query a different local coding directory.

## MCP tools

| Tool | Purpose |
| --- | --- |
| `query_code` | Searches indexed code with semantic, keyword, or hybrid ranking. |
| `index_path` | Queues high-priority indexing for a file or directory under the active scope. |
| `status` | Reports root, queue counts, store stats, recent errors, and indexer lease state. |
| `list_indexed` | Lists indexed code units with path, language, pagination, and optional source filters. |

## Indexing behavior

When running inside a Git repository, `smahties` stores state under the repository root and auto-indexes the runtime root. Outside a Git repository, state lives under the runtime root and auto-indexing is disabled.

The scanner skips binary or non-UTF-8 files, files larger than 512 KiB, and common generated or expensive paths such as `.git`, `.smahties`, `target`, `node_modules`, `.next`, and `.turbo`.

Supported tree-sitter languages include TypeScript, TSX, Rust, C#, C++, C, Go, Python, Bash, CSS, Java, and Ruby. Unsupported file extensions fall back to whole-file text units.

## Development

```bash
cargo test -p smahties
cargo test -p smahties scanner::
```

Avoid logging source snippets, embeddings, API responses, or secrets.
