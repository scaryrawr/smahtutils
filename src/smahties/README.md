# smahties

`smahties` is a local semantic code-search and duplicate-code tool. Use it when you want an MCP-compatible agent or a terminal command to find relevant code in a repository by meaning, keyword, path, or language.

## What it does

- Indexes a local repository or directory into `.smahties/smahties.sqlite`.
- Extracts functions, classes, and other code units for supported languages.
- Stores embeddings in SQLite and uses Annoy sidecar indexes for fast semantic candidate lookup.
- Serves MCP stdio tools for code search, indexing, status, and indexed-unit listing.
- Reports duplicate code in a Codigami-compatible JSON shape.

## Quick start

```bash
uv run smahties --base-url http://127.0.0.1:14892/v1 --coding-embedding-model <embedding-model>
```

With shared config in place, the flags can be omitted:

```bash
uv run smahties
```

Running without a subcommand starts the MCP stdio server. This repository's `.mcp.json` already starts it with:

```json
{
  "mcpServers": {
    "smahties": {
      "type": "local",
      "command": "uv",
      "args": ["run", "smahties"],
      "tools": ["*"]
    }
  }
}
```

## Configuration

`smahties` reads these shared settings from `$HOME/.wickedsmaht/config.json`:

```json
{
  "base_url": "http://127.0.0.1:14892/v1",
  "coding_embedding_model": "my-code-embedding-model"
}
```

CLI flags override config values:

```bash
uv run smahties --root /path/to/repo --base-url http://127.0.0.1:14892/v1 --coding-embedding-model <embedding-model>
```

## CLI commands

| Command | Purpose |
| --- | --- |
| `uv run smahties` | Start the MCP stdio server for the current directory. |
| `uv run smahties index [path]` | Queue and process indexing for a path under the active root. |
| `uv run smahties status` | Show root, queue, store, model, and recent error status. |
| `uv run smahties query <query>` | Search indexed code with hybrid ranking. |
| `uv run smahties query <query> --mode keyword` | Search without requiring embedding settings. |
| `uv run smahties list-indexed` | List indexed code units. |
| `uv run smahties duplicates [path...]` | Find duplicate code in indexed units. |

Most inspection commands accept `--json` for structured output. `query` also accepts `--limit`, `--path-prefix`, `--language`, and `--mode semantic|keyword|hybrid`.

## MCP tools

| Tool | Purpose |
| --- | --- |
| `query_code_tool` | Search indexed code by semantic, keyword, or hybrid ranking. |
| `index_path_tool` | Request high-priority indexing for a file or directory. |
| `status_tool` | Return indexing, store, queue, and lease status. |
| `list_indexed_tool` | List indexed units with optional filters and source snippets. |

## Duplicate-code reports

```bash
uv run smahties duplicates
uv run smahties duplicates src tests --threshold 0.82
uv run smahties duplicates --level function,class --output duplicates.json
uv run smahties duplicates --level file
```

Options:

| Option | Meaning |
| --- | --- |
| `path` | One or more files/directories under the active root; defaults to `.`. |
| `--threshold`, `-t` | Cosine similarity threshold from `0.0` to `1.0`; defaults to `0.92`. |
| `--level`, `-l` | Compare `function`, `class`, or `file`; repeat or comma-separate values. |
| `--language` | Restrict comparisons to a parser language such as `python`, `typescript`, or `rust`. |
| `--output`, `-o` | Write the JSON report to a file instead of stdout. |

## How indexing works

When the root is inside a Git repository, `smahties` stores state under that repository's `.smahties` directory and auto-indexes the runtime root. Outside a Git repository, state lives under the runtime root and auto-indexing is disabled until paths are explicitly indexed.

The scanner skips binary or non-UTF-8 files, files larger than 512 KiB, common generated/dependency directories, common lock files, and paths ignored by `.gitignore`.

Python uses the standard library AST parser. Tree-sitter-backed extraction is available for TypeScript, TSX, Rust, C, C++, C#, Go, Bash, CSS, Java, and Ruby. Unsupported extensions fall back to whole-file text units.

SQLite remains the source of truth for files, code units, embeddings, FTS data, queue state, and status. Annoy indexes are rebuildable sidecars under `.smahties/annoy/`.

## Safety notes

Do not commit `.smahties`, embeddings, model responses, or indexed private source snippets. Treat search output as local development context, not as data safe to paste into external systems.
