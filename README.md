# smahtutils

Utilities for turning local context into useful LLM inputs and local code-search context. This Python project contains command-line apps plus shared configuration helpers.

These are mostly vibe-coded tools for personal development workflows. They work for the author's setup, but they are not polished products; your mileage may vary.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/smahtiepants` | Mirrors DevDocs docsets into a local Markdown cache, indexes docs with embeddings, searches cached docs, and serves read-only REST/MCP/hook endpoints. |
| `src/wickedpaste` | Reads the system clipboard, prefers images over text, and asks an OpenAI-compatible chat model to return HTML and GitHub Flavored Markdown. |
| `src/smahties` | Runs a local semantic code-search MCP stdio server with indexing, code-unit extraction, embeddings, Annoy vector search, keyword/hybrid query support, and duplicate-code reports. |
| `src/wickedsmaht_config` | Loads shared defaults from `$HOME/.wickedsmaht/config.json` and resolves CLI values over config values. |
| `tests` | Unit tests for config, scanner/parser behavior, vector helpers, SQLite store behavior, Annoy indexing, and clipboard image encoding. |

## Requirements

- Python 3.11 or newer.
- `uv` for dependency locking and command execution.
- An OpenAI-compatible API endpoint for chat completions, embeddings, or both, depending on which app you run.
- Clipboard access for `wickedpaste`.
- Network access to DevDocs for installing or updating `smahtiepants` docsets.

Runtime dependencies are intentionally narrow and pinned through `uv.lock`: the official `openai` package, Spotify `annoy`, the official `mcp` package, direct tree-sitter grammar packages, and `pillow` for image clipboard support.

## Shared configuration

Both apps can read defaults from `$HOME/.wickedsmaht/config.json`:

```json
{
  "base_url": "http://127.0.0.1:14892/v1",
  "model": "my-chat-model",
  "text_embedding_model": "my-text-embedding-model",
  "coding_embedding_model": "my-code-embedding-model",
  "smahtiepants": {
    "embeddings": {
      "enabled": true,
      "batch_size": 64,
      "max_chunk_chars": 2400,
      "overlap_chars": 200
    },
    "serve": {
      "bind_address": "127.0.0.1",
      "port": 43877
    }
  }
}
```

CLI flags override config values. Kebab-case aliases such as `base-url`, `text-embedding-model`, `coding-embedding-model`, and `smahtiepants.embeddings.max-chunk-chars` are accepted in the config file.

## Common commands

Run these from the repository root:

```bash
uv sync --locked --all-groups
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Run `smahtiepants` against the default cache:

```bash
uv run smahtiepants cache path
uv run smahtiepants docs available
uv run smahtiepants docs install http css
uv run smahtiepants embeddings refresh http
uv run smahtiepants search "request headers" --slug http
uv run smahtiepants serve --host 127.0.0.1 --port 43877
```

Run `wickedpaste` against explicit settings:

```bash
uv run wickedpaste --base-url http://127.0.0.1:14892/v1 --model <model>
```

Run `smahties` as an MCP stdio server:

```bash
uv run smahties --base-url http://127.0.0.1:14892/v1 --coding-embedding-model <embedding-model>
```

The repository also includes `.mcp.json`, which starts `smahties` with `uv run smahties` and expects API settings to come from config.

## `smahties` CLI commands

```bash
uv run smahties index .
uv run smahties status
uv run smahties query "where is config loaded?"
uv run smahties query "where is config loaded?" --mode keyword --json
uv run smahties list-indexed --language python --limit 20 --include-source
uv run smahties duplicates src --threshold 0.85 --level function,class
```

Use `--root <path>` to serve or query a different local coding directory.

## `smahtiepants` behavior

`smahtiepants` stores its cache under `SMAHTIEPANTS_CACHE_DIR`, `$XDG_CACHE_HOME/smahtiepants`, or `~/.cache/smahtiepants`. On first use, an existing default `ddserve` cache directory is renamed to `smahtiepants` when the new cache directory does not already exist. Configuration is read from shared `$HOME/.wickedsmaht/config.json`; `--config` may point at an alternate shared config file. Documentation embeddings use `base_url` and `text_embedding_model`, never `coding_embedding_model`. App-specific settings live under the optional `smahtiepants` object; a legacy `ddserve` object is rewritten to `smahtiepants` when no `smahtiepants` object is present.

The Python app supports the core runtime/server surface from the original TypeScript app: DevDocs source listing, install/update/remove, Markdown page extraction, embedding refresh/rebuild/status, Annoy-accelerated semantic search with keyword fallback, a read-only REST API, a minimal MCP HTTP endpoint, and a Copilot `sessionStart` hook endpoint. Repo-level Copilot plugin manifests are intentionally not installed into this repository so they do not conflict with the existing `smahties` MCP configuration.

DevDocs aliases, common shorthands, and language-like docset names/types resolve to canonical cached docsets where possible, so inputs such as `js`, `ts`, `py`, `python`, `c++`, and `nodejs` can be used with install, update, remove, search filters, page reads, and embedding commands without creating duplicate cache entries.

## `smahties` indexing and search

When running inside a Git repository, `smahties` stores state under the repository root and auto-indexes the runtime root. Outside a Git repository, state lives under the runtime root and auto-indexing is disabled.

The scanner skips binary or non-UTF-8 files, files larger than 512 KiB, and common generated or expensive paths such as `.git`, `.smahties`, `target`, `node_modules`, `.next`, and `.turbo`.

SQLite remains the authoritative store for files, code units, embeddings, FTS keyword search, work queue, leases, and status. Annoy indexes are rebuildable sidecar files under `.smahties/annoy/`, keyed by embedding model and dimension. Semantic queries ask Annoy for a bounded candidate set and exact-score only those candidates, avoiding full brute-force scans over all stored embeddings.

## `smahties duplicates`

`smahties duplicates` migrates Codigami-style duplicate-code detection into the existing Python index. It queues indexing for the requested path or paths, uses the Annoy sidecar to find bounded nearest-neighbor candidates for stored function/class embeddings, exact-scores candidate pairs above the threshold, and prints a Codigami-compatible JSON report. File-level comparisons use transient whole-file embeddings for the run.

```bash
uv run smahties duplicates
uv run smahties duplicates src tests --threshold 0.82
uv run smahties duplicates --level function,class --output duplicates.json
uv run smahties duplicates --level file
```

Options:

- `path` — one or more files/directories under the active `--root`; defaults to `.`.
- `--threshold`, `-t` — cosine similarity threshold from `0.0` to `1.0`; defaults to `0.92`.
- `--level`, `-l` — `function`, `class`, or `file`; repeat or comma-separate values. Defaults to `function`.
- `--language` — restrict comparisons to a stored parser language such as `python`, `typescript`, or `rust`.
- `--output`, `-o` — write the JSON report to a file instead of stdout.

## `wickedpaste` behavior

`wickedpaste` checks for image data before text. Images are encoded as PNG data URLs and sent as multimodal chat content. If neither image nor text can be read, the command exits without output.

The output is expected to match this shape:

```json
{
  "html": "<p>Converted content</p>",
  "markdown": "Converted content"
}
```

## Notes

- Do not log or commit clipboard contents, base64 images, model responses, embeddings, or secrets.
