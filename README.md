# smahtutils

Utilities for turning local context into useful LLM inputs and local code-search context. This Python project contains two command-line apps plus shared configuration helpers.

These are mostly vibe-coded tools for personal development workflows. They work for the author's setup, but they are not polished products; your mileage may vary.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/wickedpaste` | Reads the system clipboard, prefers images over text, and asks an OpenAI-compatible chat model to return HTML and GitHub Flavored Markdown. |
| `src/smahties` | Runs a local semantic code-search MCP stdio server with indexing, code-unit extraction, embeddings, Annoy vector search, and keyword/hybrid query support. |
| `src/wickedsmaht_config` | Loads shared defaults from `$HOME/.wickedsmaht/config.json` and resolves CLI values over config values. |
| `tests` | Unit tests for config, scanner/parser behavior, vector helpers, SQLite store behavior, Annoy indexing, and clipboard image encoding. |

## Requirements

- Python 3.11 or newer.
- `uv` for dependency locking and command execution.
- An OpenAI-compatible API endpoint for chat completions, embeddings, or both, depending on which app you run.
- Clipboard access for `wickedpaste`.

Runtime dependencies are intentionally narrow and pinned through `uv.lock`: the official `openai` package, Spotify `annoy`, the official `mcp` package, direct tree-sitter grammar packages, and `pillow` for image clipboard support.

## Shared configuration

Both apps can read defaults from `$HOME/.wickedsmaht/config.json`:

```json
{
  "base_url": "http://127.0.0.1:14892/v1",
  "model": "my-chat-model",
  "coding_embedding_model": "my-embedding-model"
}
```

CLI flags override config values. Kebab-case aliases such as `base-url` and `coding-embedding-model` are accepted in the config file.

## Common commands

Run these from the repository root:

```bash
uv sync --locked --all-groups
uv run ruff format --check .
uv run ruff check .
uv run pytest
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
```

Use `--root <path>` to serve or query a different local coding directory.

## `smahties` indexing and search

When running inside a Git repository, `smahties` stores state under the repository root and auto-indexes the runtime root. Outside a Git repository, state lives under the runtime root and auto-indexing is disabled.

The scanner skips binary or non-UTF-8 files, files larger than 512 KiB, and common generated or expensive paths such as `.git`, `.smahties`, `target`, `node_modules`, `.next`, and `.turbo`.

SQLite remains the authoritative store for files, code units, embeddings, FTS keyword search, work queue, leases, and status. Annoy indexes are rebuildable sidecar files under `.smahties/annoy/`, keyed by embedding model and dimension. Semantic queries ask Annoy for a bounded candidate set and exact-score only those candidates, avoiding full brute-force scans over all stored embeddings.

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
