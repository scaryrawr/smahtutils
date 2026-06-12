# Repository Guidelines

## Project Structure & Module Organization

This Python project uses `uv` and a root `pyproject.toml`:

- `src/wickedpaste` — clipboard-to-LLM converter. `cli.py` resolves CLI/config and calls the official `openai` package; `clipboard.py` reads image first, then text.
- `src/smahties` — semantic code-search/RAG MCP stdio service. `cli.py` opens `<root>/.smahties/smahties.sqlite`, starts indexing, and serves MCP. `mcp_server.py` exposes `query_code`, `index_path`, `status`, and `list_indexed`; `scanner`/`parser`/`indexer`/`watcher` handle discovery, code-unit extraction, embeddings, and live updates.
- `src/wickedsmaht_config` — shared `$HOME/.wickedsmaht/config.json` loader and CLI-over-config fallback logic. Keys: `base_url`, `model`, `coding_embedding_model` (kebab-case aliases accepted).
- `tests` — unit tests for config, scanner/parser behavior, SQLite store behavior, Annoy indexing, vector helpers, and clipboard image encoding.

## Build, Test, and Development Commands

Run from the repository root.

- `uv sync --locked --all-groups` — install locked runtime and dev dependencies.
- `uv run ruff format --check .` — fastest formatting validation; `uv run ruff format .` fixes formatting.
- `uv run ruff check .` — lint Python code.
- `uv run pytest` — run all tests.
- `uv build` — build wheel and source distributions.
- `uv run wickedpaste --base-url http://127.0.0.1:14892/v1 --model <model>` — convert current clipboard; flags may be omitted when config supplies them.
- `uv run smahties --base-url http://127.0.0.1:14892/v1 --coding-embedding-model <embedding-model>` — run MCP stdio for the current directory. `.mcp.json` invokes `uv run smahties` and relies on config.

## Coding Style & Naming Conventions

Use Ruff formatting defaults. Prefer stdlib modules before adding dependencies. Keep CLI/API-facing dataclasses clear and serializable. Preserve `wickedpaste` clipboard precedence (image before text) unless explicitly changing product behavior. Keep shared config defaults/resolution in `wickedsmaht_config`.

For `smahties` language support, prefer direct upstream tree-sitter grammar packages and pin them through `uv.lock`; unsupported extensions fall back to whole-file text units.

For `smahties` similarity/query performance, keep SQLite as the authoritative store and Annoy as a rebuildable sidecar index. Store vectors and metadata in SQLite, query Annoy for bounded candidates, then exact-score only those candidates plus any required lexical matches. Physical SQLite sharding only helps when queries can target a subset; unscoped semantic search still requires fanout/merge unless a vector index is added.

## Testing Guidelines

Prefer tests next to externally visible behavior in `tests/`. Existing tests cover config loading, scanner exclusions, parser extraction, embedding batching, vectors, service query helpers, SQLite store behavior, Annoy retrieval, and clipboard image encoding.

For `smahties` indexer resume/locking changes, preserve queue-backed retry semantics: failed claimed work should return to pending and stale in-progress work should be reclaimable. Avoid tests requiring live clipboard, LLM, or embedding endpoints unless isolated with mocks.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, sometimes scoped (`wickedpaste: ...`) or conventional (`feat(config): ...`). PR notes should list validation commands and local endpoint/clipboard assumptions.

## Security & Configuration Tips

Do not log clipboard contents, base64 images, API responses, embeddings, or secrets. Keep endpoint URLs and model names configurable. `smahties` skips `.git`, `.smahties`, `target`, `node_modules`, `.next`, `.turbo`, large files, and binary/non-UTF-8 files; indexing can be expensive on large trees.
