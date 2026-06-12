# Repository Guidelines

## Project Structure & Module Organization

This Python project uses `uv` and a root `pyproject.toml`:

- `src/ddserve` — DevDocs mirror/search/server app. `cli.py` dispatches `cache`, `sources`, `docs`, `embeddings`, `search`, `serve`, and `config`; `config.py` derives runtime settings from shared `wickedsmaht_config`; `cache.py` resolves `DDSERVE_CACHE_DIR`, XDG cache, and manifest/lock paths; `install.py` downloads DevDocs JSON and writes Markdown pages; `embeddings` uses SQLite as the authoritative chunk/vector store with Annoy sidecar indexes for semantic search; `server.py` exposes read-only REST, MCP, and Copilot hook endpoints.
- `src/wickedpaste` — clipboard-to-LLM converter. `cli.py` resolves CLI/config and calls the official `openai` package; `clipboard.py` reads image first, then text.
- `src/smahties` — semantic code-search/RAG MCP stdio service plus duplicate-code reporting. `cli.py` opens `<root>/.smahties/smahties.sqlite`, starts indexing, serves MCP, and exposes `smahties duplicates`; `duplicates.py` produces Codigami-compatible reports from stored embeddings. `mcp_server.py` exposes `query_code`, `index_path`, `status`, and `list_indexed`; `scanner`/`parser`/`indexer`/`watcher` handle discovery, code-unit extraction, embeddings, and live updates.
- `src/wickedsmaht_config` — shared `$HOME/.wickedsmaht/config.json` loader and CLI-over-config fallback logic. Keys: `base_url`, `model`, `text_embedding_model`, `coding_embedding_model`, plus optional `ddserve` app-specific settings (kebab-case aliases accepted).
- `tests` — unit tests for config, ddserve config/cache/install/search/server helpers, scanner/parser behavior, SQLite store behavior, Annoy indexing, duplicate detection, vector helpers, and clipboard image encoding.

## Build, Test, and Development Commands

Run from the repository root.

- `uv sync --locked --all-groups` — install locked runtime and dev dependencies.
- `uv run ruff format --check .` — fastest formatting validation; `uv run ruff format .` fixes formatting.
- `uv run ruff check .` — lint Python code.
- `uv run pytest` — run all tests.
- `uv build` — build wheel and source distributions.
- `uv run ddserve cache path` — show the DevDocs cache root; `uv run ddserve docs install <slug...>` installs DevDocs docsets; `uv run ddserve serve --host 127.0.0.1 --port 43877` serves read-only docs APIs.
- `uv run wickedpaste --base-url http://127.0.0.1:14892/v1 --model <model>` — convert current clipboard; flags may be omitted when config supplies them.
- `uv run smahties --base-url http://127.0.0.1:14892/v1 --coding-embedding-model <embedding-model>` — run MCP stdio for the current directory. `.mcp.json` invokes `uv run smahties` and relies on config.
- `uv run smahties duplicates --threshold 0.92 --level function` — find duplicate code using the existing `.smahties` index and embedding config.

## Coding Style & Naming Conventions

Use Ruff formatting defaults. Prefer stdlib modules before adding dependencies. Keep CLI/API-facing dataclasses clear and serializable. Preserve `wickedpaste` clipboard precedence (image before text) unless explicitly changing product behavior. Keep shared config defaults/resolution in `wickedsmaht_config`.

For `ddserve`, keep generated DevDocs cache content outside the repository under the configured cache root. Do not use a legacy ddserve config path or schema; derive settings from shared `wickedsmaht_config` using `base_url`, `text_embedding_model`, and optional nested `ddserve` settings. Never use `coding_embedding_model` for documentation search. Do not copy or merge repo-level Copilot plugin manifests/skills unless explicitly requested; this repo's root `.mcp.json` belongs to `smahties`. Runtime `ddserve` MCP and hook endpoints may still live in `src/ddserve`.

For `ddserve` semantic search performance, keep SQLite authoritative and Annoy rebuildable under the cache embeddings directory. Search should ask Annoy for bounded candidate chunk IDs, load those vectors from SQLite, then exact-score candidates; preserve keyword fallback and slug/language filters.

When building any FTS5 `MATCH` query from user input (e.g. the `ddserve` keyword fallback), sanitize each term to alphanumeric/`_` prefix tokens before appending `*`, as `smahties.service.build_fts_query` does. Raw terms containing FTS5 operators (`+`, `-`, `"`, `(`, `)`, `:`) raise `sqlite3.OperationalError` on common queries like `c++`; cover this with a test using such characters.

For `smahties` language support, prefer direct upstream tree-sitter grammar packages and pin them through `uv.lock`; unsupported extensions fall back to whole-file text units.

For `smahties` similarity/query performance, keep SQLite as the authoritative store and Annoy as a rebuildable sidecar index. Store vectors and metadata in SQLite, query Annoy for bounded candidates, then exact-score only those candidates plus any required lexical matches. Physical SQLite sharding only helps when queries can target a subset; unscoped semantic search still requires fanout/merge unless a vector index is added.

For `smahties duplicates`, preserve Codigami-compatible JSON output and keep `.smahties/smahties.sqlite` as the single index. Function/class levels use stored code-unit embeddings with Annoy candidate retrieval plus exact scoring; `--level file` creates transient whole-file embeddings for that run.

## Testing Guidelines

Prefer tests next to externally visible behavior in `tests/`. Existing tests cover config loading, scanner exclusions, parser extraction, embedding batching, vectors, service query helpers, SQLite store behavior, Annoy retrieval, and clipboard image encoding.

For `ddserve`, prefer temp cache roots and fake HTTP/OpenAI clients; avoid tests that require live DevDocs, live embedding endpoints, or long-running servers.

For `smahties` indexer resume/locking changes, preserve queue-backed retry semantics: failed claimed work should return to pending and stale in-progress work should be reclaimable. Avoid tests requiring live clipboard, LLM, or embedding endpoints unless isolated with mocks.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, sometimes scoped (`wickedpaste: ...`) or conventional (`feat(config): ...`). PR notes should list validation commands and local endpoint/clipboard assumptions.

## Security & Configuration Tips

Do not log clipboard contents, base64 images, API responses, embeddings, cached documentation content, or secrets. Keep endpoint URLs and model names configurable. `smahties` skips `.git`, `.smahties`, `target`, `node_modules`, `.next`, `.turbo`, large files, binary/non-UTF-8 files, common dependency lock files, and paths matching `.gitignore`; treat `.gitignore` as an indexing exclusion even for tracked files. Indexing can be expensive on large trees.
