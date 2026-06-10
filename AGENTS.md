# Repository Guidelines

## Project Structure & Module Organization

This Rust 2024 workspace has three crates in the root `Cargo.toml`:

- `apps/wickedpaste` — clipboard-to-LLM converter. `main.rs` resolves CLI/config and calls OpenAI-compatible chat; `clipboard.rs` reads image first, then text.
- `apps/smahties` — semantic code-search/RAG MCP stdio service. `main.rs` opens `<root>/.smahties/smahties.sqlite`, starts indexing, and serves MCP. `mcp.rs` exposes `query_code`, `index_path`, `status`, and `list_indexed`; `scanner`/`parser`/`indexer`/`watcher` handle discovery, tree-sitter extraction, embeddings, and live updates.
- `crates/wickedsmaht-config` — shared `$HOME/.wickedsmaht/config.json` loader and `ResolvableSetting` fallback logic. Keys: `base_url`, `model`, `coding_embedding_model` (kebab-case aliases accepted).

## Build, Test, and Development Commands

Run from the repository root unless scoping a package.

- `cargo fmt --all -- --check` — fastest formatting validation; `cargo fmt --all` fixes formatting.
- `cargo clippy --workspace --all-targets -- -D warnings` — lint all packages with warnings as errors.
- `cargo test --workspace` — run all unit tests. Narrow examples: `cargo test -p smahties scanner::` or `cargo test -p wickedsmaht-config setting_resolution_tests::`.
- `cargo run -p wickedpaste -- --base-url http://127.0.0.1:14892/v1 --model <model>` — convert current clipboard; flags may be omitted when config supplies them.
- `cargo run -p smahties -- --base-url http://127.0.0.1:14892/v1 --coding-embedding-model <embedding-model>` — run MCP stdio for the current directory. `.mcp.json` invokes `cargo run --bin smahties` and relies on config.

Cargo currently warns that the virtual workspace resolver defaults to `1` despite edition 2024; do not treat that warning alone as validation failure.

## Coding Style & Naming Conventions

Use rustfmt defaults. Keep CLI/API-facing structs documented. Preserve `wickedpaste` clipboard precedence (image before text) unless explicitly changing product behavior. Keep shared config defaults/resolution in `wickedsmaht-config`. For `smahties` language support, update parser specs/dependencies together; unsupported extensions fall back to whole-file text units.

## Testing Guidelines

Prefer `#[cfg(test)]` modules next to pure logic. Existing tests cover config loading, scanner exclusions, embeddings, parser extraction, vectors, service query helpers, and SQLite store behavior. Avoid tests requiring live clipboard, LLM, or embedding endpoints unless isolated.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, sometimes scoped (`wickedpaste: ...`) or conventional (`feat(config): ...`). PR notes should list validation commands and local endpoint/clipboard assumptions.

## Security & Configuration Tips

Do not log clipboard contents, base64 images, API responses, embeddings, or secrets. Keep endpoint URLs and model names configurable. `smahties` skips `.git`, `.smahties`, `target`, `node_modules`, `.next`, `.turbo`, large files, and binary/non-UTF-8 files; indexing can be expensive on large trees.
