# Repository Guidelines

## Project Structure & Module Organization

This Rust 2024 workspace has three members in the root `Cargo.toml`:

- `apps/wickedpaste` — clipboard-to-LLM converter. `main.rs` handles CLI/config resolution and OpenAI-compatible chat requests; `clipboard.rs` reads image first, then text. Depends on `wickedsmaht-config`.
- `apps/smahties` — local semantic code search/RAG MCP service. `main.rs` builds state and runs MCP stdio; `mcp.rs` exposes MCP tools; `scanner`/`parser`/`indexer`/`watcher` handle discovery and tree-sitter indexing; `store.rs` persists SQLite under `<root>/.smahties/`.
- `crates/wickedsmaht-config` — shared `$HOME/.wickedsmaht/config.json` loader and `ResolvableSetting` fallback trait.

## Build, Test, and Development Commands

Run from the repository root unless scoping intentionally.

- `cargo fmt --all -- --check` — fastest formatting validation; `cargo fmt --all` fixes formatting.
- `cargo clippy --workspace --all-targets -- -D warnings` — lint all packages with warnings as errors.
- `cargo test --workspace` — run all tests. For a narrow check, use filters such as `cargo test -p smahties scanner::`.
- `cargo run -p wickedpaste -- --base-url http://127.0.0.1:14892/v1 --model <model>` — convert current clipboard content; flags may be omitted only when config supplies `base_url` and `model`.
- `cargo run -p smahties -- --base-url http://127.0.0.1:14892/v1 --coding-embedding-model <embedding-model>` — run the MCP stdio server for the current directory.
- `.mcp.json` runs the MCP stdio server and relies on config for model settings.

Cargo currently warns that the virtual workspace resolver defaults to `1` despite edition 2024; do not treat that warning as a failed validation.

## Coding Style & Naming Conventions

Use rustfmt defaults. Keep CLI/API-facing structs and behavior documented. Preserve `wickedpaste` clipboard behavior (image first, text fallback) unless explicitly changing product behavior. Keep shared config defaults/resolution in `wickedsmaht-config`.

## Testing Guidelines

Prefer `#[cfg(test)]` modules next to pure logic. Existing tests cover config loading, scanner exclusions, embeddings, parser extraction, vectors, service query helpers, and SQLite store behavior. Avoid tests requiring a live clipboard, LLM endpoint, or embedding server unless clearly isolated.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, sometimes scoped (`wickedpaste: ...`) or conventional (`feat(config): ...`). PR notes should list validation commands and local endpoint/clipboard assumptions.

## Security & Configuration Tips

Do not log clipboard contents, base64 images, API responses, embeddings, or secrets. Keep endpoint URLs and model names configurable. `smahties` writes `.smahties/smahties.sqlite`, skips large/binary/excluded paths, and may be expensive on large trees.
