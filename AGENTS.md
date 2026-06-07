# Repository Guidelines

## Project Structure & Module Organization

This is a small Rust workspace. The root `Cargo.toml` is a virtual workspace with one member, `apps/wickedpaste`. Keep application code under `apps/wickedpaste/src/`: `main.rs` owns CLI parsing, OpenAI-compatible chat requests, and stdout output; `clipboard.rs` owns system clipboard reading plus image-to-PNG-data-URL conversion. Build output belongs in `target/` and is ignored.

## Build, Test, and Development Commands

Run commands from the repository root unless you intentionally scope to a package.

- `cargo run -p wickedpaste -- --base-url http://127.0.0.1:14892/v1 --model <model>` — run the clipboard converter against a local/OpenAI-compatible endpoint. `--base-url` and `--model` are required CLI arguments.
- `cargo fmt --all` — format all workspace Rust code.
- `cargo fmt --all -- --check` — narrow formatting validation.
- `cargo clippy --workspace --all-targets -- -D warnings` — lint with warnings treated as errors.
- `cargo test --workspace` — compile and run tests for all workspace members.

Current Cargo emits a resolver warning because the virtual workspace does not set `workspace.resolver` while the crate uses edition 2024; do not treat that warning as a test failure.

## Coding Style & Naming Conventions

Use Rust 2024 and rustfmt defaults. Prefer small modules with explicit ownership like the existing split between CLI/request orchestration and clipboard helpers. Keep public structs/functions documented when they define CLI/API-facing behavior. Preserve the image-first, text-fallback clipboard behavior unless the task explicitly changes product behavior.

## Testing Guidelines

There are currently no dedicated test files; `cargo test --workspace` is still the baseline compile/test check. Add unit tests next to code with `#[cfg(test)]` modules when extracting pure logic, especially for clipboard image validation or request-shaping helpers. Avoid tests that require a live clipboard or LLM endpoint unless clearly marked and isolated.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, sometimes scoped (for example, `wickedpaste: add --base-url and --model CLI arguments`). Follow that style. PRs should state the validation commands run and mention any local endpoint/clipboard assumptions used for manual testing.

## Security & Configuration Tips

Clipboard contents may be sensitive. Do not add debug logging that prints clipboard text, base64 image payloads, API responses, or secrets. Keep endpoint URLs, model names, and credentials configurable rather than hardcoded.
