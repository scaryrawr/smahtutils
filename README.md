# smahtutils

Utilities for turning local context into useful LLM inputs and local code-search context. This Rust 2024 workspace contains two command-line apps and one shared configuration crate.

These are mostly vibe-coded tools for personal development workflows. They work for the author's setup, but they are not polished products; your mileage may vary.

## Workspace layout

| Path | Package | Purpose |
| --- | --- | --- |
| `apps/wickedpaste` | `wickedpaste` | Reads the system clipboard, prefers images over text, and asks an OpenAI-compatible chat model to return HTML and GitHub Flavored Markdown. |
| `apps/smahties` | `smahties` | Runs a local semantic code-search MCP stdio server with indexing, tree-sitter code-unit extraction, embeddings, and keyword/hybrid query support. |
| `crates/wickedsmaht-config` | `wickedsmaht-config` | Loads shared defaults from `$HOME/.wickedsmaht/config.json` and resolves CLI values over config values. |

## Requirements

- Rust with Cargo.
- An OpenAI-compatible API endpoint for chat completions, embeddings, or both, depending on which app you run.
- Clipboard access for `wickedpaste`.

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
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Run `wickedpaste` against explicit settings:

```bash
cargo run -p wickedpaste -- --base-url http://127.0.0.1:14892/v1 --model <model>
```

Run `smahties` as an MCP stdio server:

```bash
cargo run -p smahties -- --base-url http://127.0.0.1:14892/v1 --coding-embedding-model <embedding-model>
```

The repository also includes `.mcp.json`, which starts `smahties` with `cargo run --bin smahties` and expects API settings to come from config.

## Notes

- Cargo may warn that the virtual workspace resolver defaults to `1` despite edition 2024; that warning alone is not a validation failure.
- Do not log or commit clipboard contents, base64 images, model responses, embeddings, or secrets.
- `smahties` skips common expensive or generated paths such as `.git`, `.smahties`, `target`, `node_modules`, `.next`, and `.turbo`.
