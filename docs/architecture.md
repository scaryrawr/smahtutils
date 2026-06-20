# Architecture

This document is for contributors who need to understand how the packages fit together. For usage instructions, start with the root README or the package README for the app you are running.

## Package map

| Package | Role |
| --- | --- |
| `smahtiepants` | DevDocs installer, Markdown cache, documentation embeddings, search, REST server, Streamable HTTP MCP server, and Copilot hook endpoints. |
| `smahties` | Local code scanner, parser, SQLite/Annoy index, MCP stdio server, CLI query commands, and duplicate-code reports. |
| `wickedpaste` | Clipboard reader plus OpenAI-compatible chat conversion to HTML and Markdown. |
| `wickedsmaht_config` | Shared config loader and CLI-over-config resolution helpers. |

## Cross-cutting design

- Keep runtime state out of the repository unless it is an intentionally ignored app state directory such as `.smahties`.
- Prefer SQLite as the authoritative data store and rebuildable sidecar indexes for acceleration.
- Keep OpenAI-compatible endpoint, model, API-key, server, and embedding settings configurable.
- Keep docs-search embeddings and code-search embeddings separate: `text_embedding_model` is for `smahtiepants`; `coding_embedding_model` is for `smahties`.
- Avoid live network, clipboard, or model dependencies in tests unless isolated with fakes or mocks.

## smahtiepants flow

```text
DevDocs source index
  -> canonical docset slug resolution
  -> downloaded DevDocs JSON
  -> local Markdown pages and manifests
  -> chunk rows in SQLite
  -> optional embedding vectors in SQLite
  -> rebuildable Annoy sidecar indexes
  -> CLI, REST, MCP, and Copilot hook search/read surfaces
```

Important constraints:

- Resolve user-facing aliases to canonical DevDocs slugs before touching cache paths.
- Keep canonical slugs as cache directory names, manifest keys, and embedding docset IDs.
- Keep `/mcp` backed by the official `mcp` SDK Streamable HTTP transport.
- Unscoped search should exact-score bounded Annoy candidates loaded from SQLite; scoped docset/language search should exact-score vectors inside the resolved scope. Keyword hits with vectors should join the semantic candidate pool, while keyword-only hits remain fallback results.
- Search output should include query-aware excerpts and full-page read hints.

## smahties flow

```text
Runtime root
  -> scanner exclusions and .gitignore filtering
  -> parser code units or fallback file units
  -> SQLite files, units, lexical rows, embeddings, queue, and status
  -> rebuildable Annoy sidecar indexes
  -> CLI and MCP semantic/keyword/hybrid query results
  -> duplicate-code JSON reports
```

Important constraints:

- Treat `.gitignore` as an indexing exclusion even for tracked files.
- Keep `.smahties/smahties.sqlite` as the single authoritative index.
- Failed claimed work should return to pending, and stale in-progress work should be reclaimable.
- Function/class duplicate detection uses stored code-unit embeddings; file-level duplicate detection may create transient whole-file embeddings for that run.

## wickedpaste flow

```text
Clipboard image, if available
  -> PNG data URL
  -> OpenAI-compatible multimodal chat request
  -> strict JSON response with html and markdown

Clipboard text, if no image is available
  -> OpenAI-compatible chat request
  -> strict JSON response with html and markdown
```

Important constraints:

- Preserve clipboard precedence: image before text.
- Do not log clipboard contents, image data URLs, or model responses.
- Keep HTML output minimal and styling-free.

## wickedsmaht_config flow

```text
CLI value
  -> shared config value from $HOME/.wickedsmaht/config.json
  -> app default when available
  -> explicit SettingError for required missing values
```

Important constraints:

- Accept snake_case and kebab-case config keys for user convenience.
- Keep legacy `ddserve` support as compatibility, not the primary documented path.
- Do not move app-specific runtime defaults into the shared package unless more than one app consumes them.
