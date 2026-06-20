# smahtiepants CLI reference

`smahtiepants` mirrors [DevDocs](https://devdocs.io/) docsets into a local cache,
indexes them with embeddings, and searches them offline. This reference covers
the command surface most useful when grounding answers in current docs. Assume
the binary is installed and on `PATH`.

Global form:

```bash
smahtiepants [--config <path>] <command> [...]
```

`--config` points at an alternate shared `wickedsmaht_config` JSON file. Normally
omit it; defaults come from `$HOME/.wickedsmaht/config.json`.

## docs

Manage the local DevDocs mirror.

```bash
smahtiepants docs available [--offline] [--json]
smahtiepants docs installed [--json]
smahtiepants docs install <slug> [<slug>…] [--force] [--offline] [--json]
smahtiepants docs update [<slug>] [--force] [--offline] [--json]
smahtiepants docs remove <slug> [--json]
```

- `available` lists docsets DevDocs offers. `--offline` uses cached metadata
  (fast, no network); without it, the list is refreshed from DevDocs. A `*`
  column marks docsets already installed.
- `installed` lists locally installed docsets with `Slug`, `Name`, `Pages`,
  `Updated`. The `Slug` value is what you pass to `--slug`.
- `install` downloads one or more docsets and (when embeddings are configured)
  builds the search index. `--force` reinstalls even if current.
- `update` refreshes installed docsets; with no slug it updates all.
- Slugs accept aliases and language-like names (`js`, `ts`, `py`, `python`,
  `c++`, `nodejs`) that resolve to canonical docsets, so you don't create
  duplicate cache entries.

## search

Semantic + keyword search over installed docs.

```bash
smahtiepants search "<query>" [--slug <slug>]… [--language <name>]… \
  [--limit <n>] [--format text|json|xml] [--json]
```

- `--slug` (repeatable) restricts to specific docsets. Strongly preferred for
  relevance.
- `--language` (repeatable) restricts to language-like docsets.
- `--limit` defaults to 10, capped at 50.
- `--format` selects output (`text` default); `--json` is shorthand for JSON.

Ranking: when embeddings are configured, semantic similarity is the primary
signal. Scoped searches (`--slug`/`--language`) exact-score candidates inside
that scope, while keyword matches are marked `hybrid` or fill fallback slots.
Results are diversified so you see one strong hit per page before repeats.

Queries can be natural language or identifier-heavy. For language/runtime
reference docs, include exact API names, attributes, flags, config keys, and
nearby concept words, for example `cfg target_arch target_os target triple`.

### Output fields

`--format json` emits `{"matches": [...]}` with **camelCase** keys. Each match
includes:

- `score` and `matchKind` (`semantic`, `keyword`, or `hybrid`).
- `docsetSlug`, `docsetName`, `pageId`, `pageTitle`, `pagePath`, `pageType`.
- `excerpt` — a compact, query-aware snippet (what `text` format prints).
- `text` — the full stored chunk (only fully visible via `--format json`).
- `readHint` and `resourceUri` — how to fetch the entire page.

Use `--format json` when an excerpt is too short; its `text` field holds the
whole chunk. If a result looks close but ambiguous, use `readHint`/`resourceUri`
to fetch the full page before deciding the docs do not cover the topic.

## embeddings

Inspect and rebuild the semantic index.

```bash
smahtiepants embeddings status [<slug>] [--json]
smahtiepants embeddings refresh <slug> [--json]   # embed new/changed chunks
smahtiepants embeddings rebuild <slug> [--json]   # re-embed everything
```

`status` reports the database path, whether embeddings are enabled/configured,
the model, installed docset/page counts, and indexed/embedded chunk counts. If
search returns only keyword matches, check `status` — semantic search needs an
embedding model configured in `wickedsmaht_config`.

## serve

Run the read-only REST + MCP + Copilot-hook server, used for full-page reads.

```bash
smahtiepants serve [--host <host>] [--port <port>]
```

Relevant endpoints:

- `GET /api/docsets` — list installed docsets.
- `GET /api/search?q=<query>&slug=<slug>&limit=<n>` — search.
- `GET /api/docsets/<slug>/pages` — list pages in a docset.
- `GET /api/docsets/<slug>/pages/<pageId>/content` — full Markdown page content.

## cache / sources / config

```bash
smahtiepants cache path          # print the docs cache root
smahtiepants sources list        # list upstream sources (DevDocs)
smahtiepants config path         # path to the shared config file
smahtiepants config show         # effective config (secrets redacted)
```

The cache lives under `SMAHTIEPANTS_CACHE_DIR`, `$XDG_CACHE_HOME/smahtiepants`,
or `~/.cache/smahtiepants`.
