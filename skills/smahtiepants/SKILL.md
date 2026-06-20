---
name: smahtiepants
description: Use this skill before answering questions or writing code that depends on a specific library, framework, language, runtime, or CLI tool (e.g. React, Express, Node, Python, Rust, CSS, Docker, Git, Bun). Your training data has a cutoff and APIs change, get deprecated, or gain new features after it. smahtiepants searches a local, up-to-date mirror of DevDocs so you can ground answers in current documentation instead of relying on potentially stale memory. Use it whenever you are unsure about an API signature, default, option name, version-specific behavior, or "is this still the right way to do X", and whenever the user asks how to use a library or what a function/flag does.
---

# smahtiepants: search current local documentation

Your training data has a cutoff; library/framework/language APIs change after it.
**Before answering from memory, search the local docs and ground your answer in
them.** `smahtiepants` keeps a recently-mirrored copy of DevDocs (hundreds of
docsets) with fast semantic + keyword search. Assume it is installed and on
`PATH`. Skip only when the question is conceptual/version-independent or no
relevant docset exists.

## Workflow

### 1. See what documentation is available locally

```bash
smahtiepants docs installed
```

This lists installed docsets with their canonical `Slug` (e.g. `javascript`,
`react`, `express`, `python~3.13`, `cpp`, `css`, `docker`, `git`). The slug is
what you pass to `--slug`. Match the user's library/language to a slug here.

If you suspect a docset exists but isn't installed, list everything available:

```bash
smahtiepants docs available --offline   # fast, from local cache metadata
smahtiepants docs available             # refresh the list from DevDocs (network)
```

### 2. Install or update docs if needed

```bash
smahtiepants docs install react redux        # install one or more docsets
smahtiepants docs update                     # update all installed docsets
smahtiepants docs update python              # update one docset
```

Slugs accept common aliases (`js`, `ts`, `py`, `python`, `nodejs`, `c++`), which
resolve to canonical docsets. Installing also builds the search index when
embeddings are configured.

### 3. Search the docs

```bash
smahtiepants search "useEffect cleanup" --slug react --limit 5
smahtiepants search "request body parsing" --slug express
smahtiepants search "structuredClone" --language javascript
smahtiepants search "cfg target_arch target_os target triple" --slug rust
```

Useful flags:

- `--slug <slug>` — restrict to a docset (repeatable). Strongly prefer scoping
  by slug/language; it makes results far more relevant.
- `--language <name>` — filter by language-like docset (repeatable).
- `--limit <n>` — number of results (default 10, max 50).
- `--format text|json|xml` (or `--json`) — output format.

The default `text` output gives a ranked list with a query-aware **excerpt** and
a **read hint** for each match. Read the excerpts to find the right page.
Queries may be natural language or grep-like API terms; for code and language
reference docs, include exact identifiers, attributes, flags, function names, and
the surrounding concept words you expect to see.

### 4. Get more context when an excerpt isn't enough

The CLI excerpt is a snippet. To see the **full matched chunk** (not just the
trimmed excerpt), request JSON — the `text` field contains the complete chunk:

```bash
smahtiepants search "useEffect cleanup" --slug react --limit 3 --format json
```

For reading an entire documentation page, the read hint references the page's
resource (`smahtiepants://docsets/<slug>/pages/<pageId>`). Full-page reads are
served over the local server / MCP endpoint:

```bash
smahtiepants serve --host 127.0.0.1 --port 43877
# then GET /api/docsets/<slug>/pages/<pageId>/content
```

Prefer iterating on `search` with scoped queries and `--format json` first. If a
result looks close but the excerpt is ambiguous, read the full page from the
hint before concluding the docs do not contain the answer. Only stand up `serve`
when you genuinely need whole pages.

### 5. Ground your answer

Base your answer on what the docs actually say. If local docs contradict your
prior assumption, trust the docs and say what changed. Reference the docset and
page so the user can verify (e.g. "per the local `react` docs page
`hooks/useeffect`…"). If no relevant docset is installed, say so rather than
guessing, and offer to install it.

## Quick reference

| Goal | Command |
| --- | --- |
| List installed docsets + slugs | `smahtiepants docs installed` |
| List all available docsets | `smahtiepants docs available --offline` |
| Install docset(s) | `smahtiepants docs install <slug> [<slug>…]` |
| Update docset(s) | `smahtiepants docs update [<slug>]` |
| Search (scoped) | `smahtiepants search "<query>" --slug <slug> --limit 5` |
| Full chunk text | `smahtiepants search "<query>" --slug <slug> --format json` |
| Embedding/index status | `smahtiepants embeddings status` |

See `references/cli.md` for the complete command surface and output details.
