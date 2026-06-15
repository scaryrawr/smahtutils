---
name: smahtiepants-mcp
description: Use this skill when smahtiepants MCP tools are available and a task depends on current documentation for a library, framework, language, runtime, API, or CLI tool (for example React, Express, Node, Python, Rust, CSS, Docker, Git, or Bun). Prefer the MCP tools `list_docsets`, `search_docs`, and `get_page_content` before answering from memory whenever API details, option names, defaults, examples, or "is this still current" questions could be stale. This is the MCP-tool version of the smahtiepants documentation workflow; use it instead of shelling out to the CLI when the MCP server is connected.
---

# smahtiepants MCP: ground answers in current local documentation

Your training data has a cutoff; library, framework, language, and CLI APIs
change. When the `smahtiepants` MCP server is connected, **use its MCP tools
before answering from memory** for API usage, syntax, flags, defaults, examples,
or current-best-practice questions.

The server searches a local DevDocs mirror with semantic + keyword search. It is
read-only through MCP: use it to discover installed docsets, search excerpts, and
read full Markdown pages. Do not start a shell command just to query
documentation when the MCP tools are already available.

## Workflow

### 1. Discover installed docsets

Call `list_docsets` first when you do not already know the exact installed slug.
It returns lines like:

```text
react: React
python~3.13: Python 3.13
css: CSS
```

Match the user's library or language to the canonical slug. Common language-like
inputs such as `python`, `js`, `ts`, `node`, and `c++` may map to canonical
DevDocs slugs such as `python~3.13`, `javascript`, `typescript`, `node`, or
`cpp`.

### 2. Search with scope

Call `search_docs` with a concise query and the most relevant `slugs` or
`languages` filter.

Example arguments:

```json
{"query": "useEffect cleanup function dependencies", "slugs": ["react"], "limit": 5}
```

Prefer scoped searches. Broad unscoped searches are acceptable only for initial
discovery when the relevant docset is unclear.

### 3. Read the result and fetch full pages when needed

`search_docs` returns ranked text blocks with:

- `<docsetSlug>:<pageId> <pageTitle>`
- match kind and score
- a `Read full page:` hint
- a query-aware excerpt

If the excerpt is not enough to answer accurately, call `get_page_content` with
the slug and page ID from the result.

Example arguments:

```json
{"slug": "react", "pageId": "hooks/useeffect"}
```

Use `startLine` and `endLine` only when you need a specific line range from a
large page. If your harness supports MCP resources, the same pages are available
as `smahtiepants://docsets/<slug>/pages/<pageId>`.

### 4. Answer from the docs

Base your final answer on what the local docs say. If they contradict your prior
assumption, trust the docs and explain the current behavior. Mention the docset
and page you used so the user can verify, for example:

> Per the local `react` docs page `hooks/useeffect`, the cleanup function runs
> before re-running the effect and when the component unmounts.

If no relevant docset is installed, say that local docs are unavailable for that
topic rather than guessing. If a relevant docset is installed but a specific page
link is missing, `get_page_content` fails for the hinted page, or scoped search
stays too sparse to verify current syntax, use another official documentation
path available in the environment rather than retrying local variants
indefinitely. Otherwise explain the limitation.

## Quick reference

| Goal | MCP tool |
| --- | --- |
| List installed docsets | `list_docsets` |
| Search docs | `search_docs` with `query`, optional `slugs`, `languages`, `limit` |
| Read a full page | `get_page_content` with `slug` and `pageId` |
| Read via resource URI | `smahtiepants://docsets/<slug>/pages/<pageId>` |

See `references/mcp.md` for the complete MCP tool surface and output details.
