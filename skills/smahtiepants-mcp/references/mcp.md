# smahtiepants MCP reference

`smahtiepants` exposes a read-only MCP server for searching locally installed
DevDocs documentation. Prefer these tools over the `smahtiepants` CLI whenever
the MCP server is already connected to the agent harness.

## Tools

### `list_docsets`

Lists locally installed DevDocs docsets.

Arguments: none.

Typical output:

```text
javascript: JavaScript
react: React
python~3.13: Python 3.13
```

Use the slug before the colon as the `search_docs.slugs` value.

### `search_docs`

Searches installed documentation with semantic + keyword ranking.

Arguments:

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `query` | string | yes | Concise natural-language or API query. |
| `slugs` | string or string array | no | Canonical docset slug(s), strongly preferred for relevance. |
| `languages` | string or string array | no | Language-like filter when an exact slug is not known. |
| `limit` | integer | no | Defaults to 10; use 3-5 for focused checks. |

Example:

```json
{
  "query": "route parameters query string request handler",
  "slugs": ["express"],
  "limit": 5
}
```

Output is plain text, grouped by match:

```text
express:guide/routing Routing
Match: semantic score 0.812
Read full page: CLI: uv run smahtiepants docs page express guide/routing; MCP: get_page_content slug="express" pageId="guide/routing"; resource: smahtiepants://docsets/express/pages/guide/routing
Excerpt:
...
```

Each result includes a query-aware excerpt. Search ranking uses semantic
similarity as the primary signal when embeddings are configured; keyword hits
may be marked as `hybrid` or fill fallback slots.

### `get_page_content`

Reads Markdown content for an installed documentation page.

Arguments:

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `slug` | string | yes | Docset slug from `list_docsets` or a search result. |
| `pageId` / `page_id` | string | yes | Page ID from `search_docs`, e.g. `hooks/useeffect`. |
| `startLine` | integer | no | Optional inclusive start line for large pages. |
| `endLine` | integer | no | Optional inclusive end line for large pages. |

Example:

```json
{
  "slug": "css",
  "pageId": "css_containment/container_queries",
  "startLine": 1,
  "endLine": 120
}
```

Use this when an excerpt does not include enough surrounding detail, syntax, or
edge-case information to answer confidently.

## Resources

Documentation pages can also be read through MCP resource URIs when the harness
supports resource reads:

```text
smahtiepants://docsets/<slug>/pages/<pageId>
```

Prefer `get_page_content` when tool calls are easier in the current harness.

## Answering guidance

- Always scope searches by `slugs` or `languages` once you identify the target
  docset.
- Use multiple focused searches rather than one broad query if the first result
  does not answer the question.
- Cite the local docset and page ID in your answer.
- Do not claim a docset was checked if the MCP call failed or returned no
  relevant result.
- If the needed docset is not installed, say that the local mirror cannot verify
  the topic. MCP does not install docsets; use another approved documentation
  path only if it is available.
