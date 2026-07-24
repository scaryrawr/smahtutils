# smahtiepants

`smahtiepants` is a local DevDocs mirror and search server. Use it when you want agents, scripts, or your terminal to answer library and API questions from locally cached documentation instead of repeatedly browsing DevDocs.

## What it does

- Lists, installs, updates, and removes DevDocs docsets.
- Stores installed pages as Markdown in a local cache.
- Searches cached docs with keyword fallback and optional embedding-backed semantic ranking.
- Serves read-only REST endpoints, a Streamable HTTP MCP endpoint, and a Copilot session-start hook.

## Quick start

```bash
uv run smahtiepants cache path
uv run smahtiepants docs available
uv run smahtiepants docs install python http
uv run smahtiepants search "request headers" --slug http
uv run smahtiepants docs page http <pageId-from-search>
```

Start the local server:

```bash
uv run smahtiepants serve --host 127.0.0.1 --port 43877
```

The MCP endpoint is plain HTTP at:

```text
http://127.0.0.1:43877/mcp
```

## Configuration

`smahtiepants` reads shared defaults from `$HOME/.wickedsmaht/config.json`, or from a file passed with `--config`.

```json
{
  "base_url": "http://127.0.0.1:14892/v1",
  "text_embedding_model": "my-text-embedding-model",
  "smahtiepants": {
    "api_key_env": "OPENAI_API_KEY",
    "embeddings": {
      "enabled": true,
      "batch_size": 64,
      "max_chunk_chars": 2400,
      "overlap_chars": 200,
      "max_concurrent_requests": 1
    },
    "serve": {
      "bind_address": "127.0.0.1",
      "port": 43877,
      "auth": {
        "token_env": "SMAHTIEPANTS_API_TOKEN"
      },
      "cors": {
        "origins": ["http://127.0.0.1:3000"]
      }
    }
  }
}
```

Documentation embeddings use `text_embedding_model`, not `coding_embedding_model`. If `base_url` and `text_embedding_model` are absent, searches still work with keyword matching.

## Common commands

| Command | Purpose |
| --- | --- |
| `uv run smahtiepants sources list` | Show supported documentation sources. |
| `uv run smahtiepants docs available` | List DevDocs docsets available to install. |
| `uv run smahtiepants docs installed` | List docsets already cached locally. |
| `uv run smahtiepants docs install <slug...>` | Download one or more docsets. |
| `uv run smahtiepants docs update [slug]` | Update one docset, or every installed docset when no slug is given. |
| `uv run smahtiepants docs remove <slug>` | Remove an installed docset. |
| `uv run smahtiepants docs page <slug> <pageId>` | Read a full Markdown page returned by search. |
| `uv run smahtiepants embeddings status [slug]` | Show embedding/index status. |
| `uv run smahtiepants embeddings refresh <slug>` | Embed missing or stale documentation chunks. |
| `uv run smahtiepants embeddings rebuild <slug>` | Recreate embeddings for a docset. |
| `uv run smahtiepants search <query>` | Search installed docs. |
| `uv run smahtiepants serve` | Serve REST, MCP, and Copilot hook endpoints. |
| `uv run smahtiepants config show` | Show resolved runtime configuration with secrets redacted. |

Use `--json` on list/status/install/search/page commands when a script or agent should consume structured output. `search` also supports `--format text`, `--format json`, and `--format xml`.

## Cache location

The cache root is resolved in this order:

1. `SMAHTIEPANTS_CACHE_DIR`
2. `DDSERVE_CACHE_DIR` for legacy users
3. `$XDG_CACHE_HOME/smahtiepants`
4. `~/.cache/smahtiepants`

Run `uv run smahtiepants cache path` to see the active cache directory.

## Server endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Health check. |
| `GET /api` | Basic route discovery. |
| `GET /api/docsets` | List installed docsets. |
| `GET /api/docsets/{slug}` | Read docset metadata. |
| `GET /api/docsets/{slug}/pages` | List pages, optionally filtered by query/type. |
| `GET /api/docsets/{slug}/pages/{page_id}` | Read page metadata. |
| `GET /api/docsets/{slug}/pages/{page_id}/content` | Read Markdown page content. |
| `GET` or `POST /api/search` | Search installed docs. |
| `GET /api/embeddings/status[/slug]` | Read embedding status. |
| `POST /copilot/hooks/sessionStart` | Return context for Copilot session-start hooks. |
| `/mcp` | Streamable HTTP MCP tools and resources. |

Protected endpoints require `Authorization: Bearer <token>` when `smahtiepants.serve.auth.token` or `token_env` is configured.

## How it works

DevDocs docsets are resolved through upstream aliases first, curated aliases second, and docset metadata last. Installed cache directories, manifest keys, and embedding docset IDs always use canonical DevDocs slugs so aliases such as `py`, `python`, `js`, `ts`, `c++`, and `nodejs` do not create duplicate caches.

SQLite is the authoritative store for documentation chunks and vectors. Annoy indexes are rebuildable sidecar files used to find bounded semantic candidates quickly; exact scores are computed from vectors loaded back from SQLite. Search results include an excerpt plus a read hint so agents can fetch the full matched page when needed.

For direct CLI use, pass the result's slug and page ID to `uv run smahtiepants docs page <slug> <pageId>`. This reads the cached Markdown file directly; starting `serve` is only necessary for REST, MCP, or hook clients.

## Safety notes

Do not commit the cache directory, embeddings, API responses, bearer tokens, or copied documentation content. Keep the server bound to `127.0.0.1` unless you have configured auth and understand the exposure.
