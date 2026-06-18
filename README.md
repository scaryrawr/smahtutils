# smahtutils

Small Python utilities for LLM-assisted local workflows.

This repository currently ships three command-line apps and one shared configuration package:

| Package | What it is for |
| --- | --- |
| [`smahtiepants`](src/smahtiepants/README.md) | Cache DevDocs locally, search them with keyword or embedding-backed results, and expose docs over REST/MCP. |
| [`smahties`](src/smahties/README.md) | Run semantic code search and duplicate-code detection over a local repository. |
| [`wickedpaste`](src/wickedpaste/README.md) | Convert the current clipboard image or text into HTML and GitHub Flavored Markdown with an OpenAI-compatible chat model. |
| [`wickedsmaht_config`](src/wickedsmaht_config/README.md) | Share API endpoint and model defaults across the apps. |

These tools are built for local developer workflows. They are intentionally lightweight, file-system backed, and configurable for OpenAI-compatible endpoints.

## Requirements

- Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/) for installing and running the project from this checkout.
- An OpenAI-compatible API endpoint for chat completions, embeddings, or both, depending on the app.
- Network access to DevDocs when installing or updating `smahtiepants` docsets.
- Clipboard access when using `wickedpaste`.

## Install and run from a checkout

```bash
uv sync --locked
uv run smahtiepants --help
uv run smahties --help
uv run wickedpaste --help
```

## Shared configuration

All apps can read defaults from `$HOME/.wickedsmaht/config.json`. CLI flags override these values.

```json
{
  "base_url": "http://127.0.0.1:14892/v1",
  "model": "my-chat-model",
  "text_embedding_model": "my-text-embedding-model",
  "coding_embedding_model": "my-code-embedding-model",
  "smahtiepants": {
    "embeddings": {
      "enabled": true
    },
    "serve": {
      "bind_address": "127.0.0.1",
      "port": 43877
    }
  }
}
```

See [`wickedsmaht_config`](src/wickedsmaht_config/README.md) for the full shared configuration shape.

## Quick examples

```bash
# Search local DevDocs after installing docsets.
uv run smahtiepants docs install python http
uv run smahtiepants search "request headers" --language python

# Search local code.
uv run smahties index .
uv run smahties query "where is configuration loaded?"

# Convert the current clipboard.
uv run wickedpaste
```

## Contributing

Contributor setup, validation commands, and architecture references live in [`CONTRIBUTING.md`](CONTRIBUTING.md). The package READMEs are written for people using the tools rather than changing their internals.
