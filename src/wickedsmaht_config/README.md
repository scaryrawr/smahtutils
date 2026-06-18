# wickedsmaht_config

`wickedsmaht_config` is the shared configuration package used by `smahtiepants`, `smahties`, and `wickedpaste`.

Use it when a tool needs the same local OpenAI-compatible endpoint, chat model, documentation embedding model, or code embedding model without repeating flags on every command.

## Default config file

The default path is:

```text
$HOME/.wickedsmaht/config.json
```

Missing config files are allowed. Apps that require a setting will ask for the corresponding CLI flag or config key.

## Config shape

```json
{
  "base_url": "http://127.0.0.1:14892/v1",
  "model": "my-chat-model",
  "text_embedding_model": "my-text-embedding-model",
  "coding_embedding_model": "my-code-embedding-model",
  "smahtiepants": {
    "api_key_env": "OPENAI_API_KEY",
    "embeddings": {
      "enabled": true,
      "batch_size": 64,
      "max_chunk_chars": 2400,
      "min_chunk_chars": 200,
      "overlap_chars": 200,
      "max_chunks_per_page": 256,
      "max_request_bytes": 262144,
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

Kebab-case aliases such as `base-url`, `text-embedding-model`, `coding-embedding-model`, and `max-chunk-chars` are also accepted.

## Which app uses which keys

| Key | Used by |
| --- | --- |
| `base_url` | All apps that call an OpenAI-compatible endpoint. |
| `model` | `wickedpaste` chat conversion. |
| `text_embedding_model` | `smahtiepants` documentation embeddings. |
| `coding_embedding_model` | `smahties` code embeddings. |
| `smahtiepants` | `smahtiepants` embedding, API-key, server, auth, and CORS settings. |

## Precedence

CLI flags win over config values. Config values win over app defaults. If a required value is missing from both CLI flags and config, the app exits with a message naming the missing flag and config key.

## Python API

```python
from wickedsmaht_config import Config, resolve_setting

config = Config.load()
base_url = resolve_setting(
    cli_value=None,
    config_value=config.base_url,
    flag_name="--base-url",
    config_key="base_url",
)
```

Public API objects include `Config`, `ConfigError`, `SettingError`, `config_path_from_home`, and `resolve_setting`, plus dataclasses for `smahtiepants`-specific nested settings.

## Legacy compatibility

The package still accepts a legacy top-level `ddserve` object for `smahtiepants` settings when no `smahtiepants` object is present. New configs should use `smahtiepants`.
