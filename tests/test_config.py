from __future__ import annotations

from pathlib import Path

import pytest

from wickedsmaht_config import (
    Config,
    DdserveAuthSettings,
    DdserveEmbeddingSettings,
    DdserveServeSettings,
    DdserveSettings,
    SettingError,
    config_path_from_home,
    resolve_setting,
)


def test_config_path_uses_wickedsmaht_directory() -> None:
    assert config_path_from_home("/home/example") == Path("/home/example/.wickedsmaht/config.json")


def test_missing_config_loads_default(tmp_path: Path) -> None:
    assert Config.load_from_path(tmp_path / "missing.json") == Config()


def test_parses_keys_and_aliases(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        """
        {
          "base-url": "http://127.0.0.1:14892/v1",
          "model": "chat",
          "text-embedding-model": "text-embed",
          "coding-embedding-model": "embed",
          "ddserve": {
            "api-key-env": "DOCS_API_KEY",
            "embeddings": {
              "enabled": true,
              "batch-size": 8,
              "max-chunk-chars": 1200,
              "overlap-chars": 100
            },
            "serve": {
              "bind-address": "127.0.0.1",
              "port": 43877,
              "auth": {
                "token-env": "DDSERVE_TOKEN"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    assert Config.load_from_path(path) == Config(
        base_url="http://127.0.0.1:14892/v1",
        model="chat",
        text_embedding_model="text-embed",
        coding_embedding_model="embed",
        ddserve=DdserveSettings(
            api_key_env="DOCS_API_KEY",
            embeddings=DdserveEmbeddingSettings(
                enabled=True,
                batch_size=8,
                max_chunk_chars=1200,
                overlap_chars=100,
            ),
            serve=DdserveServeSettings(
                bind_address="127.0.0.1",
                port=43877,
                auth=DdserveAuthSettings(token_env="DDSERVE_TOKEN"),
            ),
        ),
    )


def test_resolve_setting_prefers_cli_and_errors() -> None:
    assert resolve_setting("cli", "config", "--model", "model") == "cli"
    assert resolve_setting(None, "config", "--model", "model") == "config"
    with pytest.raises(SettingError):
        resolve_setting(None, None, "--model", "model")
