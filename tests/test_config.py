from __future__ import annotations

from pathlib import Path

import pytest

from wickedsmaht_config import (
    Config,
    DdserveAuthSettings,
    DdserveEmbeddingSettings,
    DdserveServeSettings,
    DdserveSettings,
    SmahtiepantsAuthSettings,
    SmahtiepantsEmbeddingSettings,
    SmahtiepantsServeSettings,
    SmahtiepantsSettings,
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
          "smahtiepants": {
            "api-key-env": "DOCS_API_KEY",
            "embeddings": {
              "enabled": true,
              "batch-size": 8,
              "max-chunk-chars": 1200,
              "min-chunk-chars": 120,
              "overlap-chars": 100,
              "max-chunks-per-page": 64,
              "max-request-bytes": 4096,
              "max-concurrent-requests": 3
            },
            "serve": {
              "bind-address": "127.0.0.1",
              "port": 43877,
              "auth": {
                "token-env": "SMAHTIEPANTS_TOKEN"
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
        smahtiepants=SmahtiepantsSettings(
            api_key_env="DOCS_API_KEY",
            embeddings=SmahtiepantsEmbeddingSettings(
                enabled=True,
                batch_size=8,
                max_chunk_chars=1200,
                min_chunk_chars=120,
                overlap_chars=100,
                max_chunks_per_page=64,
                max_request_bytes=4096,
                max_concurrent_requests=3,
            ),
            serve=SmahtiepantsServeSettings(
                bind_address="127.0.0.1",
                port=43877,
                auth=SmahtiepantsAuthSettings(token_env="SMAHTIEPANTS_TOKEN"),
            ),
        ),
    )


def test_legacy_ddserve_config_key_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        """
        {
          "ddserve": {
            "api-key-env": "DOCS_API_KEY",
            "embeddings": {
              "batch-size": 8
            },
            "serve": {
              "auth": {
                "token-env": "DDSERVE_TOKEN"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    loaded = Config.load_from_path(path)

    assert loaded.smahtiepants == SmahtiepantsSettings(
        api_key_env="DOCS_API_KEY",
        embeddings=SmahtiepantsEmbeddingSettings(batch_size=8),
        serve=SmahtiepantsServeSettings(auth=SmahtiepantsAuthSettings(token_env="DDSERVE_TOKEN")),
    )
    assert loaded.ddserve == DdserveSettings(
        api_key_env="DOCS_API_KEY",
        embeddings=DdserveEmbeddingSettings(batch_size=8),
        serve=DdserveServeSettings(auth=DdserveAuthSettings(token_env="DDSERVE_TOKEN")),
    )


def test_resolve_setting_prefers_cli_and_errors() -> None:
    assert resolve_setting("cli", "config", "--model", "model") == "cli"
    assert resolve_setting(None, "config", "--model", "model") == "config"
    with pytest.raises(SettingError):
        resolve_setting(None, None, "--model", "model")
