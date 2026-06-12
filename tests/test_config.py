from __future__ import annotations

from pathlib import Path

import pytest

from wickedsmaht_config import Config, SettingError, config_path_from_home, resolve_setting


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
          "coding-embedding-model": "embed"
        }
        """,
        encoding="utf-8",
    )

    assert Config.load_from_path(path) == Config(
        base_url="http://127.0.0.1:14892/v1",
        model="chat",
        coding_embedding_model="embed",
    )


def test_resolve_setting_prefers_cli_and_errors() -> None:
    assert resolve_setting("cli", "config", "--model", "model") == "cli"
    assert resolve_setting(None, "config", "--model", "model") == "config"
    with pytest.raises(SettingError):
        resolve_setting(None, None, "--model", "model")
