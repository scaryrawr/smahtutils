from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR_NAME = ".wickedsmaht"
CONFIG_FILE_NAME = "config.json"


class ConfigError(Exception):
    pass


class SettingError(Exception):
    def __init__(self, flag_name: str, config_key: str) -> None:
        self.flag_name = flag_name
        self.config_key = config_key
        super().__init__(
            f"missing required setting: pass `{flag_name}` or set `{config_key}` "
            "in $HOME/.wickedsmaht/config.json"
        )


@dataclass(frozen=True)
class Config:
    base_url: str | None = None
    model: str | None = None
    coding_embedding_model: str | None = None

    @classmethod
    def load(cls) -> "Config":
        home = os.environ.get("HOME")
        if not home:
            raise ConfigError(
                "could not locate wickedsmaht config: HOME environment variable is not set"
            )
        return cls.load_from_path(config_path_from_home(home))

    @classmethod
    def load_from_path(cls, path: str | os.PathLike[str]) -> "Config":
        config_path = Path(path)
        try:
            raw = config_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls()
        except OSError as exc:
            raise ConfigError(f"failed to read config {config_path}: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"failed to parse config {config_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"failed to parse config {config_path}: expected JSON object")

        return cls(
            base_url=_optional_string(data, "base_url", "base-url"),
            model=_optional_string(data, "model"),
            coding_embedding_model=_optional_string(
                data, "coding_embedding_model", "coding-embedding-model"
            ),
        )


def config_path_from_home(home: str | os.PathLike[str]) -> Path:
    return Path(home) / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def resolve_setting(
    cli_value: str | None,
    config_value: str | None,
    flag_name: str,
    config_key: str,
) -> str:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    raise SettingError(flag_name, config_key)


def _optional_string(data: dict[object, object], key: str, alias: str | None = None) -> str | None:
    value = data.get(key)
    if value is None and alias is not None:
        value = data.get(alias)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"config key `{key}` must be a string")
    return value
