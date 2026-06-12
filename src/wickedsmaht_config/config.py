from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR_NAME = ".wickedsmaht"
CONFIG_FILE_NAME = "config.json"


class ConfigError(Exception):
    """Raised when shared configuration cannot be loaded or parsed."""

    pass


class SettingError(Exception):
    """Raised when a required setting is missing from CLI args and config."""

    def __init__(self, flag_name: str, config_key: str) -> None:
        self.flag_name = flag_name
        self.config_key = config_key
        super().__init__(
            f"missing required setting: pass `{flag_name}` or set `{config_key}` "
            "in $HOME/.wickedsmaht/config.json"
        )


@dataclass(frozen=True)
class DdserveEmbeddingSettings:
    """Optional ddserve embedding defaults from shared config."""

    enabled: bool | None = None
    batch_size: int | None = None
    max_chunk_chars: int | None = None
    min_chunk_chars: int | None = None
    overlap_chars: int | None = None
    max_chunks_per_page: int | None = None
    max_request_bytes: int | None = None
    max_concurrent_requests: int | None = None


@dataclass(frozen=True)
class DdserveAuthSettings:
    """Optional ddserve bearer auth defaults from shared config."""

    token_env: str | None = None
    token: str | None = None


@dataclass(frozen=True)
class DdserveCorsSettings:
    """Optional ddserve CORS defaults from shared config."""

    origins: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DdserveServeSettings:
    """Optional ddserve HTTP server defaults from shared config."""

    bind_address: str | None = None
    port: int | None = None
    auth: DdserveAuthSettings | None = None
    cors: DdserveCorsSettings | None = None


@dataclass(frozen=True)
class DdserveSettings:
    """Optional ddserve-specific settings from shared config."""

    api_key_env: str | None = None
    api_key: str | None = None
    embeddings: DdserveEmbeddingSettings = field(default_factory=DdserveEmbeddingSettings)
    serve: DdserveServeSettings | None = None


@dataclass(frozen=True)
class Config:
    """Shared optional defaults loaded from .wickedsmaht/config.json."""

    base_url: str | None = None
    model: str | None = None
    text_embedding_model: str | None = None
    coding_embedding_model: str | None = None
    ddserve: DdserveSettings = field(default_factory=DdserveSettings)

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from the default path under HOME."""

        home = os.environ.get("HOME")
        if not home:
            raise ConfigError(
                "could not locate wickedsmaht config: HOME environment variable is not set"
            )
        return cls.load_from_path(config_path_from_home(home))

    @classmethod
    def load_from_path(cls, path: str | os.PathLike[str]) -> "Config":
        """Load configuration from an explicit JSON file path."""

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
            text_embedding_model=_optional_string(
                data, "text_embedding_model", "text-embedding-model"
            ),
            coding_embedding_model=_optional_string(
                data, "coding_embedding_model", "coding-embedding-model"
            ),
            ddserve=_parse_ddserve_settings(_optional_object(data, "ddserve") or {}, "ddserve"),
        )


def config_path_from_home(home: str | os.PathLike[str]) -> Path:
    """Build the default config path for a home directory."""

    return Path(home) / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def resolve_setting(
    cli_value: str | None,
    config_value: str | None,
    flag_name: str,
    config_key: str,
) -> str:
    """Resolve a required setting, preferring CLI value over config value."""

    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    raise SettingError(flag_name, config_key)


def _parse_ddserve_settings(data: dict[object, object], path: str) -> DdserveSettings:
    """Parse the optional shared ddserve settings object."""

    return DdserveSettings(
        api_key_env=_optional_string(data, "api_key_env", "api-key-env", "apiKeyEnv"),
        api_key=_optional_string(data, "api_key", "api-key", "apiKey"),
        embeddings=_parse_ddserve_embedding_settings(
            _optional_object(data, "embeddings") or {}, f"{path}.embeddings"
        ),
        serve=_parse_ddserve_serve_settings(_optional_object(data, "serve"), f"{path}.serve"),
    )


def _parse_ddserve_embedding_settings(
    data: dict[object, object], path: str
) -> DdserveEmbeddingSettings:
    """Parse optional ddserve embedding controls."""

    return DdserveEmbeddingSettings(
        enabled=_optional_bool(data, "enabled"),
        batch_size=_optional_int(
            data, f"{path}.batch_size", "batch_size", "batch-size", "batchSize"
        ),
        max_chunk_chars=_optional_int(
            data, f"{path}.max_chunk_chars", "max_chunk_chars", "max-chunk-chars", "maxChunkChars"
        ),
        min_chunk_chars=_optional_int(
            data, f"{path}.min_chunk_chars", "min_chunk_chars", "min-chunk-chars", "minChunkChars"
        ),
        overlap_chars=_optional_int(
            data, f"{path}.overlap_chars", "overlap_chars", "overlap-chars", "overlapChars"
        ),
        max_chunks_per_page=_optional_int(
            data,
            f"{path}.max_chunks_per_page",
            "max_chunks_per_page",
            "max-chunks-per-page",
            "maxChunksPerPage",
        ),
        max_request_bytes=_optional_int(
            data,
            f"{path}.max_request_bytes",
            "max_request_bytes",
            "max-request-bytes",
            "maxRequestBytes",
        ),
        max_concurrent_requests=_optional_int(
            data,
            f"{path}.max_concurrent_requests",
            "max_concurrent_requests",
            "max-concurrent-requests",
            "maxConcurrentRequests",
        ),
    )


def _parse_ddserve_serve_settings(
    data: dict[object, object] | None, path: str
) -> DdserveServeSettings | None:
    """Parse optional ddserve HTTP server controls."""

    if data is None:
        return None
    return DdserveServeSettings(
        bind_address=_optional_string(data, "bind_address", "bind-address", "bindAddress"),
        port=_optional_int(data, f"{path}.port", "port"),
        auth=_parse_ddserve_auth_settings(_optional_object(data, "auth"), f"{path}.auth"),
        cors=_parse_ddserve_cors_settings(_optional_object(data, "cors"), f"{path}.cors"),
    )


def _parse_ddserve_auth_settings(
    data: dict[object, object] | None, path: str
) -> DdserveAuthSettings | None:
    """Parse optional ddserve auth controls."""

    if data is None:
        return None
    return DdserveAuthSettings(
        token_env=_optional_string(data, "token_env", "token-env", "tokenEnv"),
        token=_optional_string(data, "token"),
    )


def _parse_ddserve_cors_settings(
    data: dict[object, object] | None, path: str
) -> DdserveCorsSettings | None:
    """Parse optional ddserve CORS controls."""

    if data is None:
        return None
    origins = _optional_string_list(data, f"{path}.origins", "origins")
    return DdserveCorsSettings(origins=origins or [])


def _optional_object(data: dict[object, object], key: str) -> dict[object, object] | None:
    """Return an optional object value from shared config."""

    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"config key `{key}` must be an object")
    return value


def _optional_string(data: dict[object, object], key: str, *aliases: str) -> str | None:
    """Return an optional string value from shared config."""

    value = data.get(key)
    for alias in aliases:
        if value is None:
            value = data.get(alias)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"config key `{key}` must be a string")
    return value


def _optional_bool(data: dict[object, object], key: str) -> bool | None:
    """Return an optional boolean value from shared config."""

    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ConfigError(f"config key `{key}` must be a boolean")
    return value


def _optional_int(data: dict[object, object], path: str, key: str, *aliases: str) -> int | None:
    """Return an optional integer value from shared config."""

    value = data.get(key)
    for alias in aliases:
        if value is None:
            value = data.get(alias)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"config key `{path}` must be an integer")
    return value


def _optional_string_list(
    data: dict[object, object], path: str, key: str, *aliases: str
) -> list[str] | None:
    """Return an optional string list value from shared config."""

    value = data.get(key)
    for alias in aliases:
        if value is None:
            value = data.get(alias)
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list):
        raise ConfigError(f"config key `{path}` must be a string or array of strings")
    if not all(isinstance(item, str) for item in values):
        raise ConfigError(f"config key `{path}` must contain only strings")
    return values
