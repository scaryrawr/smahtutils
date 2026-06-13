from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from wickedsmaht_config import Config as SharedConfig
from wickedsmaht_config import ConfigError as SharedConfigError
from wickedsmaht_config import config_path_from_home

from .embeddings.chunks import (
    DEFAULT_CHUNK_MAX_CHARS,
    DEFAULT_CHUNK_MIN_CHARS,
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_MAX_CHUNKS_PER_PAGE,
)
from .errors import SmahtiepantsError

DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_MAX_REQUEST_BYTES = 256 * 1024
DEFAULT_EMBEDDING_MAX_CONCURRENT_REQUESTS = 1
DEFAULT_SERVE_BIND_ADDRESS = "127.0.0.1"
DEFAULT_SERVE_PORT = 43877
DEFAULT_SERVE_AUTH_TOKEN_ENV = "SMAHTIEPANTS_API_TOKEN"
LEGACY_SERVE_AUTH_TOKEN_ENV = "DDSERVE_API_TOKEN"
REDACTED_SECRET = "[redacted]"


@dataclass(frozen=True)
class OpenAiConfig:
    """OpenAI-compatible embedding endpoint settings for smahtiepants."""

    embedding_model: str
    api_key_env: str = DEFAULT_OPENAI_API_KEY_ENV
    api_key: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Chunking and embedding controls for smahtiepants documentation search."""

    enabled: bool
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    max_chunk_chars: int = DEFAULT_CHUNK_MAX_CHARS
    min_chunk_chars: int = DEFAULT_CHUNK_MIN_CHARS
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS
    max_chunks_per_page: int = DEFAULT_MAX_CHUNKS_PER_PAGE
    max_request_bytes: int = DEFAULT_EMBEDDING_MAX_REQUEST_BYTES
    max_concurrent_requests: int = DEFAULT_EMBEDDING_MAX_CONCURRENT_REQUESTS


@dataclass(frozen=True)
class ServeAuthConfig:
    """Bearer-token authentication settings for the smahtiepants HTTP server."""

    token_env: str = DEFAULT_SERVE_AUTH_TOKEN_ENV
    token: str | None = None


@dataclass(frozen=True)
class ServeCorsConfig:
    """CORS origin settings for the smahtiepants HTTP server."""

    origins: list[str]


@dataclass(frozen=True)
class ServeConfig:
    """HTTP server settings for smahtiepants."""

    bind_address: str | None = None
    port: int | None = None
    auth: ServeAuthConfig | None = None
    cors: ServeCorsConfig | None = None


@dataclass(frozen=True)
class SmahtiepantsConfig:
    """Runtime smahtiepants config derived from shared wickedsmaht config."""

    embeddings: EmbeddingsConfig
    openai: OpenAiConfig | None = None
    serve: ServeConfig | None = None


@dataclass(frozen=True)
class LoadedConfig:
    """Loaded smahtiepants runtime config plus shared config path metadata."""

    path: Path
    found: bool
    config: SmahtiepantsConfig


def resolve_config_path(config_path: str | None = None, env: dict[str, str] | None = None) -> Path:
    """Resolve the shared wickedsmaht config path used by smahtiepants."""

    if config_path is not None:
        if not config_path.strip():
            raise SmahtiepantsError("Invalid config path: path must not be empty")
        return Path(config_path).expanduser().resolve()
    env = env if env is not None else os.environ
    home = env.get("HOME")
    if not home:
        raise SmahtiepantsError(
            "could not locate wickedsmaht config: HOME environment variable is not set"
        )
    return config_path_from_home(home)


def load_config(config_path: str | None = None, env: dict[str, str] | None = None) -> LoadedConfig:
    """Load smahtiepants config exclusively from shared wickedsmaht config."""

    path = resolve_config_path(config_path, env)
    migrate_legacy_config_file(path)
    try:
        shared = SharedConfig.load_from_path(path)
    except SharedConfigError as exc:
        raise SmahtiepantsError(str(exc)) from exc
    return LoadedConfig(path=path, found=path.is_file(), config=from_shared_config(shared))


def migrate_legacy_config_file(path: str | Path) -> None:
    """Rewrite a legacy ddserve config object to smahtiepants when needed."""

    config_path = Path(path)
    try:
        raw = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SmahtiepantsError(f"failed to read config {config_path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(data, dict) or "smahtiepants" in data or "ddserve" not in data:
        return
    data["smahtiepants"] = data.pop("ddserve")
    temp = config_path.with_name(f"{config_path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(f"{json.dumps(data, indent=2)}\n", encoding="utf-8")
        temp.replace(config_path)
    except OSError as exc:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise SmahtiepantsError(
            f"failed to migrate legacy ddserve config at {config_path}: {exc}"
        ) from exc


def from_shared_config(shared: SharedConfig) -> SmahtiepantsConfig:
    """Convert shared wickedsmaht config into smahtiepants runtime config."""

    smahtiepants = shared_smahtiepants_settings(shared)
    embeddings = build_embeddings_config(
        shared.base_url is not None and shared.text_embedding_model is not None,
        smahtiepants.embeddings,
    )
    openai = (
        build_openai_config(shared)
        if embeddings.enabled and shared.base_url and shared.text_embedding_model
        else None
    )
    serve = build_serve_config(smahtiepants.serve)
    return SmahtiepantsConfig(embeddings=embeddings, openai=openai, serve=serve)


def build_openai_config(shared: SharedConfig) -> OpenAiConfig:
    """Build smahtiepants OpenAI settings from shared config."""

    smahtiepants = shared_smahtiepants_settings(shared)
    if not shared.base_url or not shared.text_embedding_model:
        raise SmahtiepantsError(
            "smahtiepants requires base_url and text_embedding_model in wickedsmaht config"
        )
    validate_base_url(shared.base_url, "base_url")
    validate_env_name(
        smahtiepants.api_key_env or DEFAULT_OPENAI_API_KEY_ENV, "smahtiepants.api_key_env"
    )
    return OpenAiConfig(
        embedding_model=shared.text_embedding_model,
        api_key_env=smahtiepants.api_key_env or DEFAULT_OPENAI_API_KEY_ENV,
        api_key=smahtiepants.api_key,
        base_url=shared.base_url,
    )


def shared_smahtiepants_settings(shared: SharedConfig) -> object:
    """Return primary smahtiepants settings, falling back to legacy ddserve settings."""

    settings = shared.smahtiepants
    legacy = getattr(shared, "ddserve", None)
    if legacy is not None and settings == type(settings)():
        return legacy
    return settings


def build_embeddings_config(enabled_default: bool, settings: object) -> EmbeddingsConfig:
    """Build smahtiepants embedding controls from shared config settings."""

    enabled = getattr(settings, "enabled", None)
    batch_size = getattr(settings, "batch_size", None)
    max_chunk_chars = getattr(settings, "max_chunk_chars", None)
    min_chunk_chars = getattr(settings, "min_chunk_chars", None)
    overlap_chars = getattr(settings, "overlap_chars", None)
    max_chunks_per_page = getattr(settings, "max_chunks_per_page", None)
    max_request_bytes = getattr(settings, "max_request_bytes", None)
    max_concurrent_requests = getattr(settings, "max_concurrent_requests", None)
    batch_size = batch_size if batch_size is not None else DEFAULT_EMBEDDING_BATCH_SIZE
    max_chunk_chars = max_chunk_chars if max_chunk_chars is not None else DEFAULT_CHUNK_MAX_CHARS
    min_chunk_chars = min_chunk_chars if min_chunk_chars is not None else DEFAULT_CHUNK_MIN_CHARS
    overlap_chars = overlap_chars if overlap_chars is not None else DEFAULT_CHUNK_OVERLAP_CHARS
    max_chunks_per_page = (
        max_chunks_per_page if max_chunks_per_page is not None else DEFAULT_MAX_CHUNKS_PER_PAGE
    )
    max_request_bytes = (
        max_request_bytes if max_request_bytes is not None else DEFAULT_EMBEDDING_MAX_REQUEST_BYTES
    )
    max_concurrent_requests = (
        max_concurrent_requests
        if max_concurrent_requests is not None
        else DEFAULT_EMBEDDING_MAX_CONCURRENT_REQUESTS
    )
    validate_positive_int(batch_size, "smahtiepants.embeddings.batch_size")
    validate_positive_int(max_chunk_chars, "smahtiepants.embeddings.max_chunk_chars")
    validate_positive_int(min_chunk_chars, "smahtiepants.embeddings.min_chunk_chars")
    validate_non_negative_int(overlap_chars, "smahtiepants.embeddings.overlap_chars")
    validate_positive_int(max_chunks_per_page, "smahtiepants.embeddings.max_chunks_per_page")
    validate_positive_int(max_request_bytes, "smahtiepants.embeddings.max_request_bytes")
    validate_positive_int(
        max_concurrent_requests, "smahtiepants.embeddings.max_concurrent_requests"
    )
    if min_chunk_chars > max_chunk_chars:
        raise SmahtiepantsError(
            "Invalid smahtiepants.embeddings.min_chunk_chars: must not exceed max_chunk_chars"
        )
    if overlap_chars >= max_chunk_chars:
        raise SmahtiepantsError(
            "Invalid smahtiepants.embeddings.overlap_chars: must be smaller than max_chunk_chars"
        )
    return EmbeddingsConfig(
        enabled=enabled if enabled is not None else enabled_default,
        batch_size=batch_size,
        max_chunk_chars=max_chunk_chars,
        min_chunk_chars=min_chunk_chars,
        overlap_chars=overlap_chars,
        max_chunks_per_page=max_chunks_per_page,
        max_request_bytes=max_request_bytes,
        max_concurrent_requests=max_concurrent_requests,
    )


def build_serve_config(settings: object | None) -> ServeConfig | None:
    """Build smahtiepants server controls from shared config settings."""

    if settings is None:
        return None
    bind_address = getattr(settings, "bind_address", None)
    port = getattr(settings, "port", None)
    if bind_address is not None and not bind_address.strip():
        raise SmahtiepantsError(
            "Invalid smahtiepants.serve.bind_address: expected a non-empty string"
        )
    if port is not None and not (1 <= port <= 65535):
        raise SmahtiepantsError(
            "Invalid smahtiepants.serve.port: expected an integer between 1 and 65535"
        )
    return ServeConfig(
        bind_address=bind_address,
        port=port,
        auth=build_serve_auth_config(getattr(settings, "auth", None)),
        cors=build_serve_cors_config(getattr(settings, "cors", None)),
    )


def build_serve_auth_config(settings: object | None) -> ServeAuthConfig | None:
    """Build smahtiepants auth controls from shared config settings."""

    if settings is None:
        return None
    token_env = getattr(settings, "token_env", None) or DEFAULT_SERVE_AUTH_TOKEN_ENV
    token = getattr(settings, "token", None)
    validate_env_name(token_env, "smahtiepants.serve.auth.token_env")
    if token is not None and not token.strip():
        raise SmahtiepantsError(
            "Invalid smahtiepants.serve.auth.token: expected a non-empty string"
        )
    return ServeAuthConfig(token_env=token_env, token=token)


def build_serve_cors_config(settings: object | None) -> ServeCorsConfig | None:
    """Build smahtiepants CORS controls from shared config settings."""

    if settings is None:
        return None
    origins = getattr(settings, "origins", [])
    if not origins:
        raise SmahtiepantsError(
            "Invalid smahtiepants.serve.cors.origins: expected at least one origin"
        )
    for origin in origins:
        if origin != "*":
            validate_base_url(origin, "smahtiepants.serve.cors.origins")
    return ServeCorsConfig(origins=origins)


def resolve_openai_api_key(
    config: SmahtiepantsConfig, env: dict[str, str] | None = None
) -> tuple[str, str] | None:
    """Resolve the API key source for smahtiepants embeddings."""

    if config.openai is None:
        return None
    env = env if env is not None else os.environ
    value = env.get(config.openai.api_key_env)
    if value and value.strip():
        return ("env", value)
    if config.openai.api_key:
        return ("config", config.openai.api_key)
    return None


def redact_config(config: SmahtiepantsConfig) -> dict[str, object]:
    """Return smahtiepants runtime config with inline secrets redacted."""

    redacted: dict[str, object] = {
        "embeddings": {
            "enabled": config.embeddings.enabled,
            "batchSize": config.embeddings.batch_size,
            "maxChunkChars": config.embeddings.max_chunk_chars,
            "minChunkChars": config.embeddings.min_chunk_chars,
            "overlapChars": config.embeddings.overlap_chars,
            "maxChunksPerPage": config.embeddings.max_chunks_per_page,
            "maxRequestBytes": config.embeddings.max_request_bytes,
            "maxConcurrentRequests": config.embeddings.max_concurrent_requests,
        }
    }
    if config.openai:
        openai: dict[str, object] = {
            "apiKeyEnv": config.openai.api_key_env,
            "embeddingModel": config.openai.embedding_model,
        }
        if config.openai.base_url:
            openai["baseURL"] = config.openai.base_url
        if config.openai.api_key:
            openai["apiKey"] = REDACTED_SECRET
        redacted["openai"] = openai
    if config.serve:
        serve: dict[str, object] = {}
        if config.serve.bind_address:
            serve["bindAddress"] = config.serve.bind_address
        if config.serve.port:
            serve["port"] = config.serve.port
        if config.serve.auth:
            auth: dict[str, object] = {"tokenEnv": config.serve.auth.token_env}
            if config.serve.auth.token:
                auth["token"] = REDACTED_SECRET
            serve["auth"] = auth
        if config.serve.cors:
            serve["cors"] = {"origins": config.serve.cors.origins}
        redacted["serve"] = serve
    return redacted


def validate_positive_int(value: int, path: str) -> None:
    """Validate a positive integer config value."""

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SmahtiepantsError(f"Invalid {path}: expected a positive integer")


def validate_non_negative_int(value: int, path: str) -> None:
    """Validate a non-negative integer config value."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SmahtiepantsError(f"Invalid {path}: expected a non-negative integer")


def validate_env_name(value: str, path: str) -> None:
    """Validate an environment variable name config value."""

    if not value.replace("_", "A").isalnum() or not (value[0].isalpha() or value[0] == "_"):
        raise SmahtiepantsError(f"Invalid {path}: expected an environment variable name")


def validate_base_url(value: str, path: str) -> None:
    """Validate an HTTP base URL config value."""

    if not value.strip():
        raise SmahtiepantsError(f"Invalid {path}: expected a non-empty URL")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SmahtiepantsError(f"Invalid {path}: expected an http or https URL")


DdserveConfig = SmahtiepantsConfig
DdserveError = SmahtiepantsError
