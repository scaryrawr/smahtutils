from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .errors import SmahtiepantsError
from .models import (
    CACHE_SCHEMA_VERSION,
    CacheManifest,
    from_cache_manifest,
    from_docset_manifest,
    to_jsonable,
)

DEFAULT_CACHE_ENV = "SMAHTIEPANTS_CACHE_DIR"
LEGACY_CACHE_ENV = "DDSERVE_CACHE_DIR"
SAFE_PATH_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9._+~-]*$", re.IGNORECASE)


@dataclass(frozen=True)
class CachePaths:
    """Represent CachePaths."""

    root: Path
    manifest: Path
    sources_root: Path
    devdocs_source_root: Path
    devdocs_source_index: Path
    docs_root: Path
    embeddings_root: Path
    embeddings_db: Path
    locks_root: Path


def resolve_cache_root(env: dict[str, str] | None = None) -> Path:
    """Resolve cache root."""
    env = env if env is not None else os.environ
    override = env.get(DEFAULT_CACHE_ENV)
    if override and override.strip():
        return Path(override).expanduser().resolve()
    legacy_override = env.get(LEGACY_CACHE_ENV)
    if legacy_override and legacy_override.strip():
        return Path(legacy_override).expanduser().resolve()
    xdg_cache_home = env.get("XDG_CACHE_HOME")
    if xdg_cache_home and xdg_cache_home.strip():
        cache_home = Path(xdg_cache_home).expanduser()
    else:
        home = env.get("HOME")
        cache_home = Path(home).expanduser() / ".cache" if home else Path.home() / ".cache"
    return migrate_legacy_cache_root(
        (cache_home / "smahtiepants").resolve(),
        (cache_home / "ddserve").resolve(),
    )


def migrate_legacy_cache_root(root: Path, legacy_root: Path) -> Path:
    """Move a legacy default ddserve cache to the smahtiepants cache path."""

    if root.exists() or not legacy_root.exists():
        return root
    if not legacy_root.is_dir():
        raise SmahtiepantsError(f"Legacy cache path is not a directory: {legacy_root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        legacy_root.replace(root)
    except FileExistsError:
        return root
    except OSError as exc:
        raise SmahtiepantsError(
            f"Failed to migrate legacy cache directory from {legacy_root} to {root}: {exc}"
        ) from exc
    return root


def cache_paths(root: str | Path | None = None) -> CachePaths:
    """Implement cache paths."""
    resolved = Path(root).expanduser().resolve() if root is not None else resolve_cache_root()
    return CachePaths(
        root=resolved,
        manifest=resolved / "manifest.json",
        sources_root=resolved / "sources",
        devdocs_source_root=resolved / "sources" / "devdocs",
        devdocs_source_index=resolved / "sources" / "devdocs" / "index.json",
        docs_root=resolved / "docs",
        embeddings_root=resolved / "embeddings",
        embeddings_db=resolved / "embeddings" / "embeddings.sqlite",
        locks_root=resolved / "locks",
    )


def ensure_cache_root(root: str | Path | None = None) -> CachePaths:
    """Implement ensure cache root."""
    paths = cache_paths(root)
    paths.devdocs_source_root.mkdir(parents=True, exist_ok=True)
    paths.docs_root.mkdir(parents=True, exist_ok=True)
    paths.locks_root.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_embedding_db_path(root: str | Path | None = None) -> Path:
    """Implement ensure embedding db path."""
    paths = cache_paths(root)
    paths.embeddings_root.mkdir(parents=True, exist_ok=True)
    return paths.embeddings_db


def create_empty_cache_manifest(now: datetime | None = None) -> CacheManifest:
    """Implement create empty cache manifest."""
    return CacheManifest(
        schema_version=CACHE_SCHEMA_VERSION,
        updated_at=(now or datetime.now(UTC)).isoformat().replace("+00:00", "Z"),
        docs={},
    )


def read_cache_manifest(root: str | Path | None = None) -> CacheManifest:
    """Read cache manifest."""
    paths = cache_paths(root)
    value = read_json_file(paths.manifest)
    if value is None:
        return create_empty_cache_manifest()
    manifest = from_cache_manifest(value)
    if manifest.schema_version != CACHE_SCHEMA_VERSION:
        raise SmahtiepantsError(f"Unsupported cache manifest at {paths.manifest}")
    return manifest


def write_cache_manifest(root: str | Path, manifest: CacheManifest) -> None:
    """Write cache manifest."""
    atomic_write_json(cache_paths(root).manifest, to_jsonable(manifest))


def read_docset_manifest(root: str | Path, slug: str):
    """Read docset manifest."""
    assert_safe_path_segment(slug, "docset slug")
    value = read_json_file(cache_paths(root).docs_root / slug / "manifest.json")
    return from_docset_manifest(value) if value is not None else None


def read_json_file(path: str | Path) -> Any | None:
    """Read json file."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise SmahtiepantsError(f"Invalid JSON in {path}: {exc}") from exc


def atomic_write_json(path: str | Path, value: object) -> None:
    """Implement atomic write json."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(
        f"{target.name}.{os.getpid()}.{int(datetime.now().timestamp() * 1000)}.tmp"
    )
    temp.write_text(f"{json.dumps(value, indent=2)}\n", encoding="utf-8")
    temp.replace(target)


def path_exists(path: str | Path) -> bool:
    """Implement path exists."""
    return Path(path).exists()


def assert_safe_path_segment(value: str, label: str) -> None:
    """Assert safe path segment."""
    if not SAFE_PATH_SEGMENT_RE.match(value):
        raise SmahtiepantsError(f"Invalid {label}: {value}")


def replace_directory(stage_dir: str | Path, final_dir: str | Path) -> None:
    """Implement replace directory."""
    stage = Path(stage_dir)
    final = Path(final_dir)
    backup = final.with_name(
        f"{final.name}.previous-{os.getpid()}-{int(datetime.now().timestamp() * 1000)}"
    )
    final_exists = final.exists()
    try:
        if final_exists:
            final.replace(backup)
        stage.replace(final)
        if final_exists:
            shutil.rmtree(backup, ignore_errors=True)
    except OSError as exc:
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        if backup.exists():
            backup.replace(final)
        raise SmahtiepantsError(
            f"Failed to replace cached docset directory {final}: {exc}"
        ) from exc


class DocsetLock:
    """Represent DocsetLock."""

    def __init__(self, lock_dir: Path) -> None:
        """Implement init."""
        self.lock_dir = lock_dir

    def release(self) -> None:
        """Implement release."""
        shutil.rmtree(self.lock_dir, ignore_errors=True)

    def __enter__(self) -> "DocsetLock":
        """Implement enter."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Implement exit."""
        self.release()


def acquire_docset_lock(root: str | Path, slug: str) -> DocsetLock:
    """Implement acquire docset lock."""
    assert_safe_path_segment(slug, "docset slug")
    paths = ensure_cache_root(root)
    lock_dir = paths.locks_root / f"{slug}.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        if not is_stale_lock(lock_dir):
            raise SmahtiepantsError(
                f'Docset "{slug}" is already being installed or updated'
            ) from exc
        shutil.rmtree(lock_dir, ignore_errors=True)
        lock_dir.mkdir()
    atomic_write_json(
        lock_dir / "owner.json",
        {"pid": os.getpid(), "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z")},
    )
    return DocsetLock(lock_dir)


def is_stale_lock(lock_dir: Path) -> bool:
    """Return whether stale lock."""
    owner = read_json_file(lock_dir / "owner.json")
    if not isinstance(owner, dict):
        return False
    pid = owner.get("pid")
    if isinstance(pid, int) and pid > 0 and not is_live_process(pid):
        return True
    created_at = owner.get("createdAt")
    if not isinstance(created_at, str):
        return False
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return datetime.now(UTC) - created > timedelta(hours=2)


def is_live_process(pid: int) -> bool:
    """Return whether live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
