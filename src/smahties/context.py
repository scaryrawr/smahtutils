from __future__ import annotations

from pathlib import Path, PurePosixPath

from .scanner import ensure_state_dir


class RuntimeContext:
    def __init__(
        self,
        repository_root: Path | None,
        storage_root: Path,
        runtime_root: Path,
        scope_prefix: str | None,
        auto_indexing_enabled: bool,
    ) -> None:
        self.repository_root = repository_root
        self.storage_root = storage_root
        self.runtime_root = runtime_root
        self.scope_prefix = scope_prefix
        self.auto_indexing_enabled = auto_indexing_enabled

    @classmethod
    def resolve(cls, runtime_root: Path) -> "RuntimeContext":
        runtime_root = runtime_root.resolve()
        repository_root = find_git_root(runtime_root)
        storage_root = repository_root or runtime_root
        scope_prefix = None
        if repository_root is not None:
            relative = runtime_root.relative_to(repository_root).as_posix()
            scope_prefix = relative or None
        return cls(
            repository_root=repository_root,
            storage_root=storage_root,
            runtime_root=runtime_root,
            scope_prefix=scope_prefix,
            auto_indexing_enabled=repository_root is not None,
        )

    def auto_index_root(self) -> Path | None:
        return self.runtime_root if self.auto_indexing_enabled else None

    def state_dir(self) -> Path:
        return ensure_state_dir(self.storage_root)

    def scoped_path_prefix(self, requested: str | None) -> str | None:
        normalized = normalize_relative_prefix(requested) if requested else None
        if self.scope_prefix and normalized and path_prefix_contains(self.scope_prefix, normalized):
            return normalized
        if self.scope_prefix and normalized:
            return join_path_prefix(self.scope_prefix, normalized)
        if self.scope_prefix:
            return self.scope_prefix
        return normalized


def find_git_root(start: Path) -> Path | None:
    current = start.parent if start.is_file() else start
    while True:
        marker = current / ".git"
        if marker.is_dir() or marker.is_file():
            return current
        if current.parent == current:
            return None
        current = current.parent


def normalize_relative_prefix(value: str) -> str | None:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise ValueError(f"path_prefix must be relative to the active smahties scope: {value}")
    parts: list[str] = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"path_prefix must not escape the active smahties scope: {value}")
        parts.append(part)
    return "/".join(parts) or None


def path_prefix_contains(scope: str, prefix: str) -> bool:
    return prefix == scope or prefix.startswith(f"{scope}/")


def join_path_prefix(scope: str, prefix: str) -> str:
    return prefix if not scope else f"{scope}/{prefix}"
