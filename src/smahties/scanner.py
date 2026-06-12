from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from .models import SourceFile

DEFAULT_MAX_FILE_BYTES = 512 * 1024
EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".next",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".smahties",
    ".tox",
    ".turbo",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "target",
    "venv",
}
EXCLUDED_DIR_NAME_SUFFIXES = (".egg-info", ".dist-info")
EXCLUDED_FILE_NAMES = {
    ".gitignore",
    ".ignore",
    ".gitattributes",
    ".gitmodules",
    "bun.lock",
    "bun.lockb",
    "Cargo.lock",
    "composer.lock",
    "Gemfile.lock",
    "npm-shrinkwrap.json",
    "package-lock.json",
    "Pipfile.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}


class Scanner:
    """Discovers and reads indexable UTF-8 source files under a root."""

    def __init__(self, root: Path, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> None:
        self.root = root.resolve()
        self.max_file_bytes = max_file_bytes
        self.git_root = discover_git_root(self.root)

    def discover_files(self, path: Path) -> list[Path]:
        """Return indexable files under a file or directory path."""

        path = path.resolve()
        if is_excluded_path(self.root, path) or self.is_ignored_by_git(path):
            return []
        if path.is_file():
            return [path] if self.is_indexable_path(path) else []
        if not path.exists():
            return []

        files: list[Path] = []
        for current, dir_names, file_names in os.walk(path, followlinks=False):
            current_path = Path(current)
            dir_candidates = [
                current_path / name
                for name in dir_names
                if not is_excluded_dir_name(name)
                and not is_excluded_path(self.root, current_path / name)
            ]
            ignored_dirs = self.git_ignored_paths(dir_candidates)
            dir_names[:] = [path.name for path in dir_candidates if path not in ignored_dirs]
            file_candidates = [current_path / file_name for file_name in file_names]
            ignored_files = self.git_ignored_paths(file_candidates)
            for candidate in file_candidates:
                if candidate not in ignored_files and self._is_indexable_path(
                    candidate, check_git=False
                ):
                    files.append(candidate)
        return files

    def read_source(self, path: Path) -> SourceFile | None:
        """Read an indexable source file, skipping binary or non-UTF-8 data."""

        if not self.is_indexable_path(path):
            return None
        data = path.read_bytes()
        if b"\0" in data:
            return None
        try:
            contents = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return SourceFile(
            absolute_path=path,
            relative_path=self.relative_path(path),
            contents=contents,
            hash=sha256_hex(contents.encode("utf-8")),
        )

    def is_discoverable_file(self, path: Path) -> bool:
        """Return whether a file would be found by discovery rules."""

        return path in self.discover_files(path)

    def relative_path(self, path: Path) -> str:
        """Return a POSIX-style path relative to the scanner root."""

        path = path.resolve()
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            relative = path
        return relative.as_posix()

    def resolve_existing_under_root(self, requested: str) -> Path:
        """Resolve an existing requested path under the scanner root."""

        return self.resolve_existing_under(self.root, requested)

    def resolve_existing_under(self, base: Path, requested: str) -> Path:
        """Resolve an existing requested path without escaping root or scope."""

        requested_path = Path(requested)
        absolute = requested_path if requested_path.is_absolute() else base / requested_path
        root = self.root.resolve()
        base = base.resolve()
        if not base.is_relative_to(root):
            raise ValueError(f"active smahties scope is outside the indexed root: {base}")
        canonical = absolute.resolve(strict=True)
        if not canonical.is_relative_to(root):
            raise ValueError(f"path is outside the indexed root: {absolute}")
        if not canonical.is_relative_to(base):
            raise ValueError(f"path is outside the active smahties scope: {absolute}")
        return canonical

    def is_indexable_path(self, path: Path) -> bool:
        """Return whether a path is an indexable file under scanner limits."""

        return self._is_indexable_path(path, check_git=True)

    def is_ignored_by_git(self, path: Path) -> bool:
        """Return whether Git ignore rules exclude the path."""

        return path in self.git_ignored_paths([path])

    def git_ignored_paths(self, paths: list[Path]) -> set[Path]:
        """Return the subset of paths ignored by Git ignore rules."""

        if self.git_root is None:
            return set()
        relative_to_path: dict[str, Path] = {}
        for path in paths:
            try:
                relative = path.relative_to(self.git_root)
            except ValueError:
                continue
            relative_text = relative.as_posix()
            if relative_text == ".":
                continue
            relative_to_path[relative_text] = path
        if not relative_to_path:
            return set()

        result = subprocess.run(
            ["git", "-C", str(self.git_root), "check-ignore", "--no-index", "--stdin"],
            input="\n".join(relative_to_path) + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 1:
            return set()
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git check-ignore failed")
        return {
            relative_to_path[line]
            for line in result.stdout.splitlines()
            if line in relative_to_path
        }

    def _is_indexable_path(self, path: Path, check_git: bool) -> bool:
        """Return whether a path is indexable, optionally checking Git ignores."""

        path = path.resolve()
        if not path.is_relative_to(self.root) or is_excluded_path(self.root, path):
            return False
        if check_git and self.is_ignored_by_git(path):
            return False
        try:
            stat = path.stat()
        except OSError:
            return False
        return path.is_file() and stat.st_size <= self.max_file_bytes


def ensure_state_dir(root: Path) -> Path:
    """Create the .smahties state directory and ignore all generated contents."""

    state_dir = root / ".smahties"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    return state_dir


def sha256_hex(data: bytes) -> str:
    """Return a lowercase SHA-256 hex digest for bytes."""

    return hashlib.sha256(data).hexdigest()


def discover_git_root(root: Path) -> Path | None:
    """Return Git's top-level directory for root, or None outside a worktree."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def is_excluded_path(root: Path, path: Path) -> bool:
    """Return whether a path contains an excluded directory or file component."""

    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return any(is_excluded_dir_name(part) or part in EXCLUDED_FILE_NAMES for part in relative.parts)


def is_excluded_dir_name(name: str) -> bool:
    """Return whether a directory name is excluded from indexing."""

    return name in EXCLUDED_DIR_NAMES or name.endswith(EXCLUDED_DIR_NAME_SUFFIXES)
