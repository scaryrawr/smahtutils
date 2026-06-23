from __future__ import annotations

import asyncio
from pathlib import Path

from .indexer import Indexer
from .models import Priority
from .scanner import is_excluded_path


class PollingWatcher:
    """Handle for an asyncio polling watcher task."""

    def __init__(self, task: asyncio.Task[None]) -> None:
        self.task = task

    def cancel(self) -> None:
        """Cancel the watcher task."""

        self.task.cancel()


def start(root: Path, indexer: Indexer, interval_seconds: float = 2.0) -> PollingWatcher:
    """Start polling a root for file changes and enqueue indexing work."""

    task = asyncio.create_task(_poll(root, indexer, interval_seconds))
    return PollingWatcher(task)


async def _poll(root: Path, indexer: Indexer, interval_seconds: float) -> None:
    seen = _snapshot(root, indexer)
    while True:
        await asyncio.sleep(interval_seconds)
        current = _snapshot(root, indexer)
        for path, signature in current.items():
            if seen.get(path) != signature:
                await indexer.enqueue_path(path, Priority.HIGH)
        for removed in set(seen) - set(current):
            await indexer.enqueue_delete(removed)
        seen = current


def _snapshot(root: Path, indexer: Indexer) -> dict[Path, tuple[int, int]]:
    """Return the current discoverable file signatures for a polling cycle."""

    current: dict[Path, tuple[int, int]] = {}
    for path in indexer.scanner.discover_files(root):
        if is_excluded_path(root, path):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        current[path] = (stat.st_mtime_ns, stat.st_size)
    return current
