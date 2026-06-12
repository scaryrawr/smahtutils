from __future__ import annotations

import asyncio
from pathlib import Path

from .indexer import Indexer
from .models import Priority
from .scanner import is_excluded_path


class PollingWatcher:
    def __init__(self, task: asyncio.Task[None]) -> None:
        self.task = task

    def cancel(self) -> None:
        self.task.cancel()


def start(root: Path, indexer: Indexer, interval_seconds: float = 2.0) -> PollingWatcher:
    task = asyncio.create_task(_poll(root, indexer, interval_seconds))
    return PollingWatcher(task)


async def _poll(root: Path, indexer: Indexer, interval_seconds: float) -> None:
    seen: dict[Path, tuple[int, int]] = {}
    while True:
        current: dict[Path, tuple[int, int]] = {}
        for path in indexer.scanner.discover_files(root):
            if is_excluded_path(root, path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            signature = (stat.st_mtime_ns, stat.st_size)
            current[path] = signature
            if seen.get(path) != signature:
                await indexer.enqueue_path(path, Priority.HIGH)
        for removed in set(seen) - set(current):
            await indexer.enqueue_delete(removed)
        seen = current
        await asyncio.sleep(interval_seconds)
