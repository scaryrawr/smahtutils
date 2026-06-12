from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .embedding import OpenAiEmbedder
from .models import CodeUnit, LeaseStatus, Priority, QueueStats, QueuedWork, SourceFile
from .parser import ParserRegistry
from .scanner import Scanner
from .store import Store

INDEXER_LEASE_NAME = "indexer"
INDEXER_LEASE_TTL_SECONDS = 15
WORK_STALE_AFTER_SECONDS = 300
MAX_INDEXER_BATCH_WORK_ITEMS = 128


@dataclass
class IndexRunSummary:
    completed: int = 0
    requeued: int = 0
    failed: int = 0


@dataclass(frozen=True)
class IndexRunOutcome:
    status: str
    summary: IndexRunSummary


@dataclass(frozen=True)
class PreparedFile:
    source_file: SourceFile
    parser_key: str
    units: list[CodeUnit]


@dataclass(frozen=True)
class PreparedEmbeddingWork:
    item: QueuedWork
    claim: "WorkClaim"
    source_file: SourceFile
    parser_key: str
    units: list[CodeUnit]


class Indexer:
    def __init__(
        self, scanner: Scanner, parser: ParserRegistry, store: Store, embedder: OpenAiEmbedder
    ) -> None:
        self.scanner = scanner
        self.parser = parser
        self.store = store
        self.embedder = embedder
        self.owner = f"{os.getpid()}:{int(time.time())}"
        self._notify = asyncio.Event()

    def root(self) -> Path:
        return self.scanner.root

    async def enqueue_requested_path_under(self, requested: str, base: Path) -> None:
        path = self.scanner.resolve_existing_under(base, requested)
        await self.enqueue_path(path, Priority.HIGH)

    async def enqueue_path(self, path: Path, priority: Priority) -> None:
        self.store.enqueue_work(path, priority, False)
        self._notify.set()

    async def enqueue_delete(self, path: Path) -> None:
        self.store.enqueue_work(path, Priority.HIGH, True)
        self._notify.set()

    def spawn_worker(self) -> asyncio.Task[None]:
        return asyncio.create_task(self.worker_loop())

    async def run_until_idle_or_interrupt(self) -> IndexRunOutcome:
        summary = IndexRunSummary()
        while True:
            if not self.acquire_lease():
                await asyncio.sleep(2)
                continue
            outcome = await self.process_next_work()
            if outcome is None:
                return IndexRunOutcome("complete", summary)
            summary.completed += outcome.completed
            summary.requeued += outcome.requeued
            summary.failed += outcome.failed

    def queue_stats(self) -> QueueStats:
        return self.store.queue_stats()

    def lease_status(self) -> LeaseStatus:
        return self.store.lease_status(INDEXER_LEASE_NAME, self.owner)

    async def worker_loop(self) -> None:
        while True:
            try:
                if not self.acquire_lease():
                    await asyncio.sleep(2)
                    continue
                outcome = await self.process_next_work()
                if outcome is None:
                    self._notify.clear()
                    try:
                        await asyncio.wait_for(self._notify.wait(), timeout=1)
                    except TimeoutError:
                        pass
            except Exception:
                await asyncio.sleep(1)

    def acquire_lease(self) -> bool:
        return self.store.acquire_lease(INDEXER_LEASE_NAME, self.owner, INDEXER_LEASE_TTL_SECONDS)

    async def process_next_work(self) -> IndexRunSummary | None:
        summary = IndexRunSummary()
        batch: list[PreparedEmbeddingWork] = []
        claimed = 0
        while claimed < MAX_INDEXER_BATCH_WORK_ITEMS:
            item = self.store.claim_next_work(self.owner, WORK_STALE_AFTER_SECONDS)
            if item is None:
                break
            claim = WorkClaim(self.store, item.id, self.owner)
            claimed += 1
            try:
                prepared = await self.prepare_item(item)
                if prepared is None:
                    claim.complete()
                    summary.completed += 1
                else:
                    batch.append(
                        PreparedEmbeddingWork(
                            item=item,
                            claim=claim,
                            source_file=prepared.source_file,
                            parser_key=prepared.parser_key,
                            units=prepared.units,
                        )
                    )
            except Exception as exc:
                self.fail_claimed_work(item, claim, exc, summary)

        if claimed == 0:
            return None
        await self.embed_and_commit_batch(batch, summary)
        return summary

    async def prepare_item(self, item: QueuedWork) -> PreparedFile | None:
        if item.delete or not item.path.exists():
            rel = self.scanner.relative_path(item.path)
            self.store.delete_path_prefix(rel)
            return None
        if item.path.is_dir():
            for path in self.scanner.discover_files(item.path):
                await self.enqueue_path(path, item.priority)
            return None
        if not self.scanner.is_discoverable_file(item.path):
            self.store.delete_file(self.scanner.relative_path(item.path))
            return None
        source_file = self.scanner.read_source(item.path)
        if source_file is None:
            self.store.delete_file(self.scanner.relative_path(item.path))
            return None
        parser_key = self.parser.cache_key_for_path(source_file.absolute_path)
        if self.store.file_complete_for_model(
            source_file.relative_path, source_file.hash, parser_key, self.embedder.model
        ):
            return None
        units = self.parser.parse(source_file)
        return PreparedFile(source_file, parser_key, units)

    async def embed_and_commit_batch(
        self,
        batch: list[PreparedEmbeddingWork],
        summary: IndexRunSummary,
    ) -> None:
        if not batch:
            return
        texts = [unit.source for work in batch for unit in work.units]
        try:
            embeddings = await self.embedder.embed_texts(texts)
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"embedding response count {len(embeddings)} did not match unit count {len(texts)}"
                )
        except Exception as exc:
            for work in batch:
                self.fail_claimed_work(work.item, work.claim, exc, summary)
            return

        offset = 0
        for work in batch:
            end = offset + len(work.units)
            file_embeddings = embeddings[offset:end]
            offset = end
            try:
                current_source = self.scanner.read_source(work.item.path)
                if current_source is None:
                    self.store.delete_file(self.scanner.relative_path(work.item.path))
                    work.claim.complete()
                    summary.completed += 1
                    continue
                if current_source.hash != work.source_file.hash:
                    work.claim.requeue("source changed while indexing")
                    summary.requeued += 1
                    continue
                if not self.acquire_lease():
                    work.claim.requeue("indexer lease expired before commit")
                    summary.requeued += 1
                    continue
                self.store.replace_file_units(
                    work.source_file.relative_path,
                    work.source_file.hash,
                    work.parser_key,
                    work.units,
                    self.embedder.model,
                    file_embeddings,
                )
                work.claim.complete()
                summary.completed += 1
            except Exception as exc:
                self.fail_claimed_work(work.item, work.claim, exc, summary)

    def fail_claimed_work(
        self,
        item: QueuedWork,
        claim: "WorkClaim",
        error: Exception,
        summary: IndexRunSummary,
    ) -> None:
        changed = claim.requeue(str(error))
        if changed:
            self.store.mark_error(self.scanner.relative_path(item.path), str(error))
        summary.failed += 1


class WorkClaim:
    def __init__(self, store: Store, id_: int, owner: str) -> None:
        self.store = store
        self.id = id_
        self.owner = owner
        self.active = True

    def complete(self) -> None:
        self.store.complete_work_for_owner(self.id, self.owner)
        self.active = False

    def requeue(self, reason: str) -> bool:
        changed = self.store.fail_work_for_owner(self.id, self.owner, reason)
        self.active = False
        return changed

    def __del__(self) -> None:
        if self.active:
            try:
                self.store.fail_work_for_owner(self.id, self.owner, "indexing interrupted")
            except Exception:
                pass
