from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ddserve.cache import read_cache_manifest, read_docset_manifest
from ddserve.config import DdserveConfig
from ddserve.errors import DdserveError
from ddserve.models import DocsetManifest

from .annoy_index import AnnoyIndexManager
from .chunks import ChunkedMarkdownPages, chunk_markdown_pages
from .openai import (
    AsyncEmbeddingClient,
    EmbeddingBatchLimits,
    EmbeddingClient,
    create_openai_async_embedding_client,
    embedding_batch_ranges,
)
from .storage import open_embedding_storage


@dataclass(frozen=True)
class EmbeddingStatus:
    """Represent EmbeddingStatus."""

    database_path: str
    enabled: bool
    configured: bool
    installed_docsets: int
    installed_pages: int
    indexed_docsets: int
    indexed_pages: int
    indexed_chunks: int
    embedded_chunks: int
    model: str | None


def refresh_docset_embeddings(
    cache_root: str | Path,
    manifest: DocsetManifest,
    config: DdserveConfig,
    env: dict[str, str] | None = None,
    client: EmbeddingClient | None = None,
    force: bool = False,
    *,
    async_client: AsyncEmbeddingClient | None = None,
    ensure_annoy: bool = True,
    on_embedding_progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Implement refresh docset embeddings."""
    if not config.embeddings.enabled:
        return {"chunks": 0, "embedded": 0}
    if config.openai is None:
        raise DdserveError("OpenAI embeddings are not configured")
    chunked = chunk_markdown_pages(
        manifest,
        cache_root,
        max_chunk_chars=config.embeddings.max_chunk_chars,
        overlap_chars=config.embeddings.overlap_chars,
        min_chunk_chars=config.embeddings.min_chunk_chars,
        max_chunks_per_page=config.embeddings.max_chunks_per_page,
    )
    storage = open_embedding_storage(cache_root)
    try:
        if not force and storage.docset_embeddings_current(
            manifest.slug, chunked.chunks, config.openai.embedding_model
        ):
            annoy_ready = (
                AnnoyIndexManager(cache_root, storage).ensure(config.openai.embedding_model)
                if ensure_annoy
                else False
            )
            return {
                **chunking_result_counts(chunked),
                "chunks": len(chunked.chunks),
                "embedded": 0,
                "skipped": len(chunked.chunks),
                "annoy": int(annoy_ready),
            }
    finally:
        storage.close()
    vectors = create_docset_embedding_vectors(
        chunked, config, env, client, async_client, on_embedding_progress
    )
    storage = open_embedding_storage(cache_root)
    try:
        storage.replace_docset_chunks(
            chunked.docset, chunked.chunks, vectors, config.openai.embedding_model
        )
        annoy_ready = (
            AnnoyIndexManager(cache_root, storage).ensure(config.openai.embedding_model)
            if ensure_annoy
            else False
        )
    finally:
        storage.close()
    return {
        **chunking_result_counts(chunked),
        "chunks": len(chunked.chunks),
        "embedded": len(vectors),
        "annoy": int(annoy_ready),
    }


def rebuild_docset_embeddings(
    cache_root: str | Path,
    manifest: DocsetManifest,
    config: DdserveConfig,
    env: dict[str, str] | None = None,
    client: EmbeddingClient | None = None,
    *,
    async_client: AsyncEmbeddingClient | None = None,
    ensure_annoy: bool = True,
    on_embedding_progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Implement rebuild docset embeddings."""
    if not config.embeddings.enabled or config.openai is None:
        raise DdserveError("OpenAI embeddings are not configured")
    return refresh_docset_embeddings(
        cache_root,
        manifest,
        config,
        env,
        client,
        force=True,
        async_client=async_client,
        ensure_annoy=ensure_annoy,
        on_embedding_progress=on_embedding_progress,
    )


def status_for_embeddings(
    cache_root: str | Path,
    config: DdserveConfig,
    slug: str | None = None,
) -> EmbeddingStatus:
    """Implement status for embeddings."""
    manifest = read_cache_manifest(cache_root)
    docs = {slug: manifest.docs[slug]} if slug and slug in manifest.docs else manifest.docs
    installed_pages = sum(doc.page_count for doc in docs.values())
    storage = open_embedding_storage(cache_root)
    try:
        stats = storage.stats(slug)
        return EmbeddingStatus(
            database_path=str(storage.path),
            enabled=config.embeddings.enabled,
            configured=config.openai is not None,
            installed_docsets=len(docs),
            installed_pages=installed_pages,
            indexed_docsets=stats["indexedDocsets"],
            indexed_pages=stats["indexedPages"],
            indexed_chunks=stats["indexedChunks"],
            embedded_chunks=stats["embeddedChunks"],
            model=config.openai.embedding_model if config.openai else None,
        )
    finally:
        storage.close()


def refresh_installed_slug(
    cache_root: str | Path,
    slug: str,
    config: DdserveConfig,
    env: dict[str, str] | None = None,
    client: EmbeddingClient | None = None,
) -> dict[str, int]:
    """Implement refresh installed slug."""
    manifest = read_docset_manifest(cache_root, slug)
    if manifest is None:
        raise DdserveError(f'Docset "{slug}" is not installed.')
    return refresh_docset_embeddings(cache_root, manifest, config, env, client)


def create_docset_embedding_vectors(
    chunked: ChunkedMarkdownPages,
    config: DdserveConfig,
    env: dict[str, str] | None,
    client: EmbeddingClient | None,
    async_client: AsyncEmbeddingClient | None,
    on_embedding_progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Create vectors for prepared chunks using configured request bounds."""
    texts = [chunk.text for chunk in chunked.chunks]
    if not texts:
        return []
    limits = EmbeddingBatchLimits(
        max_inputs=config.embeddings.batch_size,
        max_request_bytes=config.embeddings.max_request_bytes,
    )
    if client is not None:
        return create_embedding_vectors_sync(texts, limits, client, on_embedding_progress)
    resolved_async_client = async_client or create_openai_async_embedding_client(config, env)
    return run_async_embedding_batches(
        create_embedding_vectors_async(
            texts,
            limits,
            resolved_async_client,
            config.embeddings.max_concurrent_requests,
            on_embedding_progress,
        )
    )


def create_embedding_vectors_sync(
    texts: list[str],
    limits: EmbeddingBatchLimits,
    client: EmbeddingClient,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Create vectors with deterministic request-aware sync batching."""
    vectors: list[list[float]] = []
    ranges = embedding_batch_ranges(texts, limits)
    for completed, (start, end) in enumerate(ranges, start=1):
        vectors.extend(client.create_embeddings(texts[start:end]))
        if on_progress is not None:
            on_progress(completed, len(ranges))
    if len(vectors) != len(texts):
        raise DdserveError(
            f"OpenAI embedding response count mismatch: expected {len(texts)}, received {len(vectors)}"
        )
    return vectors


async def create_embedding_vectors_async(
    texts: list[str],
    limits: EmbeddingBatchLimits,
    client: AsyncEmbeddingClient,
    max_concurrent_requests: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Create vectors concurrently while preserving original input order."""
    ranges = embedding_batch_ranges(texts, limits)
    if not ranges:
        return []
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    batches: list[list[list[float]] | None] = [None] * len(ranges)
    completed_batches = 0

    async def create_batch(batch_index: int, start: int, end: int) -> None:
        nonlocal completed_batches
        async with semaphore:
            vectors = await client.create_embeddings(texts[start:end])
        if len(vectors) != end - start:
            raise DdserveError(
                "OpenAI embedding response count mismatch: "
                f"expected {end - start}, received {len(vectors)}"
            )
        batches[batch_index] = vectors
        completed_batches += 1
        if on_progress is not None:
            on_progress(completed_batches, len(ranges))

    await asyncio.gather(
        *(create_batch(index, start, end) for index, (start, end) in enumerate(ranges))
    )
    return [vector for batch in batches if batch is not None for vector in batch]


def run_async_embedding_batches(
    coroutine: Coroutine[Any, Any, list[list[float]]],
) -> list[list[float]]:
    """Run async embedding work from the synchronous ddserve CLI path."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    coroutine.close()
    raise DdserveError("ddserve embedding refresh cannot run inside an active event loop")


def chunking_result_counts(chunked: ChunkedMarkdownPages) -> dict[str, int]:
    """Return chunking counters for status and JSON output."""
    return {
        "pages": chunked.stats.pages,
        "duplicate_pages": chunked.stats.duplicate_pages,
        "truncated_pages": chunked.stats.truncated_pages,
        "truncated_chunks": chunked.stats.truncated_chunks,
        "small_chunks": chunked.stats.small_chunks,
    }
