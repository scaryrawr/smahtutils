from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ddserve.cache import read_cache_manifest, read_docset_manifest
from ddserve.config import DdserveConfig
from ddserve.errors import DdserveError
from ddserve.models import DocsetManifest

from .annoy_index import AnnoyIndexManager
from .chunks import chunk_markdown_pages
from .openai import EmbeddingClient, create_openai_embedding_client
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
    )
    storage = open_embedding_storage(cache_root)
    try:
        if not force and storage.docset_embeddings_current(
            manifest.slug, chunked.chunks, config.openai.embedding_model
        ):
            annoy_ready = AnnoyIndexManager(cache_root, storage).ensure(
                config.openai.embedding_model
            )
            return {
                "chunks": len(chunked.chunks),
                "embedded": 0,
                "skipped": len(chunked.chunks),
                "annoy": int(annoy_ready),
            }
    finally:
        storage.close()
    embedding_client = client or create_openai_embedding_client(config, env)
    vectors: list[list[float]] = []
    for start in range(0, len(chunked.chunks), config.embeddings.batch_size):
        batch = chunked.chunks[start : start + config.embeddings.batch_size]
        vectors.extend(embedding_client.create_embeddings([chunk.text for chunk in batch]))
    storage = open_embedding_storage(cache_root)
    try:
        storage.replace_docset_chunks(
            chunked.docset, chunked.chunks, vectors, config.openai.embedding_model
        )
        annoy_ready = AnnoyIndexManager(cache_root, storage).ensure(config.openai.embedding_model)
    finally:
        storage.close()
    return {"chunks": len(chunked.chunks), "embedded": len(vectors), "annoy": int(annoy_ready)}


def rebuild_docset_embeddings(
    cache_root: str | Path,
    manifest: DocsetManifest,
    config: DdserveConfig,
    env: dict[str, str] | None = None,
    client: EmbeddingClient | None = None,
) -> dict[str, int]:
    """Implement rebuild docset embeddings."""
    if not config.embeddings.enabled or config.openai is None:
        raise DdserveError("OpenAI embeddings are not configured")
    return refresh_docset_embeddings(cache_root, manifest, config, env, client, force=True)


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
