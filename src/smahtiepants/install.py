from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .aliases import resolve_installed_docset_slug
from .cache import (
    acquire_docset_lock,
    assert_safe_path_segment,
    atomic_write_json,
    cache_paths,
    ensure_cache_root,
    read_cache_manifest,
    read_docset_manifest,
    replace_directory,
    write_cache_manifest,
)
from .config import SmahtiepantsConfig, load_config
from .devdocs import (
    DEV_DOCS_INDEX_URL,
    docset_db_url,
    docset_index_url,
    find_docset,
    get_available_docsets,
)
from .embeddings.annoy_index import AnnoyIndexManager
from .embeddings.index import refresh_docset_embeddings
from .embeddings.openai import AsyncEmbeddingClient, EmbeddingClient
from .embeddings.storage import open_embedding_storage
from .errors import SmahtiepantsError, get_error_message
from .http import FetchHttpClient, HttpClient
from .models import (
    CACHE_SCHEMA_VERSION,
    DEV_DOCS_SOURCE,
    EXTRACTED_CONTENT_FORMAT,
    EXTRACTOR_VERSION,
    CacheManifestDocset,
    DocsetManifest,
    RawFileManifestEntry,
    to_jsonable,
)
from .text import extract_markdown_pages

InstallProgressCallback = Callable[[str, int, int, str, object | None], None]
EmbeddingProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class InstallResult:
    """Represent InstallResult."""

    slug: str
    name: str
    status: str
    pages: int
    skipped_entries: int
    warnings: list[str]
    embedding_chunks: int = 0
    embedded_chunks: int = 0
    skipped_embedding_chunks: int = 0
    annoy_indexed: bool = False


@dataclass(frozen=True)
class RemoveResult:
    """Represent RemoveResult."""

    slug: str
    name: str
    pages: int


def install_docset(
    slug: str,
    cache_root: str,
    http: HttpClient | None = None,
    force: bool = False,
    offline: bool = False,
    now: datetime | None = None,
    config_path: str | None = None,
    config: SmahtiepantsConfig | None = None,
    env: dict[str, str] | None = None,
    embedding_client: EmbeddingClient | None = None,
    async_embedding_client: AsyncEmbeddingClient | None = None,
    ensure_annoy: bool = True,
    on_embedding_progress: EmbeddingProgressCallback | None = None,
) -> InstallResult:
    """Implement install docset."""
    paths = ensure_cache_root(cache_root)
    available = get_available_docsets(cache_root, http=http, offline=offline, now=now)
    summary = find_docset(available.docsets, slug)
    if summary is None:
        raise SmahtiepantsError(
            f'Unknown DevDocs docset "{slug}". Run "smahtiepants docs available" to list valid slugs.'
        )
    canonical_slug = summary.slug
    assert_safe_path_segment(canonical_slug, "docset slug")
    resolved_config = config or load_config(config_path, env).config
    existing = read_docset_manifest(cache_root, canonical_slug)
    if not force and is_current(existing, summary):
        warnings = [*available.warnings]
        embedding_result = refresh_embeddings_for_installed_docset(
            cache_root,
            existing,
            resolved_config,
            warnings,
            env,
            embedding_client,
            async_embedding_client,
            ensure_annoy,
            on_embedding_progress,
        )
        return InstallResult(
            slug=canonical_slug,
            name=summary.name,
            status="skipped",
            pages=len(existing.pages),
            skipped_entries=existing.skipped_entries,
            warnings=warnings,
            embedding_chunks=embedding_result["chunks"],
            embedded_chunks=embedding_result["embedded"],
            skipped_embedding_chunks=embedding_result.get("skipped", 0),
            annoy_indexed=bool(embedding_result.get("annoy", 0)),
        )
    lock = acquire_docset_lock(cache_root, canonical_slug)
    stage_dir = (
        paths.docs_root
        / f"{canonical_slug}.partial-{os.getpid()}-{int(datetime.now().timestamp() * 1000)}"
    )
    try:
        raw_dir = stage_dir / "raw"
        pages_dir = stage_dir / "pages"
        raw_dir.mkdir(parents=True, exist_ok=True)
        pages_dir.mkdir(parents=True, exist_ok=True)
        client = http or FetchHttpClient()
        index_url = docset_index_url(canonical_slug)
        db_url = docset_db_url(canonical_slug)
        raw_files: list[RawFileManifestEntry] = []
        atomic_write_json(raw_dir / "docset.json", to_jsonable(summary))
        raw_files.append(file_manifest_entry(raw_dir / "docset.json", "raw/docset.json"))
        downloaded_index = client.download_file(index_url, raw_dir / "index.json")
        raw_files.append(
            RawFileManifestEntry(
                file="raw/index.json",
                url=index_url,
                bytes=downloaded_index.bytes,
                sha256=downloaded_index.sha256,
            )
        )
        downloaded_db = client.download_file(db_url, raw_dir / "db.json")
        raw_files.append(
            RawFileManifestEntry(
                file="raw/db.json",
                url=db_url,
                bytes=downloaded_db.bytes,
                sha256=downloaded_db.sha256,
            )
        )
        index = json.loads((raw_dir / "index.json").read_text(encoding="utf-8"))
        db = json.loads((raw_dir / "db.json").read_text(encoding="utf-8"))
        if not isinstance(index, dict) or not isinstance(db, dict):
            raise SmahtiepantsError(f'Downloaded "{canonical_slug}", but DevDocs data was invalid')
        extracted = extract_markdown_pages(
            index, {str(key): str(value) for key, value in db.items()}, pages_dir
        )
        pages = extracted["pages"]
        if not pages:
            raise SmahtiepantsError(
                f'Downloaded "{canonical_slug}", but no pages could be extracted'
            )
        timestamp = (now or datetime.now(UTC)).isoformat().replace("+00:00", "Z")
        manifest = DocsetManifest(
            schema_version=CACHE_SCHEMA_VERSION,
            extractor_version=EXTRACTOR_VERSION,
            content_format=EXTRACTED_CONTENT_FORMAT,
            source=DEV_DOCS_SOURCE,
            status="installed",
            slug=summary.slug,
            name=summary.name,
            type=summary.type,
            version=summary.version,
            release=summary.release,
            mtime=summary.mtime,
            db_size=summary.db_size,
            installed_at=existing.installed_at if existing else timestamp,
            updated_at=timestamp,
            upstream={"docsIndexUrl": DEV_DOCS_INDEX_URL, "indexUrl": index_url, "dbUrl": db_url},
            raw_files=raw_files,
            pages=pages,
            skipped_entries=int(extracted["skippedEntries"]),
        )
        atomic_write_json(stage_dir / "manifest.json", to_jsonable(manifest))
        replace_directory(stage_dir, paths.docs_root / canonical_slug)
        update_top_level_manifest(cache_root, manifest)
        warnings = [*available.warnings]
        embedding_result = refresh_embeddings_for_installed_docset(
            cache_root,
            manifest,
            resolved_config,
            warnings,
            env,
            embedding_client,
            async_embedding_client,
            ensure_annoy,
            on_embedding_progress,
        )
        return InstallResult(
            slug=canonical_slug,
            name=summary.name,
            status="updated" if existing else "installed",
            pages=len(pages),
            skipped_entries=manifest.skipped_entries,
            warnings=warnings,
            embedding_chunks=embedding_result["chunks"],
            embedded_chunks=embedding_result["embedded"],
            skipped_embedding_chunks=embedding_result.get("skipped", 0),
            annoy_indexed=bool(embedding_result.get("annoy", 0)),
        )
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        lock.release()


def install_docsets(
    slugs: list[str],
    cache_root: str,
    on_progress: InstallProgressCallback | None = None,
    **kwargs: object,
) -> list[InstallResult]:
    """Implement install docsets."""
    results: list[InstallResult] = []
    total = len(slugs)
    for index, slug in enumerate(slugs, start=1):
        if on_progress:
            on_progress(slug, index, total, "start", None)
        install_kwargs = dict(kwargs)
        if on_progress:
            install_kwargs["on_embedding_progress"] = (
                lambda completed, batches, current_slug=slug, current_index=index: on_progress(
                    current_slug,
                    current_index,
                    total,
                    "embedding",
                    {"completed": completed, "total": batches},
                )
            )
        result = install_docset(slug, cache_root, **install_kwargs)
        if on_progress:
            on_progress(result.slug, index, total, "done", result)
        results.append(result)
    return results


def update_docsets(
    slug: str | None,
    cache_root: str,
    on_progress: InstallProgressCallback | None = None,
    **kwargs: object,
) -> list[InstallResult]:
    """Implement update docsets."""
    if slug:
        if on_progress:
            on_progress(slug, 1, 1, "start", None)
        install_kwargs = dict(kwargs)
        if on_progress:
            install_kwargs["on_embedding_progress"] = lambda completed, batches: on_progress(
                slug, 1, 1, "embedding", {"completed": completed, "total": batches}
            )
        result = install_docset(slug, cache_root, **install_kwargs)
        if on_progress:
            on_progress(result.slug, 1, 1, "done", result)
        return [result]
    manifest = read_cache_manifest(cache_root)
    slugs = sorted(manifest.docs)
    if not slugs:
        return []
    resolved_config = resolve_update_config(kwargs)
    defer_annoy = (
        len(slugs) > 1 and resolved_config.embeddings.enabled and resolved_config.openai is not None
    )
    install_kwargs = dict(kwargs)
    install_kwargs["config"] = resolved_config
    if defer_annoy:
        install_kwargs["ensure_annoy"] = False
    results: list[InstallResult] = []
    total = len(slugs)
    for index, installed_slug in enumerate(slugs, start=1):
        if on_progress:
            on_progress(installed_slug, index, total, "start", None)
        per_doc_kwargs = dict(install_kwargs)
        if on_progress:
            per_doc_kwargs["on_embedding_progress"] = (
                lambda completed, batches, current_slug=installed_slug, current_index=index: (
                    on_progress(
                        current_slug,
                        current_index,
                        total,
                        "embedding",
                        {"completed": completed, "total": batches},
                    )
                )
            )
        result = install_docset(installed_slug, cache_root, **per_doc_kwargs)
        if on_progress:
            on_progress(installed_slug, index, total, "done", result)
        results.append(result)
    if defer_annoy:
        results = ensure_deferred_annoy_index(cache_root, resolved_config, results)
    return results


def remove_docset(slug: str, cache_root: str) -> RemoveResult:
    """Implement remove docset."""
    canonical_slug = resolve_installed_docset_slug(cache_root, slug) or slug
    assert_safe_path_segment(canonical_slug, "docset slug")
    paths = ensure_cache_root(cache_root)
    lock = acquire_docset_lock(cache_root, canonical_slug)
    try:
        manifest = read_cache_manifest(cache_root)
        docset = manifest.docs.get(canonical_slug)
        if docset is None:
            raise SmahtiepantsError(f'Docset "{slug}" is not installed.')
        shutil.rmtree(paths.docs_root / canonical_slug, ignore_errors=True)
        manifest.docs.pop(canonical_slug, None)
        manifest = type(manifest)(
            schema_version=manifest.schema_version,
            updated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            docs=manifest.docs,
        )
        write_cache_manifest(cache_root, manifest)
        return RemoveResult(slug=canonical_slug, name=docset.name, pages=docset.page_count)
    finally:
        lock.release()


def resolve_update_config(kwargs: dict[str, object]) -> SmahtiepantsConfig:
    """Resolve docs update config once for a multi-docset run."""
    config = kwargs.get("config")
    if isinstance(config, SmahtiepantsConfig):
        return config
    config_path = kwargs.get("config_path")
    env = kwargs.get("env")
    return load_config(
        config_path if isinstance(config_path, str) else None,
        env if isinstance(env, dict) else None,
    ).config


def ensure_deferred_annoy_index(
    cache_root: str,
    config: SmahtiepantsConfig,
    results: list[InstallResult],
) -> list[InstallResult]:
    """Build the Annoy sidecar once after a multi-docset embedding update."""
    if config.openai is None:
        return results
    try:
        storage = open_embedding_storage(cache_root)
        try:
            ready = AnnoyIndexManager(cache_root, storage).ensure(config.openai.embedding_model)
        finally:
            storage.close()
    except Exception as exc:
        warning = (
            "Failed to refresh embedding search index; docs remain installed. "
            f"{get_error_message(exc)}"
        )
        return [replace(result, warnings=[*result.warnings, warning]) for result in results]
    return [replace(result, annoy_indexed=bool(ready)) for result in results]


def is_current(existing: DocsetManifest | None, summary: object) -> bool:
    """Return whether current."""
    if (
        existing is None
        or existing.extractor_version < EXTRACTOR_VERSION
        or existing.content_format != EXTRACTED_CONTENT_FORMAT
    ):
        return False
    mtime = getattr(summary, "mtime", None)
    if isinstance(mtime, int) and isinstance(existing.mtime, int):
        return existing.mtime >= mtime
    return existing.release == getattr(summary, "release", None) and existing.version == getattr(
        summary, "version", None
    )


def refresh_embeddings_for_installed_docset(
    cache_root: str,
    manifest: DocsetManifest,
    config: SmahtiepantsConfig,
    warnings: list[str],
    env: dict[str, str] | None,
    embedding_client: EmbeddingClient | None,
    async_embedding_client: AsyncEmbeddingClient | None,
    ensure_annoy: bool,
    on_embedding_progress: EmbeddingProgressCallback | None,
) -> dict[str, int]:
    """Implement refresh embeddings for installed docset."""
    try:
        return refresh_docset_embeddings(
            cache_root,
            manifest,
            config,
            env=env,
            client=embedding_client,
            async_client=async_embedding_client,
            ensure_annoy=ensure_annoy,
            on_embedding_progress=on_embedding_progress,
        )
    except Exception as exc:
        warnings.append(
            f"Failed to refresh embeddings for {manifest.slug}; docs remain installed. {get_error_message(exc)}"
        )
        return {"chunks": 0, "embedded": 0, "skipped": 0, "annoy": 0}


def file_manifest_entry(path: Path, manifest_file: str) -> RawFileManifestEntry:
    """Implement file manifest entry."""
    data = path.read_bytes()
    return RawFileManifestEntry(
        file=manifest_file, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
    )


def update_top_level_manifest(cache_root: str, docset: DocsetManifest) -> None:
    """Implement update top level manifest."""
    manifest = read_cache_manifest(cache_root)
    docs = dict(manifest.docs)
    docs[docset.slug] = CacheManifestDocset(
        source=docset.source,
        slug=docset.slug,
        name=docset.name,
        type=docset.type,
        content_format=docset.content_format,
        version=docset.version,
        release=docset.release,
        mtime=docset.mtime,
        db_size=docset.db_size,
        installed_at=docset.installed_at,
        updated_at=docset.updated_at,
        page_count=len(docset.pages),
    )
    write_cache_manifest(
        cache_root,
        type(manifest)(
            schema_version=manifest.schema_version, updated_at=docset.updated_at, docs=docs
        ),
    )


def cleanup_partial_docset_dirs(cache_root: str) -> None:
    """Implement cleanup partial docset dirs."""
    docs_root = cache_paths(cache_root).docs_root
    if not docs_root.exists():
        return
    for item in docs_root.iterdir():
        if ".partial-" in item.name and item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
