from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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
from .config import DdserveConfig, load_config
from .devdocs import (
    DEV_DOCS_INDEX_URL,
    docset_db_url,
    docset_index_url,
    find_docset,
    get_available_docsets,
)
from .embeddings.index import refresh_docset_embeddings
from .embeddings.openai import EmbeddingClient
from .errors import DdserveError, get_error_message
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


@dataclass(frozen=True)
class InstallResult:
    """Represent InstallResult."""

    slug: str
    name: str
    status: str
    pages: int
    skipped_entries: int
    warnings: list[str]


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
    config: DdserveConfig | None = None,
    env: dict[str, str] | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> InstallResult:
    """Implement install docset."""
    assert_safe_path_segment(slug, "docset slug")
    paths = ensure_cache_root(cache_root)
    available = get_available_docsets(cache_root, http=http, offline=offline, now=now)
    summary = find_docset(available.docsets, slug)
    if summary is None:
        raise DdserveError(
            f'Unknown DevDocs docset "{slug}". Run "ddserve docs available" to list valid slugs.'
        )
    resolved_config = config or load_config(config_path, env).config
    existing = read_docset_manifest(cache_root, slug)
    if not force and is_current(existing, summary):
        warnings = [*available.warnings]
        refresh_embeddings_for_installed_docset(
            cache_root, existing, resolved_config, warnings, env, embedding_client
        )
        return InstallResult(
            slug=slug,
            name=summary.name,
            status="skipped",
            pages=len(existing.pages),
            skipped_entries=existing.skipped_entries,
            warnings=warnings,
        )
    lock = acquire_docset_lock(cache_root, slug)
    stage_dir = (
        paths.docs_root / f"{slug}.partial-{os.getpid()}-{int(datetime.now().timestamp() * 1000)}"
    )
    try:
        raw_dir = stage_dir / "raw"
        pages_dir = stage_dir / "pages"
        raw_dir.mkdir(parents=True, exist_ok=True)
        pages_dir.mkdir(parents=True, exist_ok=True)
        client = http or FetchHttpClient()
        index_url = docset_index_url(slug)
        db_url = docset_db_url(slug)
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
            raise DdserveError(f'Downloaded "{slug}", but DevDocs data was invalid')
        extracted = extract_markdown_pages(
            index, {str(key): str(value) for key, value in db.items()}, pages_dir
        )
        pages = extracted["pages"]
        if not pages:
            raise DdserveError(f'Downloaded "{slug}", but no pages could be extracted')
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
        replace_directory(stage_dir, paths.docs_root / slug)
        update_top_level_manifest(cache_root, manifest)
        warnings = [*available.warnings]
        refresh_embeddings_for_installed_docset(
            cache_root, manifest, resolved_config, warnings, env, embedding_client
        )
        return InstallResult(
            slug=slug,
            name=summary.name,
            status="updated" if existing else "installed",
            pages=len(pages),
            skipped_entries=manifest.skipped_entries,
            warnings=warnings,
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
        result = install_docset(slug, cache_root, **kwargs)
        if on_progress:
            on_progress(slug, index, total, "done", result)
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
        result = install_docset(slug, cache_root, **kwargs)
        if on_progress:
            on_progress(slug, 1, 1, "done", result)
        return [result]
    manifest = read_cache_manifest(cache_root)
    slugs = sorted(manifest.docs)
    results: list[InstallResult] = []
    total = len(slugs)
    for index, installed_slug in enumerate(slugs, start=1):
        if on_progress:
            on_progress(installed_slug, index, total, "start", None)
        result = install_docset(installed_slug, cache_root, **kwargs)
        if on_progress:
            on_progress(installed_slug, index, total, "done", result)
        results.append(result)
    return results


def remove_docset(slug: str, cache_root: str) -> RemoveResult:
    """Implement remove docset."""
    assert_safe_path_segment(slug, "docset slug")
    paths = ensure_cache_root(cache_root)
    lock = acquire_docset_lock(cache_root, slug)
    try:
        manifest = read_cache_manifest(cache_root)
        docset = manifest.docs.get(slug)
        if docset is None:
            raise DdserveError(f'Docset "{slug}" is not installed.')
        shutil.rmtree(paths.docs_root / slug, ignore_errors=True)
        manifest.docs.pop(slug, None)
        manifest = type(manifest)(
            schema_version=manifest.schema_version,
            updated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            docs=manifest.docs,
        )
        write_cache_manifest(cache_root, manifest)
        return RemoveResult(slug=slug, name=docset.name, pages=docset.page_count)
    finally:
        lock.release()


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
    config: DdserveConfig,
    warnings: list[str],
    env: dict[str, str] | None,
    embedding_client: EmbeddingClient | None,
) -> None:
    """Implement refresh embeddings for installed docset."""
    try:
        refresh_docset_embeddings(cache_root, manifest, config, env=env, client=embedding_client)
    except Exception as exc:
        warnings.append(
            f"Failed to refresh embeddings for {manifest.slug}; docs remain installed. {get_error_message(exc)}"
        )


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
