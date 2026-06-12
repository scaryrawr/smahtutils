from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ddserve.cache import assert_safe_path_segment, cache_paths
from ddserve.errors import DdserveError
from ddserve.models import DocsetManifest, PageManifestEntry
from ddserve.text import remove_unpaired_surrogates

DEFAULT_CHUNK_MAX_CHARS = 2400
DEFAULT_CHUNK_MIN_CHARS = 200
DEFAULT_CHUNK_OVERLAP_CHARS = 200
DEFAULT_MAX_CHUNKS_PER_PAGE = 512


@dataclass(frozen=True)
class PreparedEmbeddingChunk:
    """Represent PreparedEmbeddingChunk."""

    page: dict[str, object]
    ordinal: int
    content_hash: str
    source_hash: str
    text: str
    token_count: int
    metadata_json: str


@dataclass(frozen=True)
class ChunkedMarkdownPages:
    """Represent ChunkedMarkdownPages."""

    docset: dict[str, object]
    chunks: list[PreparedEmbeddingChunk]
    stats: "ChunkingStats"


@dataclass(frozen=True)
class ChunkingStats:
    """Represent chunk preparation counters."""

    pages: int = 0
    duplicate_pages: int = 0
    truncated_pages: int = 0
    truncated_chunks: int = 0
    small_chunks: int = 0


def chunk_markdown_pages(
    manifest: DocsetManifest,
    cache_root: str | Path,
    slug: str | None = None,
    max_chunk_chars: int | None = None,
    overlap_chars: int | None = None,
    min_chunk_chars: int | None = None,
    max_chunks_per_page: int | None = None,
) -> ChunkedMarkdownPages:
    """Implement chunk markdown pages."""
    slug = slug or manifest.slug
    assert_safe_path_segment(slug, "docset slug")
    if slug != manifest.slug:
        raise DdserveError(f'Manifest slug "{manifest.slug}" does not match docset slug "{slug}"')
    max_chars, overlap, min_chars, page_chunk_limit = normalize_chunk_options(
        max_chunk_chars, overlap_chars, min_chunk_chars, max_chunks_per_page
    )
    chunks: list[PreparedEmbeddingChunk] = []
    seen_body_hashes: set[str] = set()
    duplicate_pages = 0
    truncated_pages = 0
    truncated_chunks = 0
    small_chunks = 0
    for page in manifest.pages:
        markdown = read_installed_page_markdown(cache_root, slug, page)
        source_text = normalize_markdown_text(markdown)
        body_hash = hash_page_content(strip_generated_page_header(source_text))
        if body_hash in seen_body_hashes:
            duplicate_pages += 1
            continue
        seen_body_hashes.add(body_hash)
        source_hash = hash_page_content(source_text)
        bodies = split_markdown_into_chunks(source_text, max_chars, overlap, min_chars)
        if len(bodies) > page_chunk_limit:
            truncated_pages += 1
            truncated_chunks += len(bodies) - page_chunk_limit
            bodies = bodies[:page_chunk_limit]
        for ordinal, body in enumerate(bodies):
            if len(body) < min_chars:
                small_chunks += 1
            text = normalize_markdown_text(format_chunk_text(manifest, page, body))
            chunks.append(
                PreparedEmbeddingChunk(
                    page=source_page_identity(page, source_hash),
                    ordinal=ordinal,
                    content_hash=hash_chunk_content(text),
                    source_hash=source_hash,
                    text=text,
                    token_count=estimate_token_count(text),
                    metadata_json=chunk_metadata_json(manifest, page, ordinal),
                )
            )
    stats = ChunkingStats(
        pages=len(manifest.pages),
        duplicate_pages=duplicate_pages,
        truncated_pages=truncated_pages,
        truncated_chunks=truncated_chunks,
        small_chunks=small_chunks,
    )
    return ChunkedMarkdownPages(docset=docset_embedding_input(manifest), chunks=chunks, stats=stats)


def read_installed_page_markdown(cache_root: str | Path, slug: str, page: PageManifestEntry) -> str:
    """Read installed page markdown."""
    page_path = resolve_installed_page_path(cache_root, slug, page.file)
    try:
        return page_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DdserveError(
            f'Missing Markdown page file for docset "{slug}" page "{page.id}": {page.file}'
        ) from exc
    except OSError as exc:
        raise DdserveError(f"Failed to read Markdown page file {page.file}: {exc}") from exc


def split_markdown_into_chunks(
    markdown: str,
    max_chunk_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
    min_chunk_chars: int | None = None,
) -> list[str]:
    """Implement split markdown into chunks."""
    max_chars, overlap, min_chars, _max_chunks_per_page = normalize_chunk_options(
        max_chunk_chars, overlap_chars, min_chunk_chars, DEFAULT_MAX_CHUNKS_PER_PAGE
    )
    text = normalize_markdown_text(markdown)
    if not text:
        return []
    chunks: list[str] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        end = (
            len(text)
            if len(text) - cursor <= max_chars
            else find_chunk_end(text, cursor, max_chars)
        )
        chunk = normalize_markdown_text(text[cursor:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_cursor = max(cursor + 1, end - overlap) if overlap > 0 else end
        cursor = end if next_cursor <= cursor else next_cursor
    return merge_small_chunks(chunks, min_chars, max_chars)


def source_page_identity(
    page: PageManifestEntry, content_hash: str | None = None
) -> dict[str, object]:
    """Implement source page identity."""
    return {
        "id": page.id,
        "filePath": page.file,
        "title": page.name,
        "name": page.name,
        "path": page.path,
        "type": page.type,
        "contentHash": content_hash,
    }


def docset_embedding_input(manifest: DocsetManifest) -> dict[str, object]:
    """Implement docset embedding input."""
    return {
        "slug": manifest.slug,
        "name": manifest.name,
        "source": manifest.source,
        "version": manifest.version,
        "release": manifest.release,
        "mtime": manifest.mtime,
        "dbSize": manifest.db_size,
        "contentFormat": manifest.content_format,
        "installedAt": manifest.installed_at,
        "manifestUpdatedAt": manifest.updated_at,
    }


def hash_page_content(text: str) -> str:
    """Implement hash page content."""
    return hash_text_content(normalize_markdown_text(text))


def hash_chunk_content(text: str) -> str:
    """Implement hash chunk content."""
    return hash_text_content(normalize_markdown_text(text))


def hash_text_content(text: str) -> str:
    """Implement hash text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_markdown_text(text: str) -> str:
    """Normalize markdown text."""
    return remove_unpaired_surrogates(text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def estimate_token_count(text: str) -> int:
    """Implement estimate token count."""
    normalized = normalize_markdown_text(text)
    return 0 if not normalized else max(1, (len(normalized) + 3) // 4)


def resolve_installed_page_path(cache_root: str | Path, slug: str, page_file: str) -> Path:
    """Resolve installed page path."""
    assert_safe_path_segment(slug, "docset slug")
    docset_root = (cache_paths(cache_root).docs_root / slug).resolve()
    page_path = (docset_root / page_file).resolve()
    try:
        page_path.relative_to(docset_root)
    except ValueError as exc:
        raise DdserveError(
            f'Invalid Markdown page file path for docset "{slug}": {page_file}'
        ) from exc
    return page_path


def format_chunk_text(manifest: DocsetManifest, page: PageManifestEntry, body: str) -> str:
    """Format chunk text."""
    metadata = [
        f"Docset: {manifest.name} ({manifest.slug})",
        f"Page: {page.name}",
        f"Path: {page.path}",
    ]
    if page.type:
        metadata.append(f"Type: {page.type}")
    return f"{os.linesep.join(metadata)}\n\n{body.strip()}".strip()


def chunk_metadata_json(manifest: DocsetManifest, page: PageManifestEntry, ordinal: int) -> str:
    """Implement chunk metadata json."""
    return json.dumps(
        {
            "docsetSlug": manifest.slug,
            "docsetName": manifest.name,
            "pageId": page.id,
            "pageName": page.name,
            "pagePath": page.path,
            "pageType": page.type,
            "pageFile": page.file,
            "chunkOrdinal": ordinal,
        },
        separators=(",", ":"),
    )


def strip_generated_page_header(text: str) -> str:
    """Implement strip generated page header."""
    return normalize_markdown_text(
        re.sub(
            r"^# .+\n\n> DevDocs path: .+\n\n", "", normalize_markdown_text(text), flags=re.DOTALL
        )
    )


def normalize_chunk_options(
    max_chunk_chars: int | None,
    overlap_chars: int | None,
    min_chunk_chars: int | None = None,
    max_chunks_per_page: int | None = None,
) -> tuple[int, int, int, int]:
    """Normalize chunk options."""
    max_chars = DEFAULT_CHUNK_MAX_CHARS if max_chunk_chars is None else max_chunk_chars
    min_chars = DEFAULT_CHUNK_MIN_CHARS if min_chunk_chars is None else min_chunk_chars
    overlap = DEFAULT_CHUNK_OVERLAP_CHARS if overlap_chars is None else overlap_chars
    page_chunk_limit = (
        DEFAULT_MAX_CHUNKS_PER_PAGE if max_chunks_per_page is None else max_chunks_per_page
    )
    if not isinstance(max_chars, int) or max_chars <= 0:
        raise DdserveError("Invalid max chunk size: expected a positive integer")
    if not isinstance(min_chars, int) or min_chars <= 0:
        raise DdserveError("Invalid min chunk size: expected a positive integer")
    if not isinstance(overlap, int) or overlap < 0:
        raise DdserveError("Invalid chunk overlap: expected a non-negative integer")
    if not isinstance(page_chunk_limit, int) or page_chunk_limit <= 0:
        raise DdserveError("Invalid max chunks per page: expected a positive integer")
    if min_chunk_chars is None and min_chars > max_chars:
        min_chars = max_chars
    if min_chars > max_chars:
        raise DdserveError("Invalid min chunk size: must not exceed max chunk size")
    if overlap >= max_chars:
        raise DdserveError("Invalid chunk overlap: must be smaller than max chunk size")
    return max_chars, overlap, min_chars, page_chunk_limit


def merge_small_chunks(chunks: list[str], min_chunk_chars: int, max_chunk_chars: int) -> list[str]:
    """Merge undersized chunks when doing so does not exceed the maximum size."""
    if len(chunks) <= 1:
        return chunks
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chunk_chars:
            candidate = normalize_markdown_text(f"{merged[-1]}\n\n{chunk}")
            if len(candidate) <= max_chunk_chars:
                merged[-1] = candidate
                continue
        merged.append(chunk)
    if len(merged) > 1 and len(merged[0]) < min_chunk_chars:
        candidate = normalize_markdown_text(f"{merged[0]}\n\n{merged[1]}")
        if len(candidate) <= max_chunk_chars:
            return [candidate, *merged[2:]]
    return merged


def find_chunk_end(text: str, cursor: int, max_chunk_chars: int) -> int:
    """Implement find chunk end."""
    desired = min(len(text), cursor + max_chunk_chars)
    minimum = cursor + max_chunk_chars // 2
    paragraph = text.rfind("\n\n", cursor, desired)
    if paragraph >= minimum:
        return paragraph + 2
    heading = text.rfind("\n#", cursor, desired)
    if heading >= minimum:
        return heading
    sentence = find_last_sentence_break(text, cursor, desired)
    if sentence >= minimum:
        return sentence
    line = text.rfind("\n", cursor, desired)
    if line >= minimum:
        return line + 1
    space = text.rfind(" ", cursor, desired)
    if space >= minimum:
        return space + 1
    return desired


def find_last_sentence_break(text: str, cursor: int, desired_end: int) -> int:
    """Implement find last sentence break."""
    last = -1
    for match in re.finditer(r"""[.!?][)\]"'`]*\s+""", text[cursor:desired_end]):
        last = cursor + match.end()
    return last
