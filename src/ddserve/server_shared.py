from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cache import read_cache_manifest, read_docset_manifest
from .embeddings.chunks import read_installed_page_markdown
from .errors import DdserveError
from .models import CacheManifestDocset, DocsetManifest, PageManifestEntry, to_jsonable


@dataclass(frozen=True)
class PageContent:
    """Represent PageContent."""

    content: str
    start_line: int
    end_line: int
    total_lines: int


def list_docsets(cache_root: str | Path) -> list[CacheManifestDocset]:
    """Implement list docsets."""
    return [docset for _slug, docset in sorted(read_cache_manifest(cache_root).docs.items())]


def get_docset(cache_root: str | Path, slug: str) -> DocsetManifest:
    """Return docset."""
    manifest = read_docset_manifest(cache_root, slug)
    if manifest is None:
        raise DdserveError(f'Docset "{slug}" is not installed.')
    return manifest


def list_pages(
    cache_root: str | Path,
    slug: str,
    limit: int = 100,
    offset: int = 0,
    query: str | None = None,
    type_: str | None = None,
) -> dict[str, object]:
    """Implement list pages."""
    manifest = get_docset(cache_root, slug)
    pages = manifest.pages
    if query:
        q = query.lower()
        pages = [page for page in pages if q in page.name.lower() or q in page.path.lower()]
    if type_:
        pages = [page for page in pages if page.type == type_]
    bounded_limit = max(1, min(limit, 500))
    return {
        "slug": slug,
        "items": [to_jsonable(page) for page in pages[offset : offset + bounded_limit]],
        "limit": bounded_limit,
        "offset": max(0, offset),
        "total": len(pages),
    }


def get_page(cache_root: str | Path, slug: str, page_id: str) -> PageManifestEntry:
    """Return page."""
    manifest = get_docset(cache_root, slug)
    for page in manifest.pages:
        if page.id == page_id:
            return page
    raise DdserveError(f'Page "{page_id}" is not installed for docset "{slug}".')


def get_page_content(
    cache_root: str | Path,
    slug: str,
    page_id: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> PageContent:
    """Return page content."""
    page = get_page(cache_root, slug, page_id)
    content = read_installed_page_markdown(cache_root, slug, page)
    lines = content.splitlines()
    total = len(lines)
    start = max(1, start_line or 1)
    end = min(total, end_line or total)
    if end < start:
        selected: list[str] = []
    else:
        selected = lines[start - 1 : end]
    return PageContent(
        content="\n".join(selected) + ("\n" if selected else ""),
        start_line=start,
        end_line=end,
        total_lines=total,
    )
