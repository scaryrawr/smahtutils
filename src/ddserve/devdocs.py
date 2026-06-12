from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .cache import atomic_write_json, ensure_cache_root, read_json_file
from .errors import DdserveError, get_error_message
from .http import FetchHttpClient, HttpClient
from .models import DEV_DOCS_SOURCE, AvailableDocsetsResult, DocsetSummary

DEV_DOCS_INDEX_URL = "https://devdocs.io/docs.json"
DEV_DOCS_DOCUMENTS_BASE_URL = "https://documents.devdocs.io"


def get_available_docsets(
    cache_root: str,
    http: HttpClient | None = None,
    offline: bool = False,
    now: datetime | None = None,
) -> AvailableDocsetsResult:
    """Return available docsets."""
    paths = ensure_cache_root(cache_root)
    warnings: list[str] = []
    timestamp = (now or datetime.now(UTC)).isoformat().replace("+00:00", "Z")
    if not offline:
        try:
            raw = (http or FetchHttpClient()).fetch_json(DEV_DOCS_INDEX_URL)
            docsets = normalize_docsets(raw)
            atomic_write_json(
                paths.devdocs_source_index,
                {"fetchedAt": timestamp, "url": DEV_DOCS_INDEX_URL, "docsets": raw},
            )
            return AvailableDocsetsResult(
                docsets=docsets, fetched_at=timestamp, from_cache=False, warnings=[]
            )
        except Exception as exc:
            warnings.append(
                f"Failed to refresh DevDocs index; using cached index if available. {get_error_message(exc)}"
            )
    cached = read_json_file(paths.devdocs_source_index)
    if not isinstance(cached, dict) or not isinstance(cached.get("docsets"), list):
        mode = "Offline mode requested" if offline else "DevDocs index refresh failed"
        raise DdserveError(
            f"{mode}, and no cached DevDocs index exists at {paths.devdocs_source_index}"
        )
    return AvailableDocsetsResult(
        docsets=normalize_docsets(cached["docsets"]),
        fetched_at=str(cached.get("fetchedAt") or timestamp),
        from_cache=True,
        warnings=warnings,
    )


def find_docset(docsets: list[DocsetSummary], slug: str) -> DocsetSummary | None:
    """Implement find docset."""
    return next((docset for docset in docsets if docset.slug == slug), None)


def docset_index_url(slug: str) -> str:
    """Implement docset index url."""
    return f"{DEV_DOCS_DOCUMENTS_BASE_URL}/{slug}/index.json"


def docset_db_url(slug: str) -> str:
    """Implement docset db url."""
    return f"{DEV_DOCS_DOCUMENTS_BASE_URL}/{slug}/db.json"


def normalize_docsets(raw_docsets: object) -> list[DocsetSummary]:
    """Normalize docsets."""
    if not isinstance(raw_docsets, list):
        raise DdserveError("DevDocs index did not contain a docset array")
    docsets: list[DocsetSummary] = []
    for raw in raw_docsets:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        slug = raw.get("slug")
        if not isinstance(name, str) or not name or not isinstance(slug, str) or not slug:
            continue
        docsets.append(
            DocsetSummary(
                source=DEV_DOCS_SOURCE,
                name=name,
                slug=slug,
                type=str(raw.get("type") or slug),
                version=string_or_none(raw.get("version")),
                release=string_or_none(raw.get("release")),
                mtime=int_or_none(raw.get("mtime")),
                db_size=int_or_none(raw.get("db_size")),
                aliases=normalize_aliases(raw.get("alias")),
            )
        )
    return sorted(docsets, key=lambda item: item.slug)


def normalize_aliases(alias: object) -> list[str]:
    """Normalize aliases."""
    if isinstance(alias, str):
        return [alias]
    if isinstance(alias, list):
        return [item for item in alias if isinstance(item, str)]
    return []


def string_or_none(value: Any) -> str | None:
    """Implement string or none."""
    return value if isinstance(value, str) and value else None


def int_or_none(value: Any) -> int | None:
    """Implement int or none."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None
