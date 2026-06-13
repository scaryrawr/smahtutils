from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

CACHE_SCHEMA_VERSION = 1
EXTRACTOR_VERSION = 6
EXTRACTED_CONTENT_FORMAT = "markdown"
DEV_DOCS_SOURCE = "devdocs"

CacheDocsetStatus = Literal["installed"]


@dataclass(frozen=True)
class DocsetSummary:
    """Represent DocsetSummary."""

    source: str
    name: str
    slug: str
    type: str
    version: str | None = None
    release: str | None = None
    mtime: int | None = None
    db_size: int | None = None
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PageManifestEntry:
    """Represent PageManifestEntry."""

    id: str
    name: str
    path: str
    file: str
    format: str
    source_key: str
    type: str | None = None


@dataclass(frozen=True)
class RawFileManifestEntry:
    """Represent RawFileManifestEntry."""

    file: str
    bytes: int
    sha256: str
    url: str | None = None


@dataclass(frozen=True)
class DocsetManifest:
    """Represent DocsetManifest."""

    schema_version: int
    extractor_version: int
    content_format: str
    source: str
    status: str
    slug: str
    name: str
    type: str
    installed_at: str
    updated_at: str
    upstream: dict[str, str]
    raw_files: list[RawFileManifestEntry]
    pages: list[PageManifestEntry]
    skipped_entries: int
    version: str | None = None
    release: str | None = None
    mtime: int | None = None
    db_size: int | None = None


@dataclass(frozen=True)
class CacheManifestDocset:
    """Represent CacheManifestDocset."""

    source: str
    slug: str
    name: str
    type: str
    content_format: str
    installed_at: str
    updated_at: str
    page_count: int
    version: str | None = None
    release: str | None = None
    mtime: int | None = None
    db_size: int | None = None


@dataclass(frozen=True)
class CacheManifest:
    """Represent CacheManifest."""

    schema_version: int
    updated_at: str
    docs: dict[str, CacheManifestDocset]


@dataclass(frozen=True)
class AvailableDocsetsResult:
    """Represent AvailableDocsetsResult."""

    docsets: list[DocsetSummary]
    fetched_at: str
    from_cache: bool
    warnings: list[str]


@dataclass(frozen=True)
class DownloadedFile:
    """Represent DownloadedFile."""

    path: str
    bytes: int
    sha256: str


def to_jsonable(value: Any) -> Any:
    """Implement to jsonable."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            camel_case_key(key): to_jsonable(item)
            for key, item in asdict(value).items()
            if item is not None
        }
    if isinstance(value, dict):
        return {
            camel_case_key(str(key)): to_jsonable(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def from_page_manifest(value: dict[str, Any]) -> PageManifestEntry:
    """Implement from page manifest."""
    return PageManifestEntry(
        id=str(value["id"]),
        name=str(value["name"]),
        path=str(value["path"]),
        type=optional_string(value.get("type")),
        file=str(value["file"]),
        format=str(value["format"]),
        source_key=str(value.get("sourceKey", value.get("source_key", ""))),
    )


def from_docset_manifest(value: dict[str, Any]) -> DocsetManifest:
    """Implement from docset manifest."""
    return DocsetManifest(
        schema_version=int(value["schemaVersion"]),
        extractor_version=int(value["extractorVersion"]),
        content_format=str(value["contentFormat"]),
        source=str(value["source"]),
        status=str(value["status"]),
        slug=str(value["slug"]),
        name=str(value["name"]),
        type=str(value["type"]),
        version=optional_string(value.get("version")),
        release=optional_string(value.get("release")),
        mtime=optional_int(value.get("mtime")),
        db_size=optional_int(value.get("dbSize", value.get("db_size"))),
        installed_at=str(value["installedAt"]),
        updated_at=str(value["updatedAt"]),
        upstream={str(key): str(item) for key, item in value["upstream"].items()},
        raw_files=[
            RawFileManifestEntry(
                file=str(item["file"]),
                url=optional_string(item.get("url")),
                bytes=int(item["bytes"]),
                sha256=str(item["sha256"]),
            )
            for item in value["rawFiles"]
        ],
        pages=[from_page_manifest(item) for item in value["pages"]],
        skipped_entries=int(value["skippedEntries"]),
    )


def from_cache_manifest(value: dict[str, Any]) -> CacheManifest:
    """Implement from cache manifest."""
    docs: dict[str, CacheManifestDocset] = {}
    for slug, item in value.get("docs", {}).items():
        docs[str(slug)] = CacheManifestDocset(
            source=str(item["source"]),
            slug=str(item["slug"]),
            name=str(item["name"]),
            type=str(item["type"]),
            content_format=str(item["contentFormat"]),
            version=optional_string(item.get("version")),
            release=optional_string(item.get("release")),
            mtime=optional_int(item.get("mtime")),
            db_size=optional_int(item.get("dbSize", item.get("db_size"))),
            installed_at=str(item["installedAt"]),
            updated_at=str(item["updatedAt"]),
            page_count=int(item["pageCount"]),
        )
    return CacheManifest(
        schema_version=int(value["schemaVersion"]),
        updated_at=str(value["updatedAt"]),
        docs=docs,
    )


def camel_case_key(key: str) -> str:
    """Implement camel case key."""
    parts = key.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def optional_string(value: object) -> str | None:
    """Implement optional string."""
    return value if isinstance(value, str) else None


def optional_int(value: object) -> int | None:
    """Implement optional int."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None
