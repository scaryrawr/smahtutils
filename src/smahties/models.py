from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class CodeUnit:
    """Parsed source unit that can be indexed, embedded, and returned as a match."""

    id: str
    file_path: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    unit_type: str
    name: str | None
    source: str
    source_hash: str
    language: str
    parser_key: str


@dataclass(frozen=True)
class SourceFile:
    """UTF-8 source file contents plus stable path and hash metadata."""

    absolute_path: Path
    relative_path: str
    contents: str
    hash: str


@dataclass(frozen=True)
class StoredEmbeddingCandidate:
    """Embedding row loaded from SQLite for exact scoring."""

    unit_id: str
    vector: list[float]
    norm: float


@dataclass(frozen=True)
class StoredCodeUnitEmbedding:
    """Code unit and embedding vector loaded together for duplicate detection."""

    unit: CodeUnit
    vector: list[float]
    norm: float


@dataclass(frozen=True)
class LexicalMatch:
    """FTS match and raw SQLite rank for one code unit."""

    unit: CodeUnit
    rank: float


@dataclass(frozen=True)
class FileError:
    """Recent indexing error associated with a source path."""

    path: str
    error: str


@dataclass(frozen=True)
class StoreStats:
    """Aggregate SQLite store counts and recent indexing errors."""

    indexed_files: int
    indexed_units: int
    embedded_units: int
    lexical_units: int
    recent_errors: list[FileError]


class Priority(Enum):
    """Queue priority used by manual and background indexing work."""

    HIGH = 100
    LOW = 0

    @classmethod
    def from_int(cls, value: int) -> "Priority":
        """Convert a stored integer priority into the nearest enum value."""

        return cls.HIGH if value >= cls.HIGH.value else cls.LOW


@dataclass(frozen=True)
class QueueStats:
    """Counts of pending and in-progress indexing work."""

    high_priority: int
    low_priority: int
    in_progress: int


@dataclass(frozen=True)
class QueuedWork:
    """Claimed indexing or deletion request from the work queue."""

    id: int
    path: Path
    priority: Priority
    delete: bool


@dataclass(frozen=True)
class LeaseStatus:
    """Current process lease owner and expiration details."""

    owner: str | None
    expires_at_unix: int | None
    held_by_this_process: bool


@dataclass(frozen=True)
class ServiceStatus:
    """Status payload returned by the CLI and MCP status tool."""

    root: str
    repository_root: str | None
    runtime_root: str
    scope_prefix: str | None
    auto_indexing_enabled: bool
    model: str
    queue: QueueStats
    store: StoreStats
    lease: LeaseStatus


class QueryMode(str, Enum):
    """Supported query ranking modes."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class QueryMatchKind(str, Enum):
    """Origin of a returned match score."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class QueryMatch:
    """Ranked code unit returned by a query."""

    score: float
    semantic_score: float | None
    lexical_score: float | None
    match_kind: QueryMatchKind
    unit: CodeUnit


@dataclass(frozen=True)
class QueryResponse:
    """Response wrapper for query results."""

    matches: list[QueryMatch]


@dataclass(frozen=True)
class IndexResponse:
    """Response returned after an index request is queued."""

    queued: bool
    path: str


@dataclass(frozen=True)
class IndexedItem:
    """Listed code unit metadata, optionally including source text."""

    file_path: str
    language: str
    unit_type: str
    name: str | None
    start_line: int
    end_line: int
    source: str | None


@dataclass(frozen=True)
class IndexedListResponse:
    """Paginated list of indexed code units."""

    items: list[IndexedItem]
    limit: int
    offset: int
