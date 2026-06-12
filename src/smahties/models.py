from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class CodeUnit:
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
    absolute_path: Path
    relative_path: str
    contents: str
    hash: str


@dataclass(frozen=True)
class StoredEmbeddingCandidate:
    unit_id: str
    vector: list[float]
    norm: float


@dataclass(frozen=True)
class LexicalMatch:
    unit: CodeUnit
    rank: float


@dataclass(frozen=True)
class FileError:
    path: str
    error: str


@dataclass(frozen=True)
class StoreStats:
    indexed_files: int
    indexed_units: int
    embedded_units: int
    lexical_units: int
    recent_errors: list[FileError]


class Priority(Enum):
    HIGH = 100
    LOW = 0

    @classmethod
    def from_int(cls, value: int) -> "Priority":
        return cls.HIGH if value >= cls.HIGH.value else cls.LOW


@dataclass(frozen=True)
class QueueStats:
    high_priority: int
    low_priority: int
    in_progress: int


@dataclass(frozen=True)
class QueuedWork:
    id: int
    path: Path
    priority: Priority
    delete: bool


@dataclass(frozen=True)
class LeaseStatus:
    owner: str | None
    expires_at_unix: int | None
    held_by_this_process: bool


@dataclass(frozen=True)
class ServiceStatus:
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
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class QueryMatchKind(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class QueryMatch:
    score: float
    semantic_score: float | None
    lexical_score: float | None
    match_kind: QueryMatchKind
    unit: CodeUnit


@dataclass(frozen=True)
class QueryResponse:
    matches: list[QueryMatch]


@dataclass(frozen=True)
class IndexResponse:
    queued: bool
    path: str


@dataclass(frozen=True)
class IndexedItem:
    file_path: str
    language: str
    unit_type: str
    name: str | None
    start_line: int
    end_line: int
    source: str | None


@dataclass(frozen=True)
class IndexedListResponse:
    items: list[IndexedItem]
    limit: int
    offset: int
