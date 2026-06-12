from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from .annoy_index import AnnoyIndexManager
from .context import RuntimeContext
from .embedding import OpenAiEmbedder
from .indexer import Indexer
from .models import CodeUnit, SourceFile
from .parser import ParserRegistry, build_unit, spec_for_path
from .scanner import Scanner
from .store import Store
from .vector import cosine_similarity_with_norms, vector_norm

DEFAULT_DUPLICATE_THRESHOLD = 0.92
DEFAULT_COMPARISON_LEVELS = ("function",)
EXHAUSTIVE_SEARCH_LIMIT = 1024
MAX_CANDIDATES_PER_UNIT = 64
SIGNATURE_DIMENSIONS = 12
SIGNATURE_BAND_SIZE = 2
VECTOR_FINGERPRINT_PRECISION = 1000

FUNCTION_UNIT_TYPES = {
    "arrow_function",
    "compact_constructor_declaration",
    "constructor_declaration",
    "function",
    "function_declaration",
    "function_definition",
    "function_expression",
    "function_item",
    "generator_function_declaration",
    "method",
    "method_declaration",
    "method_definition",
    "rule_set",
    "singleton_method",
}
CLASS_UNIT_TYPES = {
    "class",
    "class_declaration",
    "class_definition",
    "class_specifier",
    "enum_declaration",
    "enum_item",
    "enum_specifier",
    "impl_item",
    "interface_declaration",
    "module",
    "mod_item",
    "namespace_definition",
    "record_declaration",
    "struct_declaration",
    "struct_item",
    "struct_specifier",
    "trait_item",
    "type_alias_declaration",
    "type_declaration",
}


class ComparisonLevel(str, Enum):
    """Granularity available for duplicate-code comparisons."""

    FUNCTION = "function"
    CLASS = "class"
    FILE = "file"


@dataclass(frozen=True)
class DuplicateUnit:
    """Codigami-compatible report projection of a code unit."""

    id: str
    filePath: str
    startLine: int
    endLine: int
    unitType: str
    name: str | None
    source: str
    language: str


@dataclass(frozen=True)
class DuplicatePair:
    """A pair of duplicate units with cosine similarity."""

    unitA: CodeUnit
    unitB: CodeUnit
    similarity: float


@dataclass(frozen=True)
class DuplicateReportPair:
    """Codigami-compatible pair reference within a cluster."""

    unitIdA: str
    unitIdB: str
    similarity: float


@dataclass(frozen=True)
class DuplicateCluster:
    """Codigami-compatible duplicate cluster."""

    units: list[DuplicateUnit]
    pairs: list[DuplicateReportPair]


@dataclass(frozen=True)
class DuplicateReport:
    """Codigami-compatible duplicate-code report."""

    scannedFiles: int
    totalUnits: int
    threshold: float
    timestamp: str
    duplicateClusters: list[DuplicateCluster]


@dataclass(frozen=True)
class DuplicateEntry:
    """Candidate unit plus embedding vector and precomputed norm."""

    unit: CodeUnit
    vector: list[float]
    norm: float


@dataclass(frozen=True)
class DuplicateSearchOptions:
    """Tuning knobs for bounded duplicate search."""

    exhaustive_search_limit: int = EXHAUSTIVE_SEARCH_LIMIT
    max_candidates_per_unit: int = MAX_CANDIDATES_PER_UNIT
    signature_dimensions: int = SIGNATURE_DIMENSIONS
    signature_band_size: int = SIGNATURE_BAND_SIZE


class DuplicateState(Protocol):
    """State dependencies required by duplicate detection."""

    store: Store
    indexer: Indexer
    embedder: OpenAiEmbedder
    context: RuntimeContext
    annoy: AnnoyIndexManager


async def duplicate_code(
    state: DuplicateState,
    requested_paths: Sequence[str],
    threshold: float,
    levels: Sequence[ComparisonLevel],
    language: str | None = None,
) -> DuplicateReport:
    """Find duplicate code under requested paths using the existing smahties index."""

    scanner: Scanner = state.indexer.scanner
    paths = resolve_requested_paths(scanner, state.context.runtime_root, requested_paths)
    discovered_files = unique_paths(path for item in paths for path in scanner.discover_files(item))
    prefixes = [prefix_for_path(scanner, path) for path in paths]
    if any(prefix is None for prefix in prefixes):
        prefixes = [None]

    persisted_entries = [
        DuplicateEntry(candidate.unit, candidate.vector, candidate.norm)
        for candidate in state.store.code_unit_embeddings_for_model(
            state.embedder.model, prefixes, language
        )
        if stored_unit_matches_levels(candidate.unit, levels)
    ]

    transient_entries: list[DuplicateEntry] = []
    if ComparisonLevel.FILE in levels:
        transient_entries = await transient_file_entries(
            scanner,
            ParserRegistry(),
            state.embedder,
            discovered_files,
            language,
        )

    pairs = find_duplicate_pairs_prefer_annoy(
        persisted_entries,
        transient_entries,
        threshold,
        state.annoy,
        state.embedder.model,
    )
    return DuplicateReport(
        scannedFiles=len(discovered_files),
        totalUnits=len(persisted_entries) + len(transient_entries),
        threshold=threshold,
        timestamp=utc_timestamp(),
        duplicateClusters=cluster_duplicates(pairs),
    )


def parse_comparison_levels(raw_values: Sequence[str] | None) -> tuple[ComparisonLevel, ...]:
    """Parse repeated and comma-separated comparison-level arguments."""

    values = raw_values or DEFAULT_COMPARISON_LEVELS
    levels: list[ComparisonLevel] = []
    for raw in values:
        for part in raw.split(","):
            value = part.strip()
            if not value:
                continue
            try:
                level = ComparisonLevel(value)
            except ValueError as exc:
                allowed = ", ".join(item.value for item in ComparisonLevel)
                raise ValueError(f"level must include one or more of: {allowed}") from exc
            if level not in levels:
                levels.append(level)
    if not levels:
        raise ValueError("level must include at least one comparison level")
    return tuple(levels)


def parse_threshold(raw: str) -> float:
    """Parse a duplicate similarity threshold from CLI text."""

    try:
        threshold = float(raw)
    except ValueError as exc:
        raise ValueError("threshold must be a number between 0.0 and 1.0") from exc
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be a number between 0.0 and 1.0")
    return threshold


def stored_unit_matches_levels(unit: CodeUnit, levels: Sequence[ComparisonLevel]) -> bool:
    """Return whether a persisted unit should be considered for selected levels."""

    level_set = set(levels)
    if ComparisonLevel.FUNCTION in level_set and unit.unit_type in FUNCTION_UNIT_TYPES:
        return True
    if ComparisonLevel.CLASS in level_set and unit.unit_type in CLASS_UNIT_TYPES:
        return True
    return False


def find_duplicate_pairs(
    entries: Sequence[DuplicateEntry],
    threshold: float,
    options: DuplicateSearchOptions | None = None,
) -> list[DuplicatePair]:
    """Find duplicate unit pairs at or above the cosine similarity threshold."""

    if len(entries) <= 1:
        return []
    options = normalize_search_options(options or DuplicateSearchOptions())
    if len(entries) <= options.exhaustive_search_limit:
        return find_duplicate_pairs_exhaustive(entries, threshold)
    return find_duplicate_pairs_bounded(entries, threshold, options)


def find_duplicate_pairs_prefer_annoy(
    persisted_entries: Sequence[DuplicateEntry],
    transient_entries: Sequence[DuplicateEntry],
    threshold: float,
    annoy: AnnoyIndexManager,
    model: str,
    options: DuplicateSearchOptions | None = None,
) -> list[DuplicatePair]:
    """Find duplicate pairs, using Annoy for large persisted embedding sets."""

    options = normalize_search_options(options or DuplicateSearchOptions())
    if len(persisted_entries) <= options.exhaustive_search_limit:
        pairs = find_duplicate_pairs_exhaustive(persisted_entries, threshold)
    else:
        pairs = find_duplicate_pairs_annoy(persisted_entries, threshold, annoy, model, options)

    if transient_entries:
        pairs.extend(
            find_duplicate_pairs_involving_transient(
                persisted_entries,
                transient_entries,
                threshold,
            )
        )
    return pairs


def find_duplicate_pairs_annoy(
    entries: Sequence[DuplicateEntry],
    threshold: float,
    annoy: AnnoyIndexManager,
    model: str,
    options: DuplicateSearchOptions | None = None,
) -> list[DuplicatePair]:
    """Use Annoy nearest-neighbor candidates, then exact-score each candidate pair."""

    if len(entries) <= 1:
        return []
    options = normalize_search_options(options or DuplicateSearchOptions())
    entries_by_id = {entry.unit.id: entry for entry in entries}
    pairs: list[DuplicatePair] = []
    seen: set[tuple[str, str]] = set()
    total_annoy_items = annoy.item_count(model)
    if total_annoy_items == 0:
        return find_duplicate_pairs_bounded(entries, threshold, options)
    candidate_limit = (
        min(total_annoy_items, options.max_candidates_per_unit + 1)
        if total_annoy_items == len(entries)
        else total_annoy_items
    )
    for entry in entries:
        scoped_candidates = 0
        for candidate_id in annoy.search(model, entry.vector, candidate_limit):
            candidate = entries_by_id.get(candidate_id)
            if candidate is None or candidate.unit.id == entry.unit.id:
                continue
            if scoped_candidates >= options.max_candidates_per_unit:
                break
            scoped_candidates += 1
            key = pair_key(entry.unit.id, candidate.unit.id)
            if key in seen:
                continue
            seen.add(key)
            score = cosine_similarity_with_norms(
                entry.vector,
                candidate.vector,
                entry.norm,
                candidate.norm,
            )
            if score is not None and score >= threshold:
                pairs.append(DuplicatePair(entry.unit, candidate.unit, score))
    return pairs


def find_duplicate_pairs_involving_transient(
    persisted_entries: Sequence[DuplicateEntry],
    transient_entries: Sequence[DuplicateEntry],
    threshold: float,
) -> list[DuplicatePair]:
    """Exact-score pairs where at least one side is a transient whole-file entry."""

    pairs: list[DuplicatePair] = []
    all_entries = [*persisted_entries, *transient_entries]
    transient_ids = {entry.unit.id for entry in transient_entries}
    seen: set[tuple[str, str]] = set()
    for left in transient_entries:
        for right in all_entries:
            if left.unit.id == right.unit.id:
                continue
            if left.unit.file_path == right.unit.file_path and (
                left.unit.unit_type == "file" or right.unit.unit_type == "file"
            ):
                continue
            key = pair_key(left.unit.id, right.unit.id)
            if key in seen:
                continue
            if right.unit.id in transient_ids and right.unit.id < left.unit.id:
                continue
            seen.add(key)
            score = cosine_similarity_with_norms(left.vector, right.vector, left.norm, right.norm)
            if score is not None and score >= threshold:
                pairs.append(DuplicatePair(left.unit, right.unit, score))
    return pairs


def pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    """Return an order-independent pair key."""

    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)


def normalize_search_options(options: DuplicateSearchOptions) -> DuplicateSearchOptions:
    """Clamp invalid search options to safe defaults."""

    return DuplicateSearchOptions(
        exhaustive_search_limit=options.exhaustive_search_limit
        if options.exhaustive_search_limit >= 0
        else EXHAUSTIVE_SEARCH_LIMIT,
        max_candidates_per_unit=options.max_candidates_per_unit
        if options.max_candidates_per_unit >= 1
        else MAX_CANDIDATES_PER_UNIT,
        signature_dimensions=options.signature_dimensions
        if options.signature_dimensions >= 1
        else SIGNATURE_DIMENSIONS,
        signature_band_size=options.signature_band_size
        if options.signature_band_size >= 1
        else SIGNATURE_BAND_SIZE,
    )


def find_duplicate_pairs_exhaustive(
    entries: Sequence[DuplicateEntry], threshold: float
) -> list[DuplicatePair]:
    """Compare every entry pair exactly."""

    pairs: list[DuplicatePair] = []
    for left_index, left in enumerate(entries):
        for right in entries[left_index + 1 :]:
            score = cosine_similarity_with_norms(left.vector, right.vector, left.norm, right.norm)
            if score is not None and score >= threshold:
                pairs.append(DuplicatePair(left.unit, right.unit, score))
    return pairs


def find_duplicate_pairs_bounded(
    entries: Sequence[DuplicateEntry],
    threshold: float,
    options: DuplicateSearchOptions,
) -> list[DuplicatePair]:
    """Use deterministic signature buckets to bound candidate comparisons."""

    pairs: list[DuplicatePair] = []
    buckets: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        signatures = create_signature_keys(entry.vector, options)
        candidates = select_candidates(signatures, buckets, options.max_candidates_per_unit)
        for candidate_index in candidates:
            candidate = entries[candidate_index]
            score = cosine_similarity_with_norms(
                candidate.vector, entry.vector, candidate.norm, entry.norm
            )
            if score is not None and score >= threshold:
                pairs.append(DuplicatePair(candidate.unit, entry.unit, score))
        for signature in signatures:
            buckets.setdefault(signature, []).append(index)
    return pairs


def create_signature_keys(vector: Sequence[float], options: DuplicateSearchOptions) -> list[str]:
    """Create deterministic approximate-match signatures for an embedding vector."""

    top_dimensions = sorted(
        (
            (index, "-" if value < 0 else "+", abs(value))
            for index, value in enumerate(vector)
            if abs(value) > 0.0
        ),
        key=lambda item: (-item[2], item[0], item[1]),
    )[: options.signature_dimensions]
    if not top_dimensions:
        return ["zero-vector"]

    signed_dimensions = [f"{sign}{index}" for index, sign, _magnitude in top_dimensions]
    magnitude = vector_norm(list(vector))
    keys: list[str] = []

    def add_key(key: str) -> None:
        if key not in keys:
            keys.append(key)

    if magnitude > 0.0:
        fingerprint = "|".join(
            f"{index}:{round((vector[index] / magnitude) * VECTOR_FINGERPRINT_PRECISION)}"
            for index, _sign, _magnitude in top_dimensions
        )
        add_key(f"fingerprint:{fingerprint}")
    add_key(f"top:{'|'.join(signed_dimensions)}")
    for index in range(0, len(signed_dimensions), options.signature_band_size):
        add_key(f"band:{'|'.join(signed_dimensions[index : index + options.signature_band_size])}")
    for signed_dimension in signed_dimensions:
        add_key(f"dim:{signed_dimension}")
    return keys


def select_candidates(
    signatures: Sequence[str], buckets: dict[str, list[int]], max_candidates: int
) -> list[int]:
    """Select recent and early entries from matching signature buckets."""

    candidates: list[int] = []
    seen: set[int] = set()
    bucket_scan_limit = max(1, math.ceil(max_candidates / max(1, len(signatures) * 2)))

    def add(candidate: int) -> None:
        if candidate not in seen and len(candidates) < max_candidates:
            seen.add(candidate)
            candidates.append(candidate)

    for signature in signatures:
        bucket = buckets.get(signature)
        if not bucket:
            continue
        scanned = 0
        for candidate in reversed(bucket):
            if scanned >= bucket_scan_limit or len(candidates) >= max_candidates:
                break
            add(candidate)
            scanned += 1

    for signature in signatures:
        bucket = buckets.get(signature)
        if not bucket:
            continue
        scanned = 0
        for candidate in bucket:
            if scanned >= bucket_scan_limit or len(candidates) >= max_candidates:
                break
            add(candidate)
            scanned += 1

    return candidates


def cluster_duplicates(pairs: Sequence[DuplicatePair]) -> list[DuplicateCluster]:
    """Group connected duplicate pairs into clusters."""

    if not pairs:
        return []

    parent: dict[str, str] = {}

    def find(id_: str) -> str:
        root = id_
        while parent[root] != root:
            root = parent[root]
        current = id_
        while current != root:
            next_id = parent[current]
            parent[current] = root
            current = next_id
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for pair in pairs:
        for unit in (pair.unitA, pair.unitB):
            parent.setdefault(unit.id, unit.id)
        union(pair.unitA.id, pair.unitB.id)

    clusters: dict[str, tuple[dict[str, CodeUnit], list[DuplicateReportPair]]] = {}
    for pair in pairs:
        root = find(pair.unitA.id)
        units, cluster_pairs = clusters.setdefault(root, ({}, []))
        units[pair.unitA.id] = pair.unitA
        units[pair.unitB.id] = pair.unitB
        cluster_pairs.append(DuplicateReportPair(pair.unitA.id, pair.unitB.id, pair.similarity))

    return [
        DuplicateCluster([to_duplicate_unit(unit) for unit in units.values()], cluster_pairs)
        for units, cluster_pairs in clusters.values()
    ]


async def transient_file_entries(
    scanner: Scanner,
    parser: ParserRegistry,
    embedder: OpenAiEmbedder,
    files: Sequence[Path],
    language: str | None,
) -> list[DuplicateEntry]:
    """Create and embed whole-file units without persisting them in the search index."""

    units: list[CodeUnit] = []
    for path in files:
        source_file = scanner.read_source(path)
        if source_file is None:
            continue
        unit = create_file_unit(source_file, parser)
        if language is None or unit.language == language:
            units.append(unit)
    if not units:
        return []
    vectors = await embedder.embed_texts([unit.source for unit in units])
    if len(vectors) != len(units):
        raise ValueError(
            f"embedding response count {len(vectors)} did not match file unit count {len(units)}"
        )
    return [
        DuplicateEntry(unit, vector, vector_norm(vector))
        for unit, vector in zip(units, vectors, strict=True)
    ]


def create_file_unit(source_file: SourceFile, parser: ParserRegistry) -> CodeUnit:
    """Create a whole-file comparison unit for duplicate detection."""

    language = language_for_path(source_file.absolute_path)
    end_line = max(1, len(source_file.contents.splitlines()))
    return build_unit(
        source_file,
        source_file.contents,
        1,
        end_line,
        0,
        len(source_file.contents.encode("utf-8")),
        "file",
        source_file.relative_path,
        language,
        f"{parser.cache_key_for_path(source_file.absolute_path)}|file:v1",
    )


def language_for_path(path: Path) -> str:
    """Return the parser language for a path, or text for fallback files."""

    if path.suffix == ".py":
        return "python"
    spec = spec_for_path(path)
    return spec.language_name if spec else "text"


def resolve_requested_paths(
    scanner: Scanner, runtime_root: Path, requested_paths: Sequence[str]
) -> list[Path]:
    """Resolve user paths under the active runtime scope."""

    raw_paths = requested_paths or ["."]
    return unique_paths(scanner.resolve_existing_under(runtime_root, item) for item in raw_paths)


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Return unique resolved paths in input order."""

    unique: dict[Path, None] = {}
    for path in paths:
        unique[path.resolve()] = None
    return list(unique)


def prefix_for_path(scanner: Scanner, path: Path) -> str | None:
    """Return a store path prefix for a resolved path."""

    prefix = scanner.relative_path(path)
    return None if prefix in {"", "."} else prefix


def to_duplicate_unit(unit: CodeUnit) -> DuplicateUnit:
    """Convert an indexed code unit to the Codigami report shape."""

    return DuplicateUnit(
        id=unit.id,
        filePath=unit.file_path,
        startLine=unit.start_line,
        endLine=unit.end_line,
        unitType=unit.unit_type,
        name=unit.name,
        source=unit.source,
        language=unit.language,
    )


def utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp with a JavaScript-style Z suffix."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
