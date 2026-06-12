from __future__ import annotations

from dataclasses import dataclass

from .annoy_index import AnnoyIndexManager
from .context import RuntimeContext
from .embedding import OpenAiEmbedder
from .indexer import Indexer
from .models import (
    CodeUnit,
    IndexResponse,
    IndexedListResponse,
    LexicalMatch,
    QueryMatch,
    QueryMatchKind,
    QueryMode,
    QueryResponse,
    ServiceStatus,
)
from .store import Store
from .vector import cosine_similarity_with_norms, vector_norm


@dataclass(frozen=True)
class AppState:
    store: Store
    indexer: Indexer
    embedder: OpenAiEmbedder
    context: RuntimeContext
    annoy: AnnoyIndexManager


async def status(state: AppState) -> ServiceStatus:
    state.store.ensure_lexical_index_current()
    return ServiceStatus(
        root=str(state.indexer.root()),
        repository_root=str(state.context.repository_root)
        if state.context.repository_root
        else None,
        runtime_root=str(state.context.runtime_root),
        scope_prefix=state.context.scope_prefix,
        auto_indexing_enabled=state.context.auto_indexing_enabled,
        model=state.indexer.embedder.model,
        queue=state.indexer.queue_stats(),
        store=state.store.stats(),
        lease=state.indexer.lease_status(),
    )


async def index_path(state: AppState, path: str) -> IndexResponse:
    await state.indexer.enqueue_requested_path_under(path, state.context.runtime_root)
    return IndexResponse(True, path)


async def query_code(
    state: AppState,
    query: str,
    limit: int | None = None,
    mode: QueryMode | str | None = None,
    path_prefix: str | None = None,
    language: str | None = None,
) -> QueryResponse:
    actual_limit = max(1, min(limit or 10, 100))
    actual_mode = QueryMode(mode or QueryMode.HYBRID)
    scoped_prefix = state.context.scoped_path_prefix(path_prefix)
    fts_query = build_fts_query(query)
    lexical_limit = max(50, min(actual_limit * 20, 500))

    lexical_matches: list[LexicalMatch] = []
    if actual_mode in {QueryMode.KEYWORD, QueryMode.HYBRID} and fts_query:
        state.store.ensure_lexical_index_current()
        lexical_matches = state.store.lexical_search(
            fts_query,
            scoped_prefix,
            language,
            lexical_limit,
        )

    semantic_matches: list[tuple[CodeUnit, float]] = []
    if actual_mode in {QueryMode.SEMANTIC, QueryMode.HYBRID}:
        embeddings = await state.embedder.embed_texts([query])
        if not embeddings:
            raise ValueError("embedding response was empty")
        required_ids = (
            {match.unit.id for match in lexical_matches}
            if actual_mode == QueryMode.HYBRID
            else set()
        )
        semantic_matches = semantic_top_matches(
            state,
            embeddings[0],
            scoped_prefix,
            language,
            actual_limit,
            required_ids,
        )

    matches = merge_matches(actual_mode, semantic_matches, lexical_matches)
    matches.sort(key=lambda item: item.score, reverse=True)
    return QueryResponse(matches[:actual_limit])


def semantic_top_matches(
    state: AppState,
    query_embedding: list[float],
    path_prefix: str | None,
    language: str | None,
    limit: int,
    required_unit_ids: set[str],
) -> list[tuple[CodeUnit, float]]:
    query_norm = vector_norm(query_embedding)
    if query_norm == 0.0:
        return []

    overfetch = max(50, min(5000, limit * 50 + len(required_unit_ids)))
    candidate_ids = state.annoy.search(state.embedder.model, query_embedding, overfetch)
    candidate_ids.extend(required_unit_ids)
    candidate_ids = list(dict.fromkeys(candidate_ids))
    candidates = state.store.embedding_candidates_by_ids(state.embedder.model, candidate_ids)
    scores: dict[str, float] = {}
    for candidate in candidates:
        score = cosine_similarity_with_norms(
            query_embedding,
            candidate.vector,
            query_norm,
            candidate.norm,
        )
        if score is not None:
            scores[candidate.unit_id] = score

    units = state.store.code_units_by_ids(scores)
    filtered: list[tuple[CodeUnit, float]] = []
    for unit in units:
        if path_prefix and not (
            unit.file_path == path_prefix or unit.file_path.startswith(f"{path_prefix}/")
        ):
            continue
        if language and unit.language != language:
            continue
        filtered.append((unit, scores[unit.id]))
    filtered.sort(key=lambda item: item[1], reverse=True)
    return filtered[: max(limit, len(required_unit_ids))]


def merge_matches(
    mode: QueryMode,
    semantic_matches: list[tuple[CodeUnit, float]],
    lexical_matches: list[LexicalMatch],
) -> list[QueryMatch]:
    lexical_scores = normalize_lexical_scores(lexical_matches)
    candidates: dict[str, CandidateMatch] = {}
    for unit, score in semantic_matches:
        candidates.setdefault(unit.id, CandidateMatch(unit)).semantic_score = score
    for lexical_match, lexical_score in zip(lexical_matches, lexical_scores, strict=True):
        candidates.setdefault(
            lexical_match.unit.id, CandidateMatch(lexical_match.unit)
        ).lexical_score = lexical_score
    return [candidate.into_query_match(mode) for candidate in candidates.values()]


@dataclass
class CandidateMatch:
    unit: CodeUnit
    semantic_score: float | None = None
    lexical_score: float | None = None

    def into_query_match(self, mode: QueryMode) -> QueryMatch:
        semantic_normalized = (
            max(0.0, min(1.0, (self.semantic_score + 1.0) / 2.0))
            if self.semantic_score is not None
            else None
        )
        match_kind = (
            QueryMatchKind.HYBRID
            if self.semantic_score is not None and self.lexical_score is not None
            else QueryMatchKind.SEMANTIC
            if self.semantic_score is not None
            else QueryMatchKind.KEYWORD
        )
        if mode == QueryMode.SEMANTIC:
            score = self.semantic_score or 0.0
        elif mode == QueryMode.KEYWORD:
            score = self.lexical_score or 0.0
        else:
            if semantic_normalized is not None and self.lexical_score is not None:
                score = (0.7 * semantic_normalized) + (0.3 * self.lexical_score)
            elif semantic_normalized is not None:
                score = semantic_normalized
            elif self.lexical_score is not None:
                score = self.lexical_score * 0.85
            else:
                score = 0.0
        return QueryMatch(score, self.semantic_score, self.lexical_score, match_kind, self.unit)


def normalize_lexical_scores(matches: list[LexicalMatch]) -> list[float]:
    if not matches:
        return []
    ranks = [match.rank for match in matches]
    min_rank = min(ranks)
    max_rank = max(ranks)
    if max_rank - min_rank == 0:
        return [1.0 for _ in matches]
    return [max(0.0, min(1.0, 1.0 - ((rank - min_rank) / (max_rank - min_rank)))) for rank in ranks]


def build_fts_query(query: str) -> str | None:
    tokens: list[str] = []
    for raw in query.replace("_", " ").split():
        token = "".join(ch for ch in raw.lower() if ch.isalnum() or ch == "_")
        if len(token) >= 2 and token not in tokens:
            tokens.append(token)
        if len(tokens) >= 12:
            break
    return " OR ".join(f"{token}*" for token in tokens) if tokens else None


def list_indexed(
    state: AppState,
    path_prefix: str | None = None,
    language: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    include_source: bool | None = None,
) -> IndexedListResponse:
    actual_limit = max(1, min(limit or 50, 200))
    actual_offset = offset or 0
    actual_include_source = bool(include_source) and actual_limit <= 20
    scoped_prefix = state.context.scoped_path_prefix(path_prefix)
    return IndexedListResponse(
        state.store.list_indexed_units(
            scoped_prefix,
            language,
            actual_limit,
            actual_offset,
            actual_include_source,
        ),
        actual_limit,
        actual_offset,
    )
