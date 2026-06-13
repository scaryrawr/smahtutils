from __future__ import annotations

import html
import json
from dataclasses import dataclass, replace
from pathlib import Path

from smahtiepants.config import SmahtiepantsConfig
from smahtiepants.errors import SmahtiepantsError
from smahtiepants.models import to_jsonable

from smahtiepants.embeddings.openai import EmbeddingClient, create_openai_embedding_client
from smahtiepants.embeddings.annoy_index import AnnoyIndexManager
from smahtiepants.embeddings.storage import SearchChunk, cosine_similarity, open_embedding_storage

from .filters import resolve_docset_filters
from .terms import parse_keyword_terms


@dataclass(frozen=True)
class SearchResult:
    """Represent SearchResult."""

    score: float
    match_kind: str
    docset_slug: str
    docset_name: str
    page_id: str
    page_title: str
    page_path: str
    page_type: str | None
    page_file: str
    chunk_ordinal: int
    text: str
    metadata: dict[str, object]


def search_docs(
    cache_root: str | Path,
    query: str,
    config: SmahtiepantsConfig,
    slugs: list[str] | None = None,
    languages: list[str] | None = None,
    limit: int = 10,
    env: dict[str, str] | None = None,
    client: EmbeddingClient | None = None,
) -> list[SearchResult]:
    """Implement search docs."""
    if not query.strip():
        raise SmahtiepantsError("Search query must not be empty")
    bounded_limit = max(1, min(limit, 50))
    resolved_slugs = resolve_docset_filters(cache_root, slugs, languages)
    if resolved_slugs is not None and not resolved_slugs:
        return []
    storage = open_embedding_storage(cache_root)
    try:
        semantic: list[SearchResult] = []
        terms = parse_keyword_terms(query)
        if config.openai is not None and config.embeddings.enabled:
            embedding_client = client or create_openai_embedding_client(config, env)
            query_vector = embedding_client.create_embeddings(query)[0]
            candidate_count = max(bounded_limit * 25, 100)
            manager = AnnoyIndexManager(cache_root, storage)
            candidate_ids = manager.search(
                config.openai.embedding_model, query_vector, candidate_count
            )
            chunks = storage.chunks_by_ids(
                candidate_ids, config.openai.embedding_model, resolved_slugs
            )
            if len(chunks) < bounded_limit and resolved_slugs:
                scoped_fill = storage.chunks_with_vectors(
                    config.openai.embedding_model, resolved_slugs, candidate_count
                )
                known_ids = {chunk.id for chunk in chunks}
                chunks.extend(chunk for chunk in scoped_fill if chunk.id not in known_ids)
            semantic = sorted(
                (
                    chunk_to_result(
                        chunk, cosine_similarity(query_vector, chunk.vector or []), "semantic"
                    )
                    for chunk in chunks
                ),
                key=lambda item: item.score,
                reverse=True,
            )[:bounded_limit]
        keyword_limit = max(bounded_limit * 5, 50) if semantic else bounded_limit
        keyword = [
            chunk_to_result(chunk, max(0.0, 1.0 - index / max(1, keyword_limit)), "keyword")
            for index, chunk in enumerate(
                storage.keyword_chunks(terms, resolved_slugs, keyword_limit)
            )
        ]
        if semantic and keyword:
            return merge_search_results(semantic, keyword, bounded_limit)
        if semantic:
            return semantic
        return keyword[:bounded_limit]
    finally:
        storage.close()


def merge_search_results(
    semantic: list[SearchResult], keyword: list[SearchResult], limit: int
) -> list[SearchResult]:
    """Merge semantic and keyword results, preserving one result per chunk."""

    merged: dict[tuple[str, str, int], SearchResult] = {}
    for result in semantic:
        merged[result_key(result)] = result
    for result in keyword:
        key = result_key(result)
        existing = merged.get(key)
        if existing is None:
            merged[key] = result
        else:
            merged[key] = replace(
                existing,
                score=max(existing.score, result.score),
                match_kind="hybrid",
            )
    return sorted(merged.values(), key=result_sort_key, reverse=True)[:limit]


def result_sort_key(result: SearchResult) -> tuple[float, int]:
    """Return ranking key that favors lexical matches on score ties."""

    match_priority = {"semantic": 0, "keyword": 1, "hybrid": 2}.get(result.match_kind, 0)
    return (result.score, match_priority)


def result_key(result: SearchResult) -> tuple[str, str, int]:
    """Return a stable chunk identity for result de-duplication."""

    return (result.docset_slug, result.page_id, result.chunk_ordinal)


def chunk_to_result(chunk: SearchChunk, score: float, match_kind: str) -> SearchResult:
    """Implement chunk to result."""
    try:
        metadata = json.loads(chunk.metadata_json)
    except json.JSONDecodeError:
        metadata = {}
    return SearchResult(
        score=score,
        match_kind=match_kind,
        docset_slug=chunk.docset_slug,
        docset_name=chunk.docset_name,
        page_id=chunk.page_id,
        page_title=chunk.page_title,
        page_path=chunk.page_path,
        page_type=chunk.page_type,
        page_file=chunk.page_file,
        chunk_ordinal=chunk.ordinal,
        text=chunk.text,
        metadata=metadata,
    )


def results_to_json(results: list[SearchResult]) -> str:
    """Implement results to json."""
    return json.dumps({"matches": [to_jsonable(result) for result in results]}, indent=2)


def results_to_text(results: list[SearchResult]) -> str:
    """Implement results to text."""
    if not results:
        return "No matches."
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. {result.docset_slug}:{result.page_path} {result.page_title} "
            f"[{result.match_kind} score {result.score:.3f}]"
        )
        for line in result.text.splitlines()[:12]:
            lines.append(f"    {line}")
    return "\n".join(lines)


def results_to_xml(results: list[SearchResult]) -> str:
    """Implement results to xml."""
    items = []
    for result in results:
        items.append(
            "<result>"
            f"<score>{result.score:.6f}</score>"
            f"<matchKind>{html.escape(result.match_kind)}</matchKind>"
            f"<docsetSlug>{html.escape(result.docset_slug)}</docsetSlug>"
            f"<pageId>{html.escape(result.page_id)}</pageId>"
            f"<pageTitle>{html.escape(result.page_title)}</pageTitle>"
            f"<pagePath>{html.escape(result.page_path)}</pagePath>"
            f"<text>{html.escape(result.text)}</text>"
            "</result>"
        )
    return f"<results>{''.join(items)}</results>"
