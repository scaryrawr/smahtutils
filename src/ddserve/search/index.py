from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

from ddserve.config import DdserveConfig
from ddserve.errors import DdserveError
from ddserve.models import to_jsonable

from ddserve.embeddings.openai import EmbeddingClient, create_openai_embedding_client
from ddserve.embeddings.annoy_index import AnnoyIndexManager
from ddserve.embeddings.storage import SearchChunk, cosine_similarity, open_embedding_storage

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
    config: DdserveConfig,
    slugs: list[str] | None = None,
    languages: list[str] | None = None,
    limit: int = 10,
    env: dict[str, str] | None = None,
    client: EmbeddingClient | None = None,
) -> list[SearchResult]:
    """Implement search docs."""
    if not query.strip():
        raise DdserveError("Search query must not be empty")
    bounded_limit = max(1, min(limit, 50))
    resolved_slugs = resolve_docset_filters(cache_root, slugs, languages)
    if resolved_slugs is not None and not resolved_slugs:
        return []
    storage = open_embedding_storage(cache_root)
    try:
        semantic: list[SearchResult] = []
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
        if semantic:
            return semantic
        terms = parse_keyword_terms(query)
        return [
            chunk_to_result(chunk, max(0.0, 1.0 - index / max(1, bounded_limit)), "keyword")
            for index, chunk in enumerate(
                storage.keyword_chunks(terms, resolved_slugs, bounded_limit)
            )
        ]
    finally:
        storage.close()


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
