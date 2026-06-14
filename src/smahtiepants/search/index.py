from __future__ import annotations

import html
import json
import re
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

SEMANTIC_CANDIDATE_COUNT = 1000
SEMANTIC_RESULT_POOL = 250
KEYWORD_RESULT_POOL = 250


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
    excerpt: str
    resource_uri: str
    read_hint: str
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
            manager = AnnoyIndexManager(cache_root, storage)
            candidate_ids = manager.search(
                config.openai.embedding_model, query_vector, SEMANTIC_CANDIDATE_COUNT
            )
            chunks = storage.chunks_by_ids(
                candidate_ids, config.openai.embedding_model, resolved_slugs
            )
            if len(chunks) < SEMANTIC_RESULT_POOL and resolved_slugs:
                scoped_fill = storage.chunks_with_vectors(
                    config.openai.embedding_model,
                    resolved_slugs,
                    SEMANTIC_CANDIDATE_COUNT,
                )
                known_ids = {chunk.id for chunk in chunks}
                chunks.extend(chunk for chunk in scoped_fill if chunk.id not in known_ids)
            semantic = diversify_search_results(
                sorted(
                    (
                        chunk_to_result(
                            chunk,
                            cosine_similarity(query_vector, chunk.vector or []),
                            "semantic",
                            terms,
                        )
                        for chunk in chunks
                    ),
                    key=lambda item: item.score,
                    reverse=True,
                ),
                SEMANTIC_RESULT_POOL,
            )
        keyword_limit = KEYWORD_RESULT_POOL
        keyword = [
            chunk_to_result(
                chunk,
                max(0.0, 1.0 - index / max(1, keyword_limit)),
                "keyword",
                terms,
            )
            for index, chunk in enumerate(
                storage.keyword_chunks(terms, resolved_slugs, keyword_limit)
            )
        ]
        if semantic and keyword:
            return merge_search_results(semantic, keyword, bounded_limit)
        if semantic:
            return diversify_search_results(semantic, bounded_limit)
        return diversify_search_results(keyword, bounded_limit)
    finally:
        storage.close()


def merge_search_results(
    semantic: list[SearchResult], keyword: list[SearchResult], limit: int
) -> list[SearchResult]:
    """Merge keyword metadata into semantic ranking, preserving one result per chunk."""

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
                match_kind="hybrid",
            )
    return diversify_search_results(list(merged.values()), limit)


def result_key(result: SearchResult) -> tuple[str, str, int]:
    """Return a stable chunk identity for result de-duplication."""

    return (result.docset_slug, result.page_id, result.chunk_ordinal)


def page_key(result: SearchResult) -> tuple[str, str]:
    """Return the page identity used to diversify visible results."""

    return (result.docset_slug, result.page_id)


def diversify_search_results(results: list[SearchResult], limit: int) -> list[SearchResult]:
    """Prefer one strong result per page before returning repeated chunks."""

    selected: list[SearchResult] = []
    selected_keys: set[tuple[str, str, int]] = set()
    seen_pages: set[tuple[str, str]] = set()
    for result in results:
        key = result_key(result)
        page = page_key(result)
        if page in seen_pages:
            continue
        selected.append(result)
        selected_keys.add(key)
        seen_pages.add(page)
        if len(selected) >= limit:
            return selected
    for result in results:
        key = result_key(result)
        if key in selected_keys:
            continue
        selected.append(result)
        selected_keys.add(key)
        if len(selected) >= limit:
            break
    return selected


def chunk_to_result(
    chunk: SearchChunk, score: float, match_kind: str, terms: list[str]
) -> SearchResult:
    """Implement chunk to result."""
    try:
        metadata = json.loads(chunk.metadata_json)
    except json.JSONDecodeError:
        metadata = {}
    excerpt = build_result_excerpt(chunk.text, terms)
    resource_uri = f"smahtiepants://docsets/{chunk.docset_slug}/pages/{chunk.page_id}"
    read_hint = (
        f'Read full page with get_page_content slug="{chunk.docset_slug}" '
        f'pageId="{chunk.page_id}" or resource {resource_uri}'
    )
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
        excerpt=excerpt,
        resource_uri=resource_uri,
        read_hint=read_hint,
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
        lines.append(f"    Read full page: {result.read_hint}")
        lines.append("    Excerpt:")
        for line in result.excerpt.splitlines():
            lines.append(f"    {line}")
        if index < len(results):
            lines.append("")
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
            f"<resourceUri>{html.escape(result.resource_uri)}</resourceUri>"
            f"<readHint>{html.escape(result.read_hint)}</readHint>"
            f"<excerpt>{html.escape(result.excerpt)}</excerpt>"
            f"<text>{html.escape(result.text)}</text>"
            "</result>"
        )
    return f"<results>{''.join(items)}</results>"


def build_result_excerpt(
    text: str, terms: list[str], max_lines: int = 12, max_chars: int = 1200
) -> str:
    """Build a compact query-aware excerpt from a stored search chunk."""

    lines = strip_chunk_metadata(text).splitlines()
    if not any(line.strip() for line in lines):
        lines = text.splitlines()
    if not lines:
        return ""
    target = best_matching_line(lines, terms)
    start, end = excerpt_bounds(lines, target, max_lines, max_chars)
    if start >= 2 and not lines[start - 1].strip() and is_context_label_line(lines[start - 2]):
        start -= 2
    excerpt_lines = lines[start:end]
    if start > 0:
        excerpt_lines.insert(0, "...")
    if end < len(lines):
        excerpt_lines.append("...")
    excerpt = "\n".join(excerpt_lines)
    if len(excerpt) > max_chars:
        return trim_excerpt(excerpt, terms, max_chars)
    return excerpt


def strip_chunk_metadata(text: str) -> str:
    """Remove generated chunk metadata before choosing an excerpt."""

    lines = text.splitlines()
    metadata_prefixes = ("Docset:", "Page:", "Path:", "Type:")
    cursor = 0
    while cursor < len(lines) and (
        not lines[cursor].strip() or lines[cursor].startswith(metadata_prefixes)
    ):
        cursor += 1
    return "\n".join(lines[cursor:]).strip()


def best_matching_line(lines: list[str], terms: list[str]) -> int:
    """Return the best line index for query terms, or the first content line."""

    best_index = next((index for index, line in enumerate(lines) if line.strip()), 0)
    best_score = 0
    for index, line in enumerate(lines):
        score = line_term_score(line, terms)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def line_term_score(line: str, terms: list[str]) -> int:
    """Score a line by distinct sanitized query term overlap."""

    if not terms:
        return 0
    tokens = re.findall(r"[a-z0-9_]+", line.lower())
    score = 0
    for term in terms:
        variants = term_variants(term)
        best = 0
        for token in tokens:
            for variant in variants:
                if token.startswith(variant):
                    best = max(best, len(variant))
                elif variant in token:
                    best = max(best, max(1, len(variant) // 2))
        score += best
    return score


def term_variants(term: str) -> tuple[str, ...]:
    """Return simple lexical variants for matching query terms in excerpts."""

    variants = [term]
    if len(term) > 3 and term.endswith("ies"):
        variants.append(f"{term[:-3]}y")
    if len(term) > 3 and term.endswith("s"):
        variants.append(term[:-1])
    return tuple(dict.fromkeys(variants))


def trim_excerpt(excerpt: str, terms: list[str], max_chars: int) -> str:
    """Trim a long excerpt while keeping the best available query-term match."""

    if max_chars <= 3:
        return "." * max(0, max_chars)
    lower = excerpt.lower()
    match_start = -1
    for term in terms:
        for variant in term_variants(term):
            index = lower.find(variant)
            if index >= 0 and (match_start < 0 or index < match_start):
                match_start = index
    if match_start < 0:
        return excerpt[: max_chars - 3].rstrip() + "..."
    context_before = min(match_start, max_chars // 3)
    start = max(0, match_start - context_before)
    end = min(len(excerpt), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(excerpt) else ""
    body_budget = max_chars - len(prefix) - len(suffix)
    body = excerpt[start : start + body_budget].strip()
    return f"{prefix}{body}{suffix}"


def is_context_label_line(line: str) -> bool:
    """Return whether a nearby short line should be kept as excerpt context."""

    stripped = line.strip()
    return bool(stripped and len(stripped) <= 120)


def excerpt_bounds(
    lines: list[str], target: int, max_lines: int, max_chars: int
) -> tuple[int, int]:
    """Find excerpt bounds around a target line, preferring paragraph boundaries."""

    start = target
    while start > 0 and lines[start - 1].strip():
        start -= 1
    end = target + 1
    while end < len(lines) and lines[end].strip():
        end += 1
    if end - start > max_lines or excerpt_char_count(lines, start, end) > max_chars:
        half = max(1, max_lines // 2)
        start = max(0, target - half)
        end = min(len(lines), start + max_lines)
        start = max(0, end - max_lines)
    while end - start > 1 and excerpt_char_count(lines, start, end) > max_chars:
        if target - start > end - target - 1:
            start += 1
        else:
            end -= 1
    return start, end


def excerpt_char_count(lines: list[str], start: int, end: int) -> int:
    """Return the character count for a line slice joined with newlines."""

    if end <= start:
        return 0
    return sum(len(line) for line in lines[start:end]) + max(0, end - start - 1)
