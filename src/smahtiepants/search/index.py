from __future__ import annotations

import html
import json
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path

from smahtiepants.config import SmahtiepantsConfig
from smahtiepants.errors import SmahtiepantsError
from smahtiepants.models import to_jsonable

from smahtiepants.embeddings.openai import EmbeddingClient, create_openai_embedding_client
from smahtiepants.embeddings.annoy_index import AnnoyIndexManager
from smahtiepants.embeddings.storage import (
    EmbeddingStorage,
    SearchChunk,
    cosine_similarity,
    open_embedding_storage,
)

from .filters import resolve_docset_filters
from .terms import parse_keyword_terms, term_variants

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
        keyword_limit = KEYWORD_RESULT_POOL
        keyword_chunks = storage.keyword_chunks(terms, resolved_slugs, keyword_limit)
        keyword_ids = {chunk.id for chunk in keyword_chunks}
        if config.openai is not None and config.embeddings.enabled:
            embedding_client = client or create_openai_embedding_client(config, env)
            query_vector = embedding_client.create_embeddings(query)[0]
            chunks = semantic_candidate_chunks(
                cache_root,
                storage,
                config.openai.embedding_model,
                query_vector,
                resolved_slugs,
                keyword_chunks,
            )
            semantic = diversify_search_results(
                sorted(
                    (
                        chunk_to_result(
                            chunk,
                            cosine_similarity(query_vector, chunk.vector or []),
                            "hybrid" if chunk.id in keyword_ids else "semantic",
                            terms,
                        )
                        for chunk in chunks
                    ),
                    key=lambda item: item.score,
                    reverse=True,
                ),
                SEMANTIC_RESULT_POOL,
            )
        keyword = [
            chunk_to_result(
                chunk,
                max(0.0, 1.0 - index / max(1, keyword_limit)),
                "keyword",
                terms,
            )
            for index, chunk in enumerate(keyword_chunks)
        ]
        if semantic and keyword:
            return merge_search_results(semantic, keyword, bounded_limit)
        if semantic:
            return diversify_search_results(semantic, bounded_limit)
        return diversify_search_results(keyword, bounded_limit)
    finally:
        storage.close()


def semantic_candidate_chunks(
    cache_root: str | Path,
    storage: EmbeddingStorage,
    model: str,
    query_vector: list[float],
    resolved_slugs: set[str] | None,
    keyword_chunks: list[SearchChunk],
) -> list[SearchChunk]:
    """Return exact-scoreable semantic candidates for the requested search scope."""

    if resolved_slugs is not None:
        return storage.chunks_with_vectors(model, resolved_slugs, None)
    manager = AnnoyIndexManager(cache_root, storage)
    candidate_ids = manager.search(model, query_vector, SEMANTIC_CANDIDATE_COUNT)
    chunks = storage.chunks_by_ids(candidate_ids, model, None)
    if keyword_chunks:
        keyword_vector_chunks = storage.chunks_by_ids(
            (chunk.id for chunk in keyword_chunks), model, None
        )
        chunks = merge_search_chunks(chunks, keyword_vector_chunks)
    return chunks


def merge_search_chunks(primary: list[SearchChunk], extra: list[SearchChunk]) -> list[SearchChunk]:
    """Append chunks not already present, preserving primary order."""

    seen = {chunk.id for chunk in primary}
    merged = list(primary)
    for chunk in extra:
        if chunk.id in seen:
            continue
        merged.append(chunk)
        seen.add(chunk.id)
    return merged


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
    read_hint = full_page_read_hint(chunk.docset_slug, chunk.page_id, resource_uri)
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


def full_page_read_hint(slug: str, page_id: str, resource_uri: str) -> str:
    """Return CLI, MCP, and resource options for reading a matched page."""

    command = f"uv run smahtiepants docs page {shlex.quote(slug)} {shlex.quote(page_id)}"
    return (
        f"CLI: {command}; "
        f'MCP: get_page_content slug="{slug}" pageId="{page_id}"; '
        f"resource: {resource_uri}"
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

    lines, leading_omitted = excerpt_source_lines(strip_chunk_metadata(text), terms)
    if not any(line.strip() for line in lines):
        lines, leading_omitted = excerpt_source_lines(text, terms)
    if not lines:
        return ""
    target = best_matching_line(lines, terms)
    start, end = excerpt_bounds(lines, target, max_lines, max_chars)
    start, end = expand_single_line_context(lines, target, start, end, max_lines, max_chars)
    if start >= 2 and not lines[start - 1].strip() and is_context_label_line(lines[start - 2]):
        start -= 2
    excerpt_lines = lines[start:end]
    if start > 0 or leading_omitted:
        excerpt_lines.insert(0, "...")
    if end < len(lines):
        excerpt_lines.append("...")
    excerpt = "\n".join(excerpt_lines)
    if len(excerpt) > max_chars:
        return trim_excerpt(excerpt, terms, max_chars)
    return excerpt


def excerpt_source_lines(text: str, terms: list[str]) -> tuple[list[str], bool]:
    """Return normalized excerpt lines and whether leading chunk text was omitted."""

    normalized = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    lines = [line.rstrip() for line in normalized.splitlines()]
    lines = [line for line in lines if not is_excerpt_noise_line(line)]
    lines = collapse_blank_lines(lines)
    leading_omitted = False
    while lines and not lines[0].strip():
        lines.pop(0)
    if len(lines) > 1 and is_partial_leading_line(lines[0], terms):
        lines.pop(0)
        leading_omitted = True
        while lines and not lines[0].strip():
            lines.pop(0)
    return lines, leading_omitted


def collapse_blank_lines(lines: list[str]) -> list[str]:
    """Collapse repeated blank lines for compact CLI excerpts."""

    output: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        output.append("" if blank else line)
        previous_blank = blank
    while output and not output[0].strip():
        output.pop(0)
    while output and not output[-1].strip():
        output.pop()
    return output


def is_excerpt_noise_line(line: str) -> bool:
    """Return whether a generated documentation line is poor excerpt context."""

    stripped = line.strip()
    return bool(
        stripped == "[Tests](javascript:void(0))"
        or stripped == "Tests with this rule:"
        or stripped.startswith("-   [tests/")
        or stripped.startswith("- [tests/")
        or stripped.startswith("> DevDocs path:")
    )


def is_partial_leading_line(line: str, terms: list[str]) -> bool:
    """Return whether the first chunk line appears to start mid-token or mid-markup."""

    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^[A-Za-z0-9_]*\]\(", stripped):
        return True
    if stripped == "`" or stripped[0] in ")]},.;:":
        return True
    structural_prefixes = ("#", ">", "-", "*", "+", "|", "```", "~~~", "[", "`")
    if stripped.startswith(structural_prefixes) or re.match(r"^\d+[.)]\s", stripped):
        return False
    return bool(stripped[0].islower() and line_term_score(stripped, terms) == 0)


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
    start = trim_start_to_boundary(excerpt, start, match_start)
    end = min(len(excerpt), start + max_chars)
    end = trim_end_to_boundary(excerpt, start, end)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(excerpt) else ""
    body_budget = max_chars - len(prefix) - len(suffix)
    body = excerpt[start : start + body_budget].strip()
    return f"{prefix}{body}{suffix}"


def trim_start_to_boundary(text: str, start: int, match_start: int) -> int:
    """Move a trim start forward to avoid leading partial words."""

    if start <= 0 or start >= match_start:
        return start
    if text[start - 1].isspace() or text[start].isspace():
        return start
    for index in range(start, match_start):
        if text[index].isspace():
            return index + 1
    return start


def trim_end_to_boundary(text: str, start: int, end: int) -> int:
    """Move a trim end backward to avoid trailing partial words."""

    if end >= len(text) or end <= start:
        return end
    if text[end - 1].isspace() or text[end].isspace():
        return end
    minimum = max(start, end - 40)
    for index in range(end - 1, minimum, -1):
        if text[index].isspace():
            return index
    return end


def is_context_label_line(line: str) -> bool:
    """Return whether a nearby short line should be kept as excerpt context."""

    stripped = line.strip()
    return bool(stripped and len(stripped) <= 120)


def is_markdown_heading(line: str) -> bool:
    """Return whether a line is a Markdown heading."""

    return bool(re.match(r"^#{1,6}\s+\S", line.strip()))


def expand_single_line_context(
    lines: list[str], target: int, start: int, end: int, max_lines: int, max_chars: int
) -> tuple[int, int]:
    """Add the following paragraph for isolated headings or labels."""

    if end - start != 1 or target < start or target >= end:
        return start, end
    if not (is_markdown_heading(lines[target]) or is_reference_label_line(lines[target])):
        return start, end
    cursor = target + 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor >= len(lines):
        return start, end
    expanded_end = cursor
    included_plain_text = False
    while cursor < len(lines):
        block_end = cursor + 1
        while block_end < len(lines) and lines[block_end].strip():
            block_end += 1
        candidate_end = block_end
        if (
            candidate_end - start > max_lines
            or excerpt_char_count(lines, start, candidate_end) > max_chars
        ):
            break
        expanded_end = candidate_end
        included_plain_text = included_plain_text or block_has_plain_text(lines[cursor:block_end])
        if included_plain_text:
            break
        cursor = block_end
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
    return start, max(end, expanded_end)


def block_has_plain_text(lines: list[str]) -> bool:
    """Return whether a block has explanatory text beyond headings or fences."""

    return any(
        stripped and not is_markdown_heading(stripped) and not stripped.startswith(("```", "~~~"))
        for stripped in (line.strip() for line in lines)
    )


def is_reference_label_line(line: str) -> bool:
    """Return whether a line is a standalone Markdown reference label."""

    return bool(re.match(r"^\[[^\]]+\]\([^)]+\)$", line.strip()))


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
