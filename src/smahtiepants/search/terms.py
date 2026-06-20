from __future__ import annotations

import re

KEYWORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "do",
    "for",
    "how",
    "i",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "with",
}


def parse_keyword_terms(query: str) -> list[str]:
    """Parse keyword terms."""
    seen: set[str] = set()
    terms: list[str] = []
    for normalized in re.findall(r"[a-z0-9_]+", query.lower()):
        if normalized and normalized not in KEYWORD_STOPWORDS and normalized not in seen:
            terms.append(normalized)
            seen.add(normalized)
        if len(terms) >= 8:
            break
    return terms


def term_variants(term: str) -> tuple[str, ...]:
    """Return simple lexical variants for search term matching."""

    variants = [term]
    if len(term) > 3 and term.endswith("ies"):
        variants.append(f"{term[:-3]}y")
    if len(term) > 3 and term.endswith("s"):
        variants.append(term[:-1])
    return tuple(dict.fromkeys(variants))


def keyword_fts_terms(terms: list[str]) -> list[str]:
    """Expand parsed keyword terms into sanitized FTS prefix terms."""

    seen: set[str] = set()
    output: list[str] = []
    for term in terms:
        for variant in term_variants(term):
            if variant not in seen:
                output.append(variant)
                seen.add(variant)
    return output
