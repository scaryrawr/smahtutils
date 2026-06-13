from __future__ import annotations

import re

KEYWORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "how",
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
