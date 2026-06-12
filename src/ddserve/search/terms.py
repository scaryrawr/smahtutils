from __future__ import annotations

import re


def parse_keyword_terms(query: str) -> list[str]:
    """Parse keyword terms."""
    seen: set[str] = set()
    terms: list[str] = []
    for normalized in re.findall(r"[a-z0-9_]+", query.lower()):
        if normalized and normalized not in seen:
            terms.append(normalized)
            seen.add(normalized)
        if len(terms) >= 8:
            break
    return terms
