from __future__ import annotations

import json
from pathlib import Path

from ddserve.cache import cache_paths, read_cache_manifest, read_docset_manifest


def split_filter_values(values: list[str] | None) -> list[str]:
    """Implement split filter values."""
    if not values:
        return []
    output: list[str] = []
    for value in values:
        output.extend(item.strip() for item in value.split(",") if item.strip())
    return output


def resolve_docset_filters(
    cache_root: str | Path, slugs: list[str] | None = None, languages: list[str] | None = None
) -> set[str] | None:
    """Resolve docset filters."""
    slug_values = set(split_filter_values(slugs))
    language_values = {value.lower() for value in split_filter_values(languages)}
    if not slug_values and not language_values:
        return None
    manifest = read_cache_manifest(cache_root)
    resolved = {slug for slug in slug_values if slug in manifest.docs}
    for slug, summary in manifest.docs.items():
        if (
            summary.slug.lower() in language_values
            or summary.name.lower() in language_values
            or summary.type.lower() in language_values
        ):
            resolved.add(slug)
            continue
        docset = read_docset_manifest(cache_root, slug)
        if docset and docset_aliases(cache_root, slug) & language_values:
            resolved.add(slug)
    return resolved


def docset_aliases(cache_root: str | Path, slug: str) -> set[str]:
    """Return aliases stored in a docset metadata file."""
    path = cache_paths(cache_root).docs_root / slug / "raw" / "docset.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()
    if not isinstance(raw, dict):
        return set()
    aliases = raw.get("aliases", raw.get("alias"))
    if isinstance(aliases, str):
        return {aliases.lower()}
    if isinstance(aliases, list):
        return {item.lower() for item in aliases if isinstance(item, str)}
    return set()
