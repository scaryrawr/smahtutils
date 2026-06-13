from __future__ import annotations

from pathlib import Path

from smahtiepants.aliases import read_installed_docset_aliases, resolve_installed_docset_slugs
from smahtiepants.cache import read_cache_manifest


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
    resolved = resolve_installed_docset_slugs(cache_root, list(slug_values))
    for slug, summary in manifest.docs.items():
        if (
            summary.slug.lower() in language_values
            or summary.name.lower() in language_values
            or summary.type.lower() in language_values
        ):
            resolved.add(slug)
            continue
        if read_installed_docset_aliases(cache_root, slug) & language_values:
            resolved.add(slug)
    return resolved
