from __future__ import annotations

import json
import re
from pathlib import Path

from .cache import cache_paths, read_cache_manifest
from .models import CacheManifest, CacheManifestDocset, DocsetSummary

CURATED_DOCSET_ALIASES = {
    "node.js": "node",
    "nodejs": "node",
    "react-native": "react_native",
    "reactnative": "react_native",
    "reactjs": "react",
    "rn": "react_native",
}


def normalize_docset_identifier(value: str) -> str:
    """Normalize a user-provided docset identifier for comparisons."""
    return value.strip().lower()


def find_docset_by_identifier(
    docsets: list[DocsetSummary], identifier: str
) -> DocsetSummary | None:
    """Find a DevDocs docset by canonical slug, upstream alias, or curated alias."""
    normalized = normalize_docset_identifier(identifier)
    exact = [docset for docset in docsets if normalize_docset_identifier(docset.slug) == normalized]
    if exact:
        return exact[0]
    alias_matches = [
        docset
        for docset in docsets
        if normalized
        in {normalize_docset_identifier(alias) for alias in docset.aliases if alias.strip()}
    ]
    if alias_matches:
        return preferred_docset(alias_matches)
    fallback_slug = CURATED_DOCSET_ALIASES.get(normalized)
    if fallback_slug:
        return next((docset for docset in docsets if docset.slug == fallback_slug), None)
    metadata_matches = [
        docset
        for docset in docsets
        if normalized
        in {normalize_docset_identifier(docset.name), normalize_docset_identifier(docset.type)}
    ]
    if metadata_matches:
        return preferred_docset(metadata_matches)
    return None


def resolve_installed_docset_slug(
    cache_root: str | Path, identifier: str, manifest: CacheManifest | None = None
) -> str | None:
    """Resolve an installed docset identifier to its canonical cache slug."""
    normalized = normalize_docset_identifier(identifier)
    manifest = manifest or read_cache_manifest(cache_root)
    exact = {
        normalize_docset_identifier(slug): slug
        for slug in manifest.docs
        if normalize_docset_identifier(slug) == normalized
    }
    if exact:
        return exact[normalized]
    alias_matches = [
        docset
        for slug, docset in manifest.docs.items()
        if normalized in read_installed_docset_aliases(cache_root, slug)
    ]
    if alias_matches:
        return preferred_cache_docset(alias_matches).slug
    fallback_slug = CURATED_DOCSET_ALIASES.get(normalized)
    if fallback_slug and fallback_slug in manifest.docs:
        return fallback_slug
    metadata_matches = [
        docset
        for docset in manifest.docs.values()
        if normalized
        in {normalize_docset_identifier(docset.name), normalize_docset_identifier(docset.type)}
    ]
    if metadata_matches:
        return preferred_cache_docset(metadata_matches).slug
    return None


def resolve_installed_docset_slugs(
    cache_root: str | Path,
    identifiers: list[str] | None,
    manifest: CacheManifest | None = None,
) -> set[str]:
    """Resolve installed docset identifiers, dropping unknown filters."""
    manifest = manifest or read_cache_manifest(cache_root)
    resolved: set[str] = set()
    for identifier in identifiers or []:
        slug = resolve_installed_docset_slug(cache_root, identifier, manifest)
        if slug:
            resolved.add(slug)
    return resolved


def read_installed_docset_aliases(cache_root: str | Path, slug: str) -> set[str]:
    """Return normalized aliases stored in a docset metadata file."""
    path = cache_paths(cache_root).docs_root / slug / "raw" / "docset.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()
    if not isinstance(raw, dict):
        return set()
    aliases = raw.get("aliases", raw.get("alias"))
    if isinstance(aliases, str):
        return {normalize_docset_identifier(aliases)}
    if isinstance(aliases, list):
        return {
            normalize_docset_identifier(item)
            for item in aliases
            if isinstance(item, str) and item.strip()
        }
    return set()


def preferred_docset(docsets: list[DocsetSummary]) -> DocsetSummary:
    """Choose the default docset for an alias shared by multiple versions."""
    return max(docsets, key=lambda docset: preference_key(docset.slug, docset.version))


def preferred_cache_docset(docsets: list[CacheManifestDocset]) -> CacheManifestDocset:
    """Choose the default installed docset for an alias shared by multiple versions."""
    return max(docsets, key=lambda docset: preference_key(docset.slug, docset.version))


def preference_key(slug: str, version: str | None) -> tuple[int, tuple[int, ...]]:
    """Prefer unversioned docsets, then the newest numeric version."""
    return (0 if "~" in slug else 1, numeric_version_key(slug, version))


def numeric_version_key(slug: str, version: str | None) -> tuple[int, ...]:
    """Extract a comparable numeric version key from DevDocs metadata."""
    value = version or (slug.split("~", 1)[1] if "~" in slug else "")
    return tuple(int(part) for part in re.findall(r"\d+", value))
