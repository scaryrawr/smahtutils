from __future__ import annotations

import argparse
import json
import sys

from .aliases import resolve_installed_docset_slug
from .cache import read_cache_manifest, read_docset_manifest, resolve_cache_root
from .config import load_config, redact_config
from .devdocs import get_available_docsets
from .embeddings.index import (
    rebuild_docset_embeddings,
    refresh_docset_embeddings,
    status_for_embeddings,
)
from .errors import SmahtiepantsError
from .format import format_bytes, format_table
from .install import install_docsets, remove_docset, update_docsets
from .models import to_jsonable
from .search.index import results_to_json, results_to_text, results_to_xml, search_docs
from .server import serve
from .server_shared import get_page_content


def main() -> None:
    """Implement main."""
    try:
        run_cli(sys.argv[1:])
    except SmahtiepantsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def run_cli(argv: list[str]) -> None:
    """Implement run cli."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cache_root = str(resolve_cache_root())
    if args.command == "cache" and args.cache_command == "path":
        print(cache_root)
        return
    if args.command == "config":
        handle_config(args)
        return
    if args.command == "sources":
        handle_sources(args)
        return
    if args.command == "docs":
        handle_docs(args, cache_root)
        return
    if args.command == "embeddings":
        handle_embeddings(args, cache_root)
        return
    if args.command == "search":
        handle_search(args, cache_root)
        return
    if args.command == "serve":
        loaded = load_config(args.config)
        serve(cache_root, loaded.config, args.host, args.port)
        return
    parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    """Implement build parser."""
    parser = argparse.ArgumentParser(prog="smahtiepants")
    parser.add_argument("--config")
    subparsers = parser.add_subparsers(dest="command")

    cache = subparsers.add_parser("cache")
    cache_sub = cache.add_subparsers(dest="cache_command")
    cache_sub.add_parser("path")

    sources = subparsers.add_parser("sources")
    sources_sub = sources.add_subparsers(dest="sources_command")
    sources_list = sources_sub.add_parser("list")
    sources_list.add_argument("--json", action="store_true")

    docs = subparsers.add_parser("docs")
    docs_sub = docs.add_subparsers(dest="docs_command")
    available = docs_sub.add_parser("available")
    available.add_argument("--json", action="store_true")
    available.add_argument("--offline", action="store_true")
    installed = docs_sub.add_parser("installed")
    installed.add_argument("--json", action="store_true")
    install = docs_sub.add_parser("install")
    install.add_argument("slug", nargs="+")
    install.add_argument("--json", action="store_true")
    install.add_argument("--force", action="store_true")
    install.add_argument("--offline", action="store_true")
    update = docs_sub.add_parser("update")
    update.add_argument("slug", nargs="?")
    update.add_argument("--json", action="store_true")
    update.add_argument("--force", action="store_true")
    update.add_argument("--offline", action="store_true")
    remove = docs_sub.add_parser("remove")
    remove.add_argument("slug")
    remove.add_argument("--json", action="store_true")
    page = docs_sub.add_parser("page")
    page.add_argument("slug")
    page.add_argument("page_id")
    page.add_argument("--start-line", type=int)
    page.add_argument("--end-line", type=int)
    page.add_argument("--json", action="store_true")

    embeddings = subparsers.add_parser("embeddings")
    emb_sub = embeddings.add_subparsers(dest="embeddings_command")
    for name in ("status", "refresh", "rebuild"):
        command = emb_sub.add_parser(name)
        command.add_argument("slug", nargs="?")
        command.add_argument("--json", action="store_true")

    search = subparsers.add_parser("search")
    search.add_argument("query")
    search.add_argument("--slug", action="append")
    search.add_argument("--language", action="append")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--format", choices=["text", "json", "xml"], default="text")
    search.add_argument("--json", action="store_true")

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host")
    serve_parser.add_argument("--port", type=int)

    config = subparsers.add_parser("config")
    config_sub = config.add_subparsers(dest="config_command")
    config_sub.add_parser("path")
    show = config_sub.add_parser("show")
    show.add_argument("--json", action="store_true")
    return parser


def handle_config(args: argparse.Namespace) -> None:
    """Handle config."""
    loaded = load_config(args.config)
    if args.config_command == "path":
        print(loaded.path)
        return
    payload = {
        "path": str(loaded.path),
        "found": loaded.found,
        "config": redact_config(loaded.config),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload["config"], indent=2))


def handle_sources(args: argparse.Namespace) -> None:
    """Handle sources."""
    sources = [{"name": "devdocs", "url": "https://devdocs.io/"}]
    if args.json:
        print(json.dumps({"sources": sources}, indent=2))
    else:
        print(
            format_table(
                sources, [("Name", lambda row: row["name"]), ("URL", lambda row: row["url"])]
            )
        )


def handle_docs(args: argparse.Namespace, cache_root: str) -> None:
    """Handle docs."""
    if args.docs_command == "available":
        result = get_available_docsets(cache_root, offline=args.offline)
        if args.json:
            print(json.dumps(to_jsonable(result), indent=2))
        else:
            installed_slugs = set(read_cache_manifest(cache_root).docs)
            columns = [
                ("Slug", lambda row: row.slug),
                ("Aliases", lambda row: ", ".join(row.aliases)),
                ("Name", lambda row: row.name),
                ("Type", lambda row: row.type),
                ("Size", lambda row: format_bytes(row.db_size)),
            ]
            if any(docset.slug in installed_slugs for docset in result.docsets):
                columns.insert(0, ("", lambda row: "*" if row.slug in installed_slugs else ""))
            print(
                format_table(
                    result.docsets,
                    columns,
                )
            )
            print_warnings(result.warnings)
        return
    if args.docs_command == "installed":
        manifest = read_cache_manifest(cache_root)
        docs = [doc for _slug, doc in sorted(manifest.docs.items())]
        if args.json:
            print(json.dumps({"docsets": [to_jsonable(doc) for doc in docs]}, indent=2))
        else:
            print(
                format_table(
                    docs,
                    [
                        ("Slug", lambda row: row.slug),
                        ("Name", lambda row: row.name),
                        ("Pages", lambda row: row.page_count),
                        ("Updated", lambda row: row.updated_at),
                    ],
                )
            )
        return
    if args.docs_command == "install":
        results = install_docsets(
            args.slug,
            cache_root,
            force=args.force,
            offline=args.offline,
            config_path=args.config,
            on_progress=None if args.json else print_docs_progress("Installing"),
        )
        print_results(results, args.json, include_warnings=args.json)
        return
    if args.docs_command == "update":
        results = update_docsets(
            args.slug,
            cache_root,
            force=args.force,
            offline=args.offline,
            config_path=args.config,
            on_progress=None if args.json else print_docs_progress("Updating"),
        )
        print_results(results, args.json, include_warnings=args.json)
        return
    if args.docs_command == "remove":
        result = remove_docset(args.slug, cache_root)
        if args.json:
            print(json.dumps(to_jsonable(result), indent=2))
        else:
            print(f"Removed {result.slug} ({result.pages} pages).")
        return
    if args.docs_command == "page":
        content = get_page_content(
            cache_root,
            args.slug,
            args.page_id,
            start_line=args.start_line,
            end_line=args.end_line,
        )
        if args.json:
            print(json.dumps(to_jsonable(content), indent=2))
        else:
            print(content.content, end="")
        return


def handle_embeddings(args: argparse.Namespace, cache_root: str) -> None:
    """Handle embeddings."""
    loaded = load_config(args.config)
    if args.embeddings_command == "status":
        status = status_for_embeddings(cache_root, loaded.config, args.slug)
        if args.json:
            print(json.dumps(to_jsonable(status), indent=2))
        else:
            print(f"Database: {status.database_path}")
            print(f"Enabled: {status.enabled}")
            print(f"Configured: {status.configured}")
            print(f"Model: {status.model or '<none>'}")
            print(f"Installed: {status.installed_docsets} docsets, {status.installed_pages} pages")
            print(f"Indexed: {status.indexed_chunks} chunks, {status.embedded_chunks} embeddings")
        return
    if not args.slug:
        raise SmahtiepantsError(f"embeddings {args.embeddings_command} requires a docset slug")
    canonical_slug = resolve_installed_docset_slug(cache_root, args.slug) or args.slug
    manifest = read_docset_manifest(cache_root, canonical_slug)
    if manifest is None:
        raise SmahtiepantsError(f'Docset "{args.slug}" is not installed.')
    if args.embeddings_command == "refresh":
        result = refresh_docset_embeddings(cache_root, manifest, loaded.config)
    else:
        result = rebuild_docset_embeddings(cache_root, manifest, loaded.config)
    print(
        json.dumps(result, indent=2) if args.json else f"{result['embedded']} embeddings written."
    )


def handle_search(args: argparse.Namespace, cache_root: str) -> None:
    """Handle search."""
    loaded = load_config(args.config)
    results = search_docs(
        cache_root,
        args.query,
        loaded.config,
        slugs=args.slug,
        languages=args.language,
        limit=args.limit,
    )
    output_format = "json" if args.json else args.format
    if output_format == "json":
        print(results_to_json(results))
    elif output_format == "xml":
        print(results_to_xml(results))
    else:
        print(results_to_text(results))


def print_results(results: list[object], as_json: bool, include_warnings: bool = True) -> None:
    """Implement print results."""
    if as_json:
        print(json.dumps({"results": [to_jsonable(result) for result in results]}, indent=2))
        return
    for result in results:
        print(f"{result.slug}: {result.status} ({format_install_result_counts(result)})")
        if include_warnings:
            print_warnings(result.warnings)


def print_docs_progress(label: str):
    """Return a docs progress printer."""

    def print_progress(
        slug: str, index: int, total: int, phase: str, result: object | None
    ) -> None:
        """Print docs progress."""
        if phase == "start":
            print(f"{label} {slug} ({index}/{total})...", file=sys.stderr)
            return
        if phase == "embedding" and isinstance(result, dict):
            completed = int(result.get("completed", 0))
            batches = int(result.get("total", 0))
            interval = max(1, batches // 10)
            if completed in {1, batches} or completed % interval == 0:
                print(f"Embedding {slug}: {completed}/{batches} batches...", file=sys.stderr)
            return
        if result is not None:
            print(
                f"Finished {slug}: {result.status} ({format_install_result_counts(result)})",
                file=sys.stderr,
            )
            print_warnings(result.warnings)

    return print_progress


def format_install_result_counts(result: object) -> str:
    """Format install/update counters."""
    parts = [f"{result.pages} pages", f"{result.skipped_entries} skipped"]
    embedding_chunks = getattr(result, "embedding_chunks", 0)
    if embedding_chunks:
        parts.append(f"{getattr(result, 'embedded_chunks', 0)}/{embedding_chunks} embedded")
        skipped_embeddings = getattr(result, "skipped_embedding_chunks", 0)
        if skipped_embeddings:
            parts.append(f"{skipped_embeddings} current")
        if getattr(result, "annoy_indexed", False):
            parts.append("search index ready")
    return ", ".join(parts)


def print_warnings(warnings: list[str]) -> None:
    """Implement print warnings."""
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
