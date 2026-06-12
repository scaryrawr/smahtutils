from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .app import build_state, start_mcp_state
from .mcp_server import serve
from .models import QueryMode
from .serialization import to_jsonable
from .service import index_path, list_indexed, query_code, status


def main() -> None:
    asyncio.run(async_main())


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command
    api_required = command in {None, "index"} or (
        command == "query" and args.mode != QueryMode.KEYWORD.value
    )
    state = build_state(Path(args.root), args.base_url, args.coding_embedding_model, api_required)

    if command == "query":
        response = await query_code(
            state,
            args.query,
            args.limit,
            args.mode,
            args.path_prefix,
            args.language,
        )
        print_json_or_query(response, args.json)
        return
    if command == "index":
        await index_path(state, str(args.path))
        print(f"Indexing {args.path}. Press Ctrl+C to stop and resume later.")
        outcome = await state.indexer.run_until_idle_or_interrupt()
        payload = {
            "status": outcome.status,
            "completed": outcome.summary.completed,
            "requeued": outcome.summary.requeued,
            "failed": outcome.summary.failed,
            "queue": to_jsonable(state.indexer.queue_stats()),
            "store": to_jsonable(state.store.stats()),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"{payload['status']}: {payload['completed']} completed, "
                f"{payload['requeued']} requeued, {payload['failed']} failed."
            )
        return
    if command == "status":
        response = await status(state)
        print_json_or_status(response, args.json)
        return
    if command in {"list-indexed", "list"}:
        response = list_indexed(
            state,
            args.path_prefix,
            args.language,
            args.limit,
            args.offset,
            args.include_source,
        )
        print_json_or_list(response, args.json)
        return

    _watcher = await start_mcp_state(state)
    await serve(state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smahties")
    parser.add_argument(
        "--root", default=".", help="Repository or local coding directory to serve."
    )
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--coding-embedding-model", dest="coding_embedding_model")
    subparsers = parser.add_subparsers(dest="command")

    query = subparsers.add_parser("query")
    query.add_argument("query")
    query.add_argument("--limit", type=int)
    query.add_argument(
        "--mode", choices=[item.value for item in QueryMode], default=QueryMode.HYBRID.value
    )
    query.add_argument("--path-prefix")
    query.add_argument("--language")
    query.add_argument("--json", action="store_true")

    index = subparsers.add_parser("index")
    index.add_argument("path", nargs="?", default=".")
    index.add_argument("--json", action="store_true")

    stat = subparsers.add_parser("status")
    stat.add_argument("--json", action="store_true")

    list_indexed_parser = subparsers.add_parser("list-indexed", aliases=["list"])
    list_indexed_parser.add_argument("--path-prefix")
    list_indexed_parser.add_argument("--language")
    list_indexed_parser.add_argument("--limit", type=int)
    list_indexed_parser.add_argument("--offset", type=int)
    list_indexed_parser.add_argument("--include-source", action="store_true")
    list_indexed_parser.add_argument("--json", action="store_true")

    return parser


def print_json_or_query(response: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(to_jsonable(response), indent=2))
        return
    matches = response.matches
    if not matches:
        print("No matches.")
        return
    for index, item in enumerate(matches, start=1):
        name = f" {item.unit.name}" if item.unit.name else ""
        print(
            f"{index}. {item.unit.file_path}:{item.unit.start_line}-{item.unit.end_line}{name} "
            f"[{item.unit.language} {item.unit.unit_type} {item.match_kind.value} score {item.score:.3f}]"
        )
        for line in item.unit.source.splitlines()[:12]:
            print(f"    {line}")


def print_json_or_status(response: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(to_jsonable(response), indent=2))
        return
    print(f"Root: {response.root}")
    if response.repository_root:
        print(f"Repository root: {response.repository_root}")
    print(f"Runtime root: {response.runtime_root}")
    print(f"Scope: {response.scope_prefix or '<root>'}")
    print(f"Auto indexing: {'enabled' if response.auto_indexing_enabled else 'disabled'}")
    print(f"Model: {response.model}")
    print(
        f"Queue: {response.queue.high_priority} high, {response.queue.low_priority} low, "
        f"{response.queue.in_progress} in progress"
    )
    print(
        f"Indexed: {response.store.indexed_files} files, {response.store.indexed_units} units, "
        f"{response.store.embedded_units} embeddings, {response.store.lexical_units} lexical units"
    )


def print_json_or_list(response: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(to_jsonable(response), indent=2))
        return
    if not response.items:
        print("No indexed units.")
        return
    for item in response.items:
        name = f" {item.name}" if item.name else ""
        print(
            f"{item.file_path}:{item.start_line}-{item.end_line}{name} [{item.language} {item.unit_type}]"
        )
        if item.source:
            for line in item.source.splitlines()[:12]:
                print(f"    {line}")
    print(f"Showing {len(response.items)} items from offset {response.offset}.")


if __name__ == "__main__":
    main()
