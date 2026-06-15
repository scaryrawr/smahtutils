from __future__ import annotations

from pathlib import Path

from .config import SmahtiepantsConfig
from .models import to_jsonable
from .search import search_docs
from .server_shared import get_page_content, list_docsets


def handle_mcp_request(
    cache_root: str | Path, config: SmahtiepantsConfig, payload: dict[str, object]
) -> dict[str, object]:
    """Handle mcp request."""
    method = str(payload.get("method") or "")
    request_id = payload.get("id")
    try:
        if method == "initialize":
            result: object = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "smahtiepants", "version": "0.1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": "list_docsets", "description": "List installed DevDocs docsets."},
                    {"name": "search_docs", "description": "Search installed documentation."},
                    {
                        "name": "get_page_content",
                        "description": "Read Markdown content for an installed page.",
                    },
                ]
            }
        elif method == "tools/call":
            params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
            name = str(params.get("name") or "")
            arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            result = {
                "content": [
                    {"type": "text", "text": tool_text(cache_root, config, name, arguments)}
                ]
            }
        elif method == "resources/read":
            params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
            uri = str(params.get("uri") or "")
            result = read_resource(cache_root, uri)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Method not found"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": to_jsonable(result)}
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}}


def make_fastmcp_server(
    cache_root: str | Path,
    config: SmahtiepantsConfig,
    host: str,
    port: int,
):
    """Create the official Streamable HTTP MCP server for smahtiepants."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
        raise RuntimeError("official `mcp` package is required to run the MCP server") from exc

    server = FastMCP(
        "smahtiepants",
        host=host,
        port=port,
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        log_level="WARNING",
    )

    @server.tool(
        name="list_docsets",
        description="List installed DevDocs docsets.",
        structured_output=False,
    )
    def _list_docsets() -> str:
        return tool_text(cache_root, config, "list_docsets", {})

    @server.tool(
        name="search_docs",
        description="Search installed documentation.",
        structured_output=False,
    )
    def _search_docs(
        query: str,
        slugs: list[str] | None = None,
        languages: list[str] | None = None,
        limit: int = 10,
    ) -> str:
        return tool_text(
            cache_root,
            config,
            "search_docs",
            {"query": query, "slugs": slugs, "languages": languages, "limit": limit},
        )

    @server.tool(
        name="get_page_content",
        description="Read Markdown content for an installed page.",
        structured_output=False,
    )
    def _get_page_content(
        slug: str,
        pageId: str = "",
        page_id: str = "",
        startLine: int | None = None,
        endLine: int | None = None,
    ) -> str:
        return tool_text(
            cache_root,
            config,
            "get_page_content",
            {
                "slug": slug,
                "pageId": pageId,
                "page_id": page_id,
                "startLine": startLine,
                "endLine": endLine,
            },
        )

    @server.resource(
        "smahtiepants://docsets/{slug}/pages/{pageId}",
        name="page_content",
        description="Read Markdown content for an installed DevDocs page.",
        mime_type="text/markdown",
    )
    def _page_content_resource(slug: str, pageId: str) -> str:
        return get_page_content(cache_root, slug, pageId).content

    return server


def tool_text(
    cache_root: str | Path, config: SmahtiepantsConfig, name: str, arguments: dict[str, object]
) -> str:
    """Implement tool text."""
    if name == "list_docsets":
        return "\n".join(f"{docset.slug}: {docset.name}" for docset in list_docsets(cache_root))
    if name == "search_docs":
        query = str(arguments.get("query") or "")
        slugs = string_list(arguments.get("slugs"))
        languages = string_list(arguments.get("languages"))
        limit = int(arguments.get("limit") or 10)
        results = search_docs(
            cache_root, query, config, slugs=slugs, languages=languages, limit=limit
        )
        return "\n\n".join(
            "\n".join(
                [
                    f"{result.docset_slug}:{result.page_id} {result.page_title}",
                    f"Match: {result.match_kind} score {result.score:.3f}",
                    f"Read full page: {result.read_hint}",
                    "Excerpt:",
                    result.excerpt,
                ]
            )
            for result in results
        )
    if name == "get_page_content":
        slug = str(arguments.get("slug") or "")
        page_id = str(arguments.get("pageId") or arguments.get("page_id") or "")
        start_line = optional_int(arguments.get("startLine"))
        end_line = optional_int(arguments.get("endLine"))
        return get_page_content(cache_root, slug, page_id, start_line, end_line).content
    raise ValueError(f"Unknown tool: {name}")


def read_resource(cache_root: str | Path, uri: str) -> dict[str, object]:
    """Read resource."""
    prefix = "smahtiepants://docsets/"
    if not uri.startswith(prefix):
        raise ValueError(f"Unsupported resource URI: {uri}")
    rest = uri[len(prefix) :]
    slug, _, page_part = rest.partition("/pages/")
    content = get_page_content(cache_root, slug, page_part).content
    return {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": content}]}


def string_list(value: object) -> list[str] | None:
    """Implement string list."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def optional_int(value: object) -> int | None:
    """Implement optional int."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
