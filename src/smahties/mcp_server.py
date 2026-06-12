from __future__ import annotations

from .serialization import to_jsonable
from .service import AppState, index_path, list_indexed, query_code, status


async def serve(state: AppState) -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("official `mcp` package is required to run the MCP server") from exc

    server = FastMCP(
        "smahties",
        instructions="Local semantic code search scoped to the directory where smahties is running.",
    )

    @server.tool()
    async def query_code_tool(
        query: str,
        limit: int | None = None,
        mode: str | None = None,
        path_prefix: str | None = None,
        language: str | None = None,
    ) -> dict[str, object]:
        """Search indexed code with semantic, keyword, or hybrid ranking."""
        return to_jsonable(await query_code(state, query, limit, mode, path_prefix, language))

    @server.tool()
    async def index_path_tool(path: str) -> dict[str, object]:
        """Request high-priority indexing for a file or directory under the active scope."""
        return to_jsonable(await index_path(state, path))

    @server.tool()
    async def status_tool() -> dict[str, object]:
        """Show indexing status, queue counts, recent errors, and active indexer lease state."""
        return to_jsonable(await status(state))

    @server.tool()
    def list_indexed_tool(
        path_prefix: str | None = None,
        language: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include_source: bool | None = None,
    ) -> dict[str, object]:
        """List indexed code units with path/language filters and bounded pagination."""
        return to_jsonable(
            list_indexed(state, path_prefix, language, limit, offset, include_source)
        )

    await server.run_stdio_async()
