from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .config import (
    DEFAULT_SERVE_AUTH_TOKEN_ENV,
    DEFAULT_SERVE_BIND_ADDRESS,
    DEFAULT_SERVE_PORT,
    LEGACY_SERVE_AUTH_TOKEN_ENV,
    SmahtiepantsConfig,
)
from .copilot_hooks import session_start_context
from .embeddings.index import status_for_embeddings
from .errors import SmahtiepantsError
from .mcp_server import make_fastmcp_server
from .models import to_jsonable
from .search import search_docs
from .server_shared import get_docset, get_page, get_page_content, list_docsets, list_pages

ASGIReceive = Any
ASGIScope = dict[str, Any]
ASGISend = Any


def serve(
    cache_root: str | Path,
    config: SmahtiepantsConfig,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Run the read-only REST, MCP, and Copilot hook server."""

    bind = host or (
        config.serve.bind_address
        if config.serve and config.serve.bind_address
        else DEFAULT_SERVE_BIND_ADDRESS
    )
    bind_port = port or (
        config.serve.port if config.serve and config.serve.port else DEFAULT_SERVE_PORT
    )
    app = make_app(str(cache_root), config, bind, bind_port)
    print(f"smahtiepants listening on http://{bind}:{bind_port} (MCP: /mcp)")

    import uvicorn

    uvicorn.run(app, host=bind, port=bind_port, log_level="warning")


def make_app(
    cache_root: str | Path, config: SmahtiepantsConfig, bind: str, port: int | None = None
):
    """Create the ASGI app that serves REST routes and Streamable HTTP MCP."""

    from starlette.requests import Request

    bind_port = port or (
        config.serve.port if config.serve and config.serve.port else DEFAULT_SERVE_PORT
    )
    server = make_fastmcp_server(cache_root, config, bind, bind_port)

    def route(handler):
        async def wrapped(request: Request):
            try:
                return await handler(request)
            except SmahtiepantsError as exc:
                return json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                return json_response({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        return wrapped

    @server.custom_route("/health", methods=["GET"], include_in_schema=False)
    @route
    async def _health(_request: Request):
        return json_response({"ok": True})

    @server.custom_route("/api", methods=["GET"], include_in_schema=False)
    @route
    async def _api(_request: Request):
        return json_response(
            {
                "name": "smahtiepants",
                "links": ["/api/docsets", "/api/search", "/api/embeddings/status"],
            }
        )

    @server.custom_route("/api/docsets", methods=["GET"], include_in_schema=False)
    @route
    async def _api_docsets(_request: Request):
        return json_response({"items": [to_jsonable(item) for item in list_docsets(cache_root)]})

    @server.custom_route("/api/search", methods=["GET", "POST"], include_in_schema=False)
    @route
    async def _api_search(request: Request):
        if request.method == "GET":
            return json_response(
                search_payload(cache_root, config, query_dict(request.query_params))
            )
        body = await read_json_body(request)
        return json_response(search_payload_from_body(cache_root, config, body))

    @server.custom_route("/api/embeddings/status", methods=["GET"], include_in_schema=False)
    @route
    async def _api_embedding_status(_request: Request):
        return json_response(to_jsonable(status_for_embeddings(cache_root, config)))

    @server.custom_route(
        "/api/embeddings/status/{slug:path}", methods=["GET"], include_in_schema=False
    )
    @route
    async def _api_embedding_status_slug(request: Request):
        return json_response(
            to_jsonable(status_for_embeddings(cache_root, config, request.path_params["slug"]))
        )

    @server.custom_route("/copilot/hooks/sessionStart", methods=["POST"], include_in_schema=False)
    @route
    async def _copilot_session_start(request: Request):
        body = await read_json_body(request)
        prompt = (
            str(body.get("prompt") or body.get("message") or "") if isinstance(body, dict) else ""
        )
        return json_response(session_start_context(cache_root, config, prompt, os.environ))

    @server.custom_route("/api/docsets/{rest:path}", methods=["GET"], include_in_schema=False)
    @route
    async def _api_docset_routes(request: Request):
        parts = [
            unquote(part) for part in request.path_params["rest"].strip("/").split("/") if part
        ]
        if not parts:
            return json_response({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        slug = parts[0]
        query = query_dict(request.query_params)
        if len(parts) == 1:
            return json_response(to_jsonable(get_docset(cache_root, slug)))
        if len(parts) == 2 and parts[1] == "pages":
            return json_response(
                list_pages(
                    cache_root,
                    slug,
                    limit=int(first(query, "limit") or 100),
                    offset=int(first(query, "offset") or 0),
                    query=first(query, "q"),
                    type_=first(query, "type"),
                )
            )
        if len(parts) >= 3 and parts[1] == "pages":
            is_content = parts[-1] == "content"
            page_id = "/".join(parts[2:-1] if is_content else parts[2:])
            if is_content:
                return json_response(
                    to_jsonable(
                        get_page_content(
                            cache_root,
                            slug,
                            page_id,
                            optional_int(first(query, "startLine")),
                            optional_int(first(query, "endLine")),
                        )
                    )
                )
            return json_response(to_jsonable(get_page(cache_root, slug, page_id)))
        return json_response({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    return SmahtiepantsHTTPMiddleware(server.streamable_http_app(), config, bind)


class SmahtiepantsHTTPMiddleware:
    """Apply smahtiepants host, auth, and CORS policy around the ASGI app."""

    def __init__(self, app, config: SmahtiepantsConfig, bind: str) -> None:
        self.app = app
        self.config = config
        self.bind = bind

    async def __call__(self, scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = scope_headers(scope)
        path = str(scope.get("path") or "")
        method = str(scope.get("method") or "")

        if not host_allowed(headers, self.bind):
            await send_json(send, {"error": "Host header is not allowed"}, HTTPStatus.FORBIDDEN)
            return

        if method == "OPTIONS":
            await send_empty(
                send, HTTPStatus.NO_CONTENT, common_headers(headers, path, self.config, self.bind)
            )
            return

        if protected_path(path) and not authorized(headers, self.config):
            await send_json(
                send,
                {"error": "Unauthorized"},
                HTTPStatus.UNAUTHORIZED,
                common_headers(headers, path, self.config, self.bind),
            )
            return

        async def send_with_common_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                message["headers"].extend(common_headers(headers, path, self.config, self.bind))
            await send(message)

        await self.app(scope, receive, send_with_common_headers)


async def read_json_body(request) -> object:
    """Read a JSON body, preserving the old empty-body behavior."""

    if not request.headers.get("content-length"):
        return {}
    return await request.json()


def json_response(value: object, status: HTTPStatus = HTTPStatus.OK):
    """Build a JSON Starlette response."""

    from starlette.responses import JSONResponse

    return JSONResponse(normalize_json(value), status_code=int(status))


async def send_empty(
    send: ASGISend, status: HTTPStatus, headers: list[tuple[bytes, bytes]]
) -> None:
    """Send an empty ASGI response."""

    await send({"type": "http.response.start", "status": int(status), "headers": headers})
    await send({"type": "http.response.body", "body": b""})


async def send_json(
    send: ASGISend,
    value: object,
    status: HTTPStatus,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    """Send a JSON ASGI response."""

    body = json.dumps(normalize_json(value), indent=2).encode("utf-8")
    response_headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    response_headers.extend(headers or [])
    await send({"type": "http.response.start", "status": int(status), "headers": response_headers})
    await send({"type": "http.response.body", "body": body})


def scope_headers(scope: ASGIScope) -> dict[str, str]:
    """Return lowercase HTTP headers from an ASGI scope."""

    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }


def protected_path(path: str) -> bool:
    """Return whether the route requires bearer auth when configured."""

    return path.startswith("/api") or path in {"/mcp", "/copilot/hooks/sessionStart"}


def host_allowed(headers: dict[str, str], bind: str) -> bool:
    """Return whether the request Host is allowed for the configured bind address."""

    if bind in {"0.0.0.0", "::"}:
        return True
    raw_host = headers.get("host") or ""
    if raw_host.startswith("["):
        host = raw_host[1:].split("]", 1)[0]
    elif raw_host.count(":") == 1:
        host = raw_host.rsplit(":", 1)[0]
    else:
        host = raw_host
    return host in {bind, "localhost", "127.0.0.1", "::1"}


def authorized(headers: dict[str, str], config: SmahtiepantsConfig) -> bool:
    """Return whether the request has the configured bearer token."""

    if not config.serve or not config.serve.auth:
        return True
    expected = os.environ.get(config.serve.auth.token_env)
    if not expected and config.serve.auth.token_env == DEFAULT_SERVE_AUTH_TOKEN_ENV:
        expected = os.environ.get(LEGACY_SERVE_AUTH_TOKEN_ENV)
    expected = expected or config.serve.auth.token
    if not expected:
        return False
    return headers.get("authorization") == f"Bearer {expected}"


def common_headers(
    headers: dict[str, str], path: str, config: SmahtiepantsConfig, bind: str
) -> list[tuple[bytes, bytes]]:
    """Return common CORS headers for REST and MCP responses."""

    origin = headers.get("origin")
    allow_origin = allowed_cors_origin(origin, path, config, bind)
    if allow_origin is None:
        return []
    return [
        (b"access-control-allow-origin", allow_origin.encode("utf-8")),
        (b"access-control-allow-methods", b"GET,POST,DELETE,OPTIONS"),
        (
            b"access-control-allow-headers",
            b"authorization,content-type,accept,mcp-session-id,mcp-protocol-version,last-event-id",
        ),
        (b"access-control-expose-headers", b"mcp-session-id"),
        (b"vary", b"Origin"),
    ]


def allowed_cors_origin(
    origin: str | None, path: str, config: SmahtiepantsConfig, bind: str
) -> str | None:
    """Resolve the CORS origin allowed for this request."""

    if config.serve and config.serve.cors:
        origins = config.serve.cors.origins
        if "*" in origins:
            return "*"
        if origin in origins:
            return origin
        return None
    if path == "/mcp" and origin and bind in {"127.0.0.1", "localhost", "::1"}:
        return origin
    return None


def search_payload(
    cache_root: str | Path, config: SmahtiepantsConfig, query: dict[str, list[str]]
) -> dict[str, object]:
    """Build the REST search payload."""

    results = search_docs(
        cache_root,
        first(query, "q") or "",
        config,
        slugs=query.get("slug"),
        languages=query.get("language"),
        limit=int(first(query, "limit") or 10),
    )
    return {"matches": [to_jsonable(result) for result in results]}


def search_payload_from_body(
    cache_root: str | Path, config: SmahtiepantsConfig, body: object
) -> dict[str, object]:
    """Build the REST search payload from a JSON body."""

    if not isinstance(body, dict):
        raise SmahtiepantsError("Search body must be a JSON object")
    results = search_docs(
        cache_root,
        str(body.get("query") or ""),
        config,
        slugs=string_list(body.get("slugs")),
        languages=string_list(body.get("languages")),
        limit=int(body.get("limit") or 10),
    )
    return {"matches": [to_jsonable(result) for result in results]}


def first(query: dict[str, list[str]], key: str) -> str | None:
    """Return the first value for a query key."""

    values = query.get(key)
    return values[0] if values else None


def optional_int(value: str | None) -> int | None:
    """Parse an optional integer."""

    return int(value) if value else None


def string_list(value: object) -> list[str] | None:
    """Coerce a JSON value to a string list."""

    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def query_dict(query_params) -> dict[str, list[str]]:
    """Convert Starlette query params to the old dict-of-lists shape."""

    return {key: query_params.getlist(key) for key in query_params}


def normalize_json(value: object) -> object:
    """Normalize dataclasses and nested values for JSON output."""

    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(value)
    if isinstance(value, dict):
        return {key: normalize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    return asdict(value) if is_dataclass(value) else value
