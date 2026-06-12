from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .config import DEFAULT_SERVE_BIND_ADDRESS, DEFAULT_SERVE_PORT, DdserveConfig
from .copilot_hooks import session_start_context
from .embeddings.index import status_for_embeddings
from .errors import DdserveError
from .mcp_server import handle_mcp_request
from .models import to_jsonable
from .search import search_docs
from .server_shared import get_docset, get_page, get_page_content, list_docsets, list_pages


def serve(
    cache_root: str | Path, config: DdserveConfig, host: str | None = None, port: int | None = None
) -> None:
    """Implement serve."""
    bind = host or (
        config.serve.bind_address
        if config.serve and config.serve.bind_address
        else DEFAULT_SERVE_BIND_ADDRESS
    )
    bind_port = port or (
        config.serve.port if config.serve and config.serve.port else DEFAULT_SERVE_PORT
    )
    handler = make_handler(str(cache_root), config, bind)
    server = ThreadingHTTPServer((bind, bind_port), handler)
    print(f"ddserve listening on http://{bind}:{bind_port}")
    server.serve_forever()


def make_handler(cache_root: str, config: DdserveConfig, bind_host: str):
    """Implement make handler."""

    class DdserveHandler(BaseHTTPRequestHandler):
        """Represent DdserveHandler."""

        server_version = "ddserve/0.1.0"

        def do_GET(self) -> None:
            """Implement do GET."""
            self.handle_request("GET")

        def do_POST(self) -> None:
            """Implement do POST."""
            self.handle_request("POST")

        def do_OPTIONS(self) -> None:
            """Implement do OPTIONS."""
            self.send_response(HTTPStatus.NO_CONTENT)
            self.add_common_headers()
            self.end_headers()

        def handle_request(self, method: str) -> None:
            """Handle request."""
            try:
                if not self.host_allowed(bind_host):
                    self.write_json({"error": "Host header is not allowed"}, HTTPStatus.FORBIDDEN)
                    return
                parsed = urlparse(self.path)
                if parsed.path == "/health" and method == "GET":
                    self.write_json({"ok": True})
                    return
                if parsed.path.startswith("/api") or parsed.path in {
                    "/mcp",
                    "/copilot/hooks/sessionStart",
                }:
                    if not self.authorized():
                        self.write_json({"error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
                        return
                self.route(method, parsed.path, parse_qs(parsed.query))
            except DdserveError as exc:
                self.write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self.write_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def route(self, method: str, path: str, query: dict[str, list[str]]) -> None:
            """Implement route."""
            if path == "/api" and method == "GET":
                self.write_json(
                    {
                        "name": "ddserve",
                        "links": ["/api/docsets", "/api/search", "/api/embeddings/status"],
                    }
                )
                return
            if path == "/api/docsets" and method == "GET":
                self.write_json({"items": [to_jsonable(item) for item in list_docsets(cache_root)]})
                return
            if path == "/api/search":
                if method == "GET":
                    self.write_json(search_payload(query))
                    return
                body = self.read_json_body()
                self.write_json(search_payload_from_body(body))
                return
            if path == "/api/embeddings/status" and method == "GET":
                self.write_json(to_jsonable(status_for_embeddings(cache_root, config)))
                return
            if path.startswith("/api/embeddings/status/") and method == "GET":
                slug = unquote(path.rsplit("/", 1)[1])
                self.write_json(to_jsonable(status_for_embeddings(cache_root, config, slug)))
                return
            if path == "/copilot/hooks/sessionStart" and method == "POST":
                body = self.read_json_body()
                prompt = (
                    str(body.get("prompt") or body.get("message") or "")
                    if isinstance(body, dict)
                    else ""
                )
                self.write_json(session_start_context(cache_root, config, prompt, os.environ))
                return
            if path == "/mcp" and method == "POST":
                body = self.read_json_body()
                if not isinstance(body, dict):
                    self.write_json({"error": "Invalid MCP request"}, HTTPStatus.BAD_REQUEST)
                else:
                    self.write_json(handle_mcp_request(cache_root, config, body))
                return
            parts = [unquote(part) for part in path.strip("/").split("/")]
            if len(parts) >= 3 and parts[:2] == ["api", "docsets"]:
                slug = parts[2]
                if len(parts) == 3 and method == "GET":
                    self.write_json(to_jsonable(get_docset(cache_root, slug)))
                    return
                if len(parts) == 4 and parts[3] == "pages" and method == "GET":
                    self.write_json(
                        list_pages(
                            cache_root,
                            slug,
                            limit=int(first(query, "limit") or 100),
                            offset=int(first(query, "offset") or 0),
                            query=first(query, "q"),
                            type_=first(query, "type"),
                        )
                    )
                    return
                if len(parts) == 5 and parts[3] == "pages" and method == "GET":
                    self.write_json(to_jsonable(get_page(cache_root, slug, parts[4])))
                    return
                if (
                    len(parts) == 6
                    and parts[3] == "pages"
                    and parts[5] == "content"
                    and method == "GET"
                ):
                    self.write_json(
                        to_jsonable(
                            get_page_content(
                                cache_root,
                                slug,
                                parts[4],
                                optional_int(first(query, "startLine")),
                                optional_int(first(query, "endLine")),
                            )
                        )
                    )
                    return
            self.write_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def read_json_body(self) -> object:
            """Read json body."""
            length = int(self.headers.get("content-length") or 0)
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def write_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            """Write json."""
            body = json.dumps(normalize_json(value), indent=2).encode("utf-8")
            self.send_response(status)
            self.add_common_headers()
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def add_common_headers(self) -> None:
            """Implement add common headers."""
            if config.serve and config.serve.cors:
                origins = config.serve.cors.origins
                origin = self.headers.get("origin")
                if "*" in origins:
                    self.send_header("access-control-allow-origin", "*")
                elif origin in origins:
                    self.send_header("access-control-allow-origin", origin)
                self.send_header("access-control-allow-methods", "GET,POST,OPTIONS")
                self.send_header("access-control-allow-headers", "authorization,content-type")

        def authorized(self) -> bool:
            """Implement authorized."""
            if not config.serve or not config.serve.auth:
                return True
            expected = os.environ.get(config.serve.auth.token_env) or config.serve.auth.token
            if not expected:
                return False
            return self.headers.get("authorization") == f"Bearer {expected}"

        def host_allowed(self, bind: str) -> bool:
            """Implement host allowed."""
            if bind in {"0.0.0.0", "::"}:
                return True
            host = (self.headers.get("host") or "").split(":", 1)[0]
            return host in {bind, "localhost", "127.0.0.1", "::1"}

        def log_message(self, _format: str, *_args: object) -> None:
            """Implement log message."""
            return

    def search_payload(query: dict[str, list[str]]) -> dict[str, object]:
        """Implement search payload."""
        results = search_docs(
            cache_root,
            first(query, "q") or "",
            config,
            slugs=query.get("slug"),
            languages=query.get("language"),
            limit=int(first(query, "limit") or 10),
        )
        return {"matches": [to_jsonable(result) for result in results]}

    def search_payload_from_body(body: object) -> dict[str, object]:
        """Implement search payload from body."""
        if not isinstance(body, dict):
            raise DdserveError("Search body must be a JSON object")
        results = search_docs(
            cache_root,
            str(body.get("query") or ""),
            config,
            slugs=string_list(body.get("slugs")),
            languages=string_list(body.get("languages")),
            limit=int(body.get("limit") or 10),
        )
        return {"matches": [to_jsonable(result) for result in results]}

    return DdserveHandler


def first(query: dict[str, list[str]], key: str) -> str | None:
    """Implement first."""
    values = query.get(key)
    return values[0] if values else None


def optional_int(value: str | None) -> int | None:
    """Implement optional int."""
    return int(value) if value else None


def string_list(value: object) -> list[str] | None:
    """Implement string list."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def normalize_json(value: object) -> object:
    """Normalize json."""
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(value)
    if isinstance(value, dict):
        return {key: normalize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    return asdict(value) if is_dataclass(value) else value
