from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from smahtiepants.cache import migrate_legacy_cache_root, read_cache_manifest, resolve_cache_root
from smahtiepants.cli import run_cli
from smahtiepants.config import (
    SmahtiepantsConfig,
    EmbeddingsConfig,
    OpenAiConfig,
    load_config,
    redact_config,
)
from smahtiepants.devdocs import find_docset, normalize_docsets
from smahtiepants.embeddings.annoy_index import AnnoyIndexManager
from smahtiepants.embeddings.chunks import (
    ChunkedMarkdownPages,
    ChunkingStats,
    PreparedEmbeddingChunk,
    chunk_markdown_pages,
    split_markdown_into_chunks,
)
from smahtiepants.embeddings.index import (
    create_docset_embedding_vectors,
    create_embedding_vectors_async,
    rebuild_docset_embeddings,
)
from smahtiepants.embeddings.openai import EmbeddingBatchLimits, embedding_batch_ranges
from smahtiepants.embeddings.storage import open_embedding_storage
from smahtiepants.http import HttpClient
from smahtiepants.install import install_docset, remove_docset, update_docsets
from smahtiepants.search import search_docs
from smahtiepants.search.filters import resolve_docset_filters
from smahtiepants.search.terms import parse_keyword_terms
from smahtiepants.server_shared import get_page_content, list_pages
from smahtiepants.text import extract_html_section, normalize_link_href, render_markdown


class FakeHttp(HttpClient):
    """Represent FakeHttp."""

    def __init__(self, root: Path) -> None:
        """Implement init."""
        self.root = root

    def fetch_json(self, _url: str) -> object:
        """Implement fetch json."""
        return [
            {
                "name": "HTTP",
                "slug": "http",
                "type": "protocol",
                "mtime": 10,
                "alias": ["headers"],
            }
        ]

    def download_file(self, url: str, destination: str | Path):
        """Implement download file."""
        from smahtiepants.models import DownloadedFile

        target = Path(destination)
        if url.endswith("/index.json"):
            payload = {"entries": [{"name": "Headers", "path": "headers", "type": "Guide"}]}
        else:
            payload = {"headers": "<h1>Headers</h1><p>Request headers document metadata.</p>"}
        target.write_text(json.dumps(payload), encoding="utf-8")
        data = target.read_bytes()
        import hashlib

        return DownloadedFile(
            path=str(target), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )


class MultiFakeHttp(HttpClient):
    """Represent MultiFakeHttp."""

    def fetch_json(self, _url: str) -> object:
        """Implement fetch json."""
        return [
            {"name": "HTTP", "slug": "http", "type": "protocol", "mtime": 10},
            {"name": "CSS", "slug": "css", "type": "style", "mtime": 10},
        ]

    def download_file(self, url: str, destination: str | Path):
        """Implement download file."""
        from smahtiepants.models import DownloadedFile

        target = Path(destination)
        slug = url.rstrip("/").split("/")[-2]
        if url.endswith("/index.json"):
            payload = {"entries": [{"name": slug.upper(), "path": slug, "type": "Guide"}]}
        else:
            payload = {slug: f"<h1>{slug.upper()}</h1><p>{slug} request metadata.</p>"}
        target.write_text(json.dumps(payload), encoding="utf-8")
        data = target.read_bytes()
        import hashlib

        return DownloadedFile(
            path=str(target), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )


class NodeFakeHttp(HttpClient):
    """Represent NodeFakeHttp."""

    def fetch_json(self, _url: str) -> object:
        """Implement fetch json."""
        return [{"name": "Node.js", "slug": "node", "type": "node", "mtime": 10}]

    def download_file(self, url: str, destination: str | Path):
        """Implement download file."""
        from smahtiepants.models import DownloadedFile

        target = Path(destination)
        slug = url.rstrip("/").split("/")[-2]
        if url.endswith("/index.json"):
            payload = {"entries": [{"name": "Node", "path": "node", "type": "Guide"}]}
        else:
            payload = {slug: "<h1>Node</h1><p>Node.js runtime documentation.</p>"}
        target.write_text(json.dumps(payload), encoding="utf-8")
        data = target.read_bytes()
        import hashlib

        return DownloadedFile(
            path=str(target), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )


class PythonFakeHttp(HttpClient):
    """Represent PythonFakeHttp."""

    def fetch_json(self, _url: str) -> object:
        """Implement fetch json."""
        return [
            {
                "name": "Python",
                "slug": "python~3.13",
                "type": "python",
                "version": "3.13",
                "mtime": 10,
                "alias": "py",
            },
            {
                "name": "Python",
                "slug": "python~3.14",
                "type": "python",
                "version": "3.14",
                "mtime": 20,
                "alias": "py",
            },
        ]

    def download_file(self, url: str, destination: str | Path):
        """Implement download file."""
        from smahtiepants.models import DownloadedFile

        target = Path(destination)
        slug = url.rstrip("/").split("/")[-2]
        if url.endswith("/index.json"):
            payload = {"entries": [{"name": "Lists", "path": "lists", "type": "Guide"}]}
        else:
            payload = {"lists": f"<h1>Lists</h1><p>{slug} list documentation.</p>"}
        target.write_text(json.dumps(payload), encoding="utf-8")
        data = target.read_bytes()
        import hashlib

        return DownloadedFile(
            path=str(target), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )


def test_smahtiepants_config_parses_and_redacts_secrets() -> None:
    """Validate smahtiepants config parses shared smahtiepants settings and redacts secrets."""
    from wickedsmaht_config import (
        Config,
        SmahtiepantsAuthSettings,
        SmahtiepantsCorsSettings,
        SmahtiepantsServeSettings,
        SmahtiepantsSettings,
    )

    from smahtiepants.config import from_shared_config

    config = from_shared_config(
        Config(
            base_url="http://127.0.0.1:11434/v1",
            text_embedding_model="embed",
            smahtiepants=SmahtiepantsSettings(
                api_key="secret",
                serve=SmahtiepantsServeSettings(
                    auth=SmahtiepantsAuthSettings(token="token"),
                    cors=SmahtiepantsCorsSettings(origins=["http://localhost:3000"]),
                ),
            ),
        )
    )

    redacted = redact_config(config)

    assert config.embeddings.enabled is True
    assert config.openai and config.openai.embedding_model == "embed"
    assert redacted["openai"]["apiKey"] == "[redacted]"
    assert redacted["serve"]["auth"]["token"] == "[redacted]"


def test_smahtiepants_config_accepts_legacy_ddserve_settings() -> None:
    """Validate legacy shared ddserve settings still feed smahtiepants config."""
    from wickedsmaht_config import Config, DdserveEmbeddingSettings, DdserveSettings

    from smahtiepants.config import from_shared_config

    config = from_shared_config(
        Config(
            base_url="http://127.0.0.1:11434/v1",
            text_embedding_model="embed",
            ddserve=DdserveSettings(
                api_key_env="DOCS_API_KEY",
                embeddings=DdserveEmbeddingSettings(batch_size=7),
            ),
        )
    )

    assert config.openai and config.openai.api_key_env == "DOCS_API_KEY"
    assert config.embeddings.batch_size == 7


def test_smahtiepants_config_loads_wickedsmaht_text_embedding(tmp_path: Path) -> None:
    """Validate smahtiepants config loads shared text embedding settings."""
    home = tmp_path / "home"
    config_dir = home / ".wickedsmaht"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "base-url": "http://127.0.0.1:14892/v1",
                "text-embedding-model": "text-embed",
                "coding-embedding-model": "code-embed",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(env={"HOME": str(home)})

    assert loaded.found is True
    assert loaded.config.embeddings.enabled is True
    assert loaded.config.openai is not None
    assert loaded.config.openai.base_url == "http://127.0.0.1:14892/v1"
    assert loaded.config.openai.embedding_model == "text-embed"


def test_smahtiepants_config_migrates_legacy_ddserve_key(tmp_path: Path) -> None:
    """Validate loading config rewrites legacy ddserve settings to smahtiepants."""
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "base-url": "http://127.0.0.1:14892/v1",
                "text-embedding-model": "text-embed",
                "ddserve": {"api-key-env": "DOCS_API_KEY"},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(str(path))
    migrated = json.loads(path.read_text(encoding="utf-8"))

    assert loaded.config.openai and loaded.config.openai.api_key_env == "DOCS_API_KEY"
    assert "smahtiepants" in migrated
    assert "ddserve" not in migrated


def test_smahtiepants_config_does_not_use_coding_embedding_model(tmp_path: Path) -> None:
    """Validate smahtiepants config does not use coding embedding settings."""
    home = tmp_path / "home"
    config_dir = home / ".wickedsmaht"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "base-url": "http://127.0.0.1:14892/v1",
                "coding-embedding-model": "code-embed",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(env={"HOME": str(home)})

    assert loaded.config.embeddings.enabled is False
    assert loaded.config.openai is None


def test_smahtiepants_cache_root_prefers_env(tmp_path: Path) -> None:
    """Validate smahtiepants cache root prefers env."""
    assert (
        resolve_cache_root({"SMAHTIEPANTS_CACHE_DIR": str(tmp_path / "cache")})
        == tmp_path / "cache"
    )
    assert (
        resolve_cache_root({"DDSERVE_CACHE_DIR": str(tmp_path / "legacy")}) == tmp_path / "legacy"
    )
    assert (
        resolve_cache_root({"XDG_CACHE_HOME": str(tmp_path / "xdg")})
        == tmp_path / "xdg" / "smahtiepants"
    )


def test_smahtiepants_cache_root_migrates_legacy_default_cache(tmp_path: Path) -> None:
    """Validate default ddserve cache is moved to the smahtiepants cache path."""
    legacy = tmp_path / ".cache" / "ddserve"
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text("{}", encoding="utf-8")

    migrated = resolve_cache_root({"HOME": str(tmp_path)})

    assert migrated == tmp_path / ".cache" / "smahtiepants"
    assert (migrated / "manifest.json").is_file()
    assert not legacy.exists()


def test_legacy_cache_migration_tolerates_concurrent_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate cache migration succeeds if another process already moved the legacy path."""
    root = tmp_path / ".cache" / "smahtiepants"
    legacy = tmp_path / ".cache" / "ddserve"
    legacy.mkdir(parents=True)

    def concurrent_replace(_self: Path, target: Path) -> None:
        legacy.rmdir()
        target.mkdir()
        raise FileNotFoundError(str(legacy))

    monkeypatch.setattr(Path, "replace", concurrent_replace)

    assert migrate_legacy_cache_root(root, legacy) == root


def test_devdocs_normalizes_docsets() -> None:
    """Validate devdocs normalizes docsets."""
    docsets = normalize_docsets(
        [
            {"name": "B", "slug": "b"},
            {"name": "", "slug": "skip"},
            {"name": "A", "slug": "a", "db_size": 123, "alias": "alpha"},
        ]
    )

    assert [docset.slug for docset in docsets] == ["a", "b"]
    assert docsets[0].aliases == ["alpha"]
    assert docsets[0].db_size == 123


def test_devdocs_resolves_aliases_and_curated_shorthands() -> None:
    """Validate DevDocs identifiers resolve through aliases and curated shorthands."""
    docsets = normalize_docsets(
        [
            {
                "name": "Python",
                "slug": "python~3.13",
                "type": "python",
                "version": "3.13",
                "alias": "py",
            },
            {
                "name": "Python",
                "slug": "python~3.14",
                "type": "python",
                "version": "3.14",
                "alias": "py",
            },
            {"name": "TypeScript", "slug": "typescript", "alias": "ts"},
            {"name": "Node.js", "slug": "node"},
        ]
    )

    assert find_docset(docsets, "ts").slug == "typescript"
    assert find_docset(docsets, "py").slug == "python~3.14"
    assert find_docset(docsets, "python").slug == "python~3.14"
    assert find_docset(docsets, "python~3.13").slug == "python~3.13"
    assert find_docset(docsets, "nodejs").slug == "node"


def test_text_extracts_anchor_sections_and_normalizes_links() -> None:
    """Validate text extracts anchor sections and normalizes links."""
    html = '<h2 id="one">One</h2><p>First</p><h2 id="two">Two</h2><p>Second</p>'

    assert "Second" not in extract_html_section(html, "page#one")
    assert normalize_link_href("#part", "guide/start") == "guide/start#part"
    assert normalize_link_href("../other", "guide/start") == "other"


def test_render_markdown_keeps_generated_header_and_code_block() -> None:
    """Validate render markdown keeps generated header and code block."""
    markdown = render_markdown(
        "Example", "example", "<h1>Example</h1><pre>if ok:\n    return x</pre>"
    )

    assert markdown.startswith("# Example\n\n> DevDocs path: example")
    assert "```" in markdown
    assert "    return x" in markdown


def test_render_markdown_preserves_text_after_images() -> None:
    """Validate render markdown preserves text after images."""
    markdown = render_markdown("Example", "example", "<p>Before</p><img src='x.png'><p>After</p>")

    assert "Before" in markdown
    assert "After" in markdown


def test_markdown_chunking_uses_overlap_and_sentence_boundaries() -> None:
    """Validate markdown chunking uses overlap and sentence boundaries."""
    chunks = split_markdown_into_chunks("One sentence. Two sentence.\n\nThree sentence.", 24, 5)

    assert len(chunks) >= 2
    assert chunks[0].startswith("One sentence.")


def test_markdown_chunking_bounds_chunks_per_page(tmp_path: Path) -> None:
    """Validate chunking caps pathological page chunk counts."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )
    from smahtiepants.cache import read_docset_manifest

    docset = read_docset_manifest(tmp_path, "http")
    assert docset is not None
    page = docset.pages[0]
    page_path = tmp_path / "docs" / "http" / page.file
    page_path.write_text(" ".join(f"Sentence {index}." for index in range(80)), encoding="utf-8")

    chunked = chunk_markdown_pages(
        docset,
        tmp_path,
        max_chunk_chars=40,
        overlap_chars=0,
        min_chunk_chars=1,
        max_chunks_per_page=2,
    )

    assert len(chunked.chunks) == 2
    assert chunked.stats.truncated_pages == 1
    assert chunked.stats.truncated_chunks > 0


def test_embedding_batch_ranges_respect_request_bytes() -> None:
    """Validate embedding batches are bounded by request bytes as well as count."""
    ranges = embedding_batch_ranges(
        ["aa", "bbb", "cccc", "d"],
        EmbeddingBatchLimits(max_inputs=10, max_request_bytes=6),
    )

    assert ranges == [(0, 2), (2, 4)]


def test_async_embedding_batches_preserve_order_and_limit_concurrency() -> None:
    """Validate async embedding batches are bounded and ordered."""

    class TrackingAsyncClient:
        """Represent TrackingAsyncClient."""

        def __init__(self) -> None:
            """Implement init."""
            self.active = 0
            self.max_active = 0

        async def create_embeddings(self, input_):
            """Implement create embeddings."""
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return [[float(value), 0.0] for value in input_]

    async def run() -> TrackingAsyncClient:
        client = TrackingAsyncClient()
        vectors = await create_embedding_vectors_async(
            ["1", "2", "3", "4", "5"],
            EmbeddingBatchLimits(max_inputs=1, max_request_bytes=16),
            client,
            max_concurrent_requests=2,
        )
        assert vectors == [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
        return client

    client = asyncio.run(run())
    assert client.max_active == 2


def test_created_async_embedding_client_closes_before_event_loop_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate internally created async embedding clients close in their event loop."""
    events: list[tuple[str, int, bool]] = []

    class ManagedAsyncClient:
        """Represent ManagedAsyncClient."""

        async def create_embeddings(self, input_):
            """Implement create embeddings."""
            loop = asyncio.get_running_loop()
            events.append(("create", id(loop), loop.is_closed()))
            values = [input_] if isinstance(input_, str) else list(input_)
            return [[float(len(value)), 0.0] for value in values]

        async def aclose(self) -> None:
            """Implement aclose."""
            loop = asyncio.get_running_loop()
            events.append(("close", id(loop), loop.is_closed()))

    def create_client(_config, _env):
        loop = asyncio.get_running_loop()
        events.append(("factory", id(loop), loop.is_closed()))
        return ManagedAsyncClient()

    monkeypatch.setattr(
        "smahtiepants.embeddings.index.create_openai_async_embedding_client", create_client
    )

    chunked = ChunkedMarkdownPages(
        docset={"slug": "http"},
        chunks=[
            PreparedEmbeddingChunk(
                page={"slug": "http", "path": "alpha"},
                ordinal=0,
                content_hash="hash-alpha",
                source_hash="source-alpha",
                text="alpha",
                token_count=1,
                metadata_json="{}",
            ),
            PreparedEmbeddingChunk(
                page={"slug": "http", "path": "beta"},
                ordinal=1,
                content_hash="hash-beta",
                source_hash="source-beta",
                text="beta",
                token_count=1,
                metadata_json="{}",
            ),
        ],
        stats=ChunkingStats(pages=1),
    )

    vectors = create_docset_embedding_vectors(
        chunked,
        SmahtiepantsConfig(
            embeddings=EmbeddingsConfig(enabled=True),
            openai=OpenAiConfig(embedding_model="embed"),
        ),
        env={},
        client=None,
        async_client=None,
    )

    assert vectors == [[5.0, 0.0], [4.0, 0.0]]
    assert [event[0] for event in events] == ["factory", "create", "close"]
    assert len({event[1] for event in events}) == 1
    assert all(not event[2] for event in events)


def test_install_docset_writes_manifests_and_pages(tmp_path: Path) -> None:
    """Validate install docset writes manifests and pages."""
    result = install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )

    manifest = read_cache_manifest(tmp_path)
    pages = list_pages(tmp_path, "http")
    page = get_page_content(tmp_path, "http", pages["items"][0]["id"])

    assert result.status == "installed"
    assert manifest.docs["http"].page_count == 1
    assert "Request headers" in page.content


def test_install_docset_accepts_upstream_alias(tmp_path: Path) -> None:
    """Validate docs install resolves upstream aliases to canonical slugs."""
    result = install_docset(
        "headers",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )

    manifest = read_cache_manifest(tmp_path)

    assert result.slug == "http"
    assert "http" in manifest.docs
    assert not (tmp_path / "docs" / "headers").exists()


def test_install_docset_accepts_curated_alias(tmp_path: Path) -> None:
    """Validate docs install resolves curated shorthands to canonical slugs."""
    result = install_docset(
        "nodejs",
        str(tmp_path),
        http=NodeFakeHttp(),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )

    manifest = read_cache_manifest(tmp_path)

    assert result.slug == "node"
    assert "node" in manifest.docs
    assert not (tmp_path / "docs" / "nodejs").exists()


def test_install_docset_accepts_type_as_slug(tmp_path: Path) -> None:
    """Validate docs install resolves language-like slugs to versioned docsets."""
    result = install_docset(
        "python",
        str(tmp_path),
        http=PythonFakeHttp(),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )

    pages = list_pages(tmp_path, "python")
    content = get_page_content(tmp_path, "python", pages["items"][0]["id"])

    assert result.slug == "python~3.14"
    assert pages["slug"] == "python~3.14"
    assert "python~3.14 list documentation" in content.content


def test_installed_docset_helpers_accept_aliases(tmp_path: Path) -> None:
    """Validate installed docset helpers resolve stored aliases."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )

    pages = list_pages(tmp_path, "headers")
    content = get_page_content(tmp_path, "headers", pages["items"][0]["id"])

    assert pages["slug"] == "http"
    assert "Request headers" in content.content


def test_remove_docset_accepts_aliases(tmp_path: Path) -> None:
    """Validate docs remove resolves stored aliases."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )

    result = remove_docset("headers", str(tmp_path))

    assert result.slug == "http"
    assert "http" not in read_cache_manifest(tmp_path).docs


def test_docs_available_marks_installed_docsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validate docs available marks installed docsets."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )
    monkeypatch.setenv("SMAHTIEPANTS_CACHE_DIR", str(tmp_path))

    run_cli(["docs", "available", "--offline"])

    output = capsys.readouterr().out
    assert "*  http" in output


def test_docset_filters_use_stored_aliases(tmp_path: Path) -> None:
    """Validate docset filters use stored aliases."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )

    assert resolve_docset_filters(tmp_path, languages=["headers"]) == {"http"}
    assert resolve_docset_filters(tmp_path, slugs=["headers"]) == {"http"}


def test_docset_slug_filters_match_language_metadata(tmp_path: Path) -> None:
    """Validate slug filters accept language-like docset identifiers."""
    install_docset(
        "python",
        str(tmp_path),
        http=PythonFakeHttp(),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )

    assert resolve_docset_filters(tmp_path, slugs=["python"]) == {"python~3.14"}
    assert resolve_docset_filters(tmp_path, languages=["python"]) == {"python~3.14"}


def test_docs_update_prints_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validate docs update prints progress."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )
    monkeypatch.setenv("SMAHTIEPANTS_CACHE_DIR", str(tmp_path))

    run_cli(["docs", "update", "--offline"])

    captured = capsys.readouterr()
    assert "Updating http (1/1)..." in captured.err
    assert "Finished http: skipped" in captured.err


def test_keyword_search_falls_back_to_indexed_chunks(tmp_path: Path) -> None:
    """Validate keyword search falls back to indexed chunks."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )
    from smahtiepants.cache import read_docset_manifest
    from smahtiepants.embeddings.chunks import chunk_markdown_pages

    docset = read_docset_manifest(tmp_path, "http")
    assert docset is not None
    chunked = chunk_markdown_pages(docset, tmp_path)
    storage = open_embedding_storage(tmp_path)
    try:
        storage.replace_docset_chunks(chunked.docset, chunked.chunks, None, None)
    finally:
        storage.close()

    results = search_docs(
        tmp_path,
        "request headers",
        SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False), openai=None),
    )

    assert results[0].docset_slug == "http"
    assert results[0].match_kind == "keyword"


def test_keyword_search_sanitizes_fts_query_terms(tmp_path: Path) -> None:
    """Validate keyword search sanitizes FTS query terms."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )
    from smahtiepants.cache import read_docset_manifest
    from smahtiepants.embeddings.chunks import chunk_markdown_pages

    docset = read_docset_manifest(tmp_path, "http")
    assert docset is not None
    chunked = chunk_markdown_pages(docset, tmp_path)
    storage = open_embedding_storage(tmp_path)
    try:
        storage.replace_docset_chunks(chunked.docset, chunked.chunks, None, None)
    finally:
        storage.close()

    results = search_docs(
        tmp_path,
        'request+headers "metadata"',
        SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False), openai=None),
    )

    assert results[0].docset_slug == "http"


def test_keyword_terms_ignore_common_stopwords() -> None:
    """Validate keyword terms skip trivial query words."""

    assert parse_keyword_terms("sorting a list") == ["sorting", "list"]


def test_search_with_unknown_slug_returns_no_results(tmp_path: Path) -> None:
    """Validate search with unknown slug returns no results."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )
    from smahtiepants.cache import read_docset_manifest
    from smahtiepants.embeddings.chunks import chunk_markdown_pages

    docset = read_docset_manifest(tmp_path, "http")
    assert docset is not None
    chunked = chunk_markdown_pages(docset, tmp_path)
    storage = open_embedding_storage(tmp_path)
    try:
        storage.replace_docset_chunks(chunked.docset, chunked.chunks, None, None)
    finally:
        storage.close()

    results = search_docs(
        tmp_path,
        "request headers",
        SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False), openai=None),
        slugs=["missing"],
    )

    assert results == []


def test_server_shared_lists_pages_and_line_ranges(tmp_path: Path) -> None:
    """Validate server shared lists pages and line ranges."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )

    pages = list_pages(tmp_path, "http", query="headers")
    content = get_page_content(tmp_path, "http", pages["items"][0]["id"], start_line=1, end_line=2)

    assert pages["total"] == 1
    assert content.total_lines >= 2


def test_semantic_search_uses_configured_embedding_client(tmp_path: Path) -> None:
    """Validate semantic search uses configured embedding client."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )
    from smahtiepants.cache import read_docset_manifest
    from smahtiepants.embeddings.chunks import chunk_markdown_pages

    docset = read_docset_manifest(tmp_path, "http")
    assert docset is not None
    chunked = chunk_markdown_pages(docset, tmp_path)
    storage = open_embedding_storage(tmp_path)
    try:
        storage.replace_docset_chunks(chunked.docset, chunked.chunks, [[1.0, 0.0]], "embed")
    finally:
        storage.close()

    class FakeEmbeddingClient:
        """Represent FakeEmbeddingClient."""

        def create_embeddings(self, _input):
            """Implement create embeddings."""
            return [[1.0, 0.0]]

    results = search_docs(
        tmp_path,
        "headers",
        SmahtiepantsConfig(
            embeddings=EmbeddingsConfig(enabled=True),
            openai=OpenAiConfig(embedding_model="embed"),
        ),
        client=FakeEmbeddingClient(),
    )

    assert results[0].match_kind == "hybrid"
    assert results[0].score == pytest.approx(1.0)
    storage = open_embedding_storage(tmp_path)
    try:
        assert storage.annoy_index_metadata("embed") is not None
    finally:
        storage.close()


def test_search_keeps_semantic_ranking_when_keyword_hits_disagree(tmp_path: Path) -> None:
    """Validate keyword hits do not outrank stronger semantic matches."""

    install_docset(
        "http",
        str(tmp_path),
        http=MultiFakeHttp(),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )
    install_docset(
        "css",
        str(tmp_path),
        http=MultiFakeHttp(),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )
    from smahtiepants.cache import read_docset_manifest
    from smahtiepants.embeddings.chunks import chunk_markdown_pages

    http_docset = read_docset_manifest(tmp_path, "http")
    css_docset = read_docset_manifest(tmp_path, "css")
    assert http_docset is not None
    assert css_docset is not None

    http_page = tmp_path / "docs" / "http" / http_docset.pages[0].file
    css_page = tmp_path / "docs" / "css" / css_docset.pages[0].file
    http_page.write_text("# Arrays\n\nShell arrays use indexed containers.", encoding="utf-8")
    css_page.write_text("# Sorting\n\nSorting a list is covered here.", encoding="utf-8")

    http_chunks = chunk_markdown_pages(http_docset, tmp_path)
    css_chunks = chunk_markdown_pages(css_docset, tmp_path)
    storage = open_embedding_storage(tmp_path)
    try:
        storage.replace_docset_chunks(http_chunks.docset, http_chunks.chunks, [[1.0, 0.0]], "embed")
        storage.replace_docset_chunks(css_chunks.docset, css_chunks.chunks, [[0.0, 1.0]], "embed")
    finally:
        storage.close()

    class FakeEmbeddingClient:
        """Represent FakeEmbeddingClient."""

        def create_embeddings(self, _input):
            """Implement create embeddings."""
            return [[1.0, 0.0]]

    results = search_docs(
        tmp_path,
        "sorting a list",
        SmahtiepantsConfig(
            embeddings=EmbeddingsConfig(enabled=True),
            openai=OpenAiConfig(embedding_model="embed"),
        ),
        client=FakeEmbeddingClient(),
    )

    assert results[0].docset_slug == "http"
    assert results[0].match_kind == "semantic"
    css_result = next(result for result in results if result.docset_slug == "css")
    assert css_result.match_kind == "hybrid"


def test_smahtiepants_annoy_index_rebuilds_when_embeddings_change(tmp_path: Path) -> None:
    """Validate smahtiepants annoy index rebuilds when embeddings change."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )
    from smahtiepants.cache import read_docset_manifest
    from smahtiepants.embeddings.chunks import chunk_markdown_pages

    docset = read_docset_manifest(tmp_path, "http")
    assert docset is not None
    chunked = chunk_markdown_pages(docset, tmp_path)
    storage = open_embedding_storage(tmp_path)
    try:
        storage.replace_docset_chunks(chunked.docset, chunked.chunks, [[1.0, 0.0]], "embed")
        manager = AnnoyIndexManager(tmp_path, storage, trees=2)
        first_ids = manager.search("embed", [1.0, 0.0], 1)
        first_metadata = storage.annoy_index_metadata("embed")

        storage.replace_docset_chunks(chunked.docset, chunked.chunks, [[0.0, 1.0]], "embed")
        second_ids = manager.search("embed", [0.0, 1.0], 1)
        second_metadata = storage.annoy_index_metadata("embed")

        assert first_ids
        assert second_ids
        assert first_metadata is not None
        assert second_metadata is not None
        assert first_metadata["source_version"] != second_metadata["source_version"]
        assert (tmp_path / "embeddings" / "annoy").is_dir()
    finally:
        storage.close()


def test_update_skips_current_embeddings_and_preserves_annoy(tmp_path: Path) -> None:
    """Validate update skips current embeddings and preserves annoy."""
    config = SmahtiepantsConfig(
        embeddings=EmbeddingsConfig(enabled=True),
        openai=OpenAiConfig(embedding_model="embed"),
    )

    class CountingEmbeddingClient:
        """Represent CountingEmbeddingClient."""

        def __init__(self) -> None:
            """Implement init."""
            self.calls = 0

        def create_embeddings(self, input_):
            """Implement create embeddings."""
            self.calls += 1
            if isinstance(input_, str):
                return [[1.0, 0.0]]
            return [[1.0, 0.0] for _item in input_]

    class FailingEmbeddingClient:
        """Represent FailingEmbeddingClient."""

        def create_embeddings(self, _input):
            """Implement create embeddings."""
            raise AssertionError("current embeddings should not be rebuilt")

    install_client = CountingEmbeddingClient()
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=config,
        embedding_client=install_client,
    )
    storage = open_embedding_storage(tmp_path)
    try:
        manager = AnnoyIndexManager(tmp_path, storage, trees=2)
        assert manager.search("embed", [1.0, 0.0], 1)
        metadata_before = storage.annoy_index_metadata("embed")
    finally:
        storage.close()

    results = update_docsets(
        "http",
        str(tmp_path),
        offline=True,
        config=config,
        embedding_client=FailingEmbeddingClient(),
    )

    storage = open_embedding_storage(tmp_path)
    try:
        metadata_after = storage.annoy_index_metadata("embed")
    finally:
        storage.close()

    assert install_client.calls == 1
    assert results[0].status == "skipped"
    assert metadata_before == metadata_after


def test_update_builds_missing_annoy_for_current_embeddings(tmp_path: Path) -> None:
    """Validate update builds missing annoy for current embeddings."""
    config = SmahtiepantsConfig(
        embeddings=EmbeddingsConfig(enabled=True),
        openai=OpenAiConfig(embedding_model="embed"),
    )

    class FakeEmbeddingClient:
        """Represent FakeEmbeddingClient."""

        def create_embeddings(self, input_):
            """Implement create embeddings."""
            if isinstance(input_, str):
                return [[1.0, 0.0]]
            return [[1.0, 0.0] for _item in input_]

    class FailingEmbeddingClient:
        """Represent FailingEmbeddingClient."""

        def create_embeddings(self, _input):
            """Implement create embeddings."""
            raise AssertionError("current embeddings should not be rebuilt")

    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=config,
        embedding_client=FakeEmbeddingClient(),
    )
    storage = open_embedding_storage(tmp_path)
    try:
        metadata = storage.annoy_index_metadata("embed")
        assert metadata is not None
        Path(str(metadata["path"])).unlink()
        storage.conn.execute("DELETE FROM annoy_indexes WHERE model = ?", ("embed",))
        storage.conn.execute("DELETE FROM annoy_items WHERE model = ?", ("embed",))
        storage.conn.commit()
    finally:
        storage.close()

    results = update_docsets(
        "http",
        str(tmp_path),
        offline=True,
        config=config,
        embedding_client=FailingEmbeddingClient(),
    )

    storage = open_embedding_storage(tmp_path)
    try:
        rebuilt = storage.annoy_index_metadata("embed")
    finally:
        storage.close()

    assert results[0].status == "skipped"
    assert rebuilt is not None
    assert Path(str(rebuilt["path"])).is_file()


def test_update_migrates_legacy_embeddings_without_content_hash(tmp_path: Path) -> None:
    """Validate update migrates legacy embeddings without content hash."""
    config = SmahtiepantsConfig(
        embeddings=EmbeddingsConfig(enabled=True),
        openai=OpenAiConfig(embedding_model="embed"),
    )

    class FakeEmbeddingClient:
        """Represent FakeEmbeddingClient."""

        def create_embeddings(self, input_):
            """Implement create embeddings."""
            if isinstance(input_, str):
                return [[1.0, 0.0]]
            return [[1.0, 0.0] for _item in input_]

    class FailingEmbeddingClient:
        """Represent FailingEmbeddingClient."""

        def create_embeddings(self, _input):
            """Implement create embeddings."""
            raise AssertionError("current embeddings should not be rebuilt")

    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=config,
        embedding_client=FakeEmbeddingClient(),
    )
    database = tmp_path / "embeddings" / "embeddings.sqlite"
    conn = sqlite3.connect(database)
    try:
        conn.executescript(
            """
            ALTER TABLE embeddings RENAME TO embeddings_current;
            CREATE TABLE embeddings (
                chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (chunk_id, model)
            );
            INSERT INTO embeddings(chunk_id, model, dimensions, vector)
            SELECT chunk_id, model, dimensions, vector FROM embeddings_current;
            DROP TABLE embeddings_current;
            """
        )
        conn.commit()
    finally:
        conn.close()

    results = update_docsets(
        "http",
        str(tmp_path),
        offline=True,
        config=config,
        embedding_client=FailingEmbeddingClient(),
    )

    storage = open_embedding_storage(tmp_path)
    try:
        assert "content_hash" in storage.table_columns("embeddings")
        assert storage.annoy_index_metadata("embed") is not None
    finally:
        storage.close()
    assert results[0].status == "skipped"


def test_rebuild_embeddings_forces_current_vectors(tmp_path: Path) -> None:
    """Validate rebuild embeddings forces current vectors."""
    config = SmahtiepantsConfig(
        embeddings=EmbeddingsConfig(enabled=True),
        openai=OpenAiConfig(embedding_model="embed"),
    )

    class CountingEmbeddingClient:
        """Represent CountingEmbeddingClient."""

        def __init__(self) -> None:
            """Implement init."""
            self.calls = 0

        def create_embeddings(self, input_):
            """Implement create embeddings."""
            self.calls += 1
            if isinstance(input_, str):
                return [[1.0, 0.0]]
            return [[1.0, 0.0] for _item in input_]

    client = CountingEmbeddingClient()
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=config,
        embedding_client=client,
    )
    from smahtiepants.cache import read_docset_manifest

    docset = read_docset_manifest(tmp_path, "http")
    assert docset is not None

    result = rebuild_docset_embeddings(tmp_path, docset, config, client=client)

    assert client.calls == 2
    assert result["embedded"] == 1


def test_embedding_storage_uses_wal_and_busy_timeout(tmp_path: Path) -> None:
    """Validate smahtiepants SQLite storage is configured for concurrent readers."""
    storage = open_embedding_storage(tmp_path)
    try:
        journal_mode = storage.conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = storage.conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        storage.close()

    assert journal_mode == "wal"
    assert busy_timeout == 30000


def test_embedding_storage_rebuilds_chunk_fts(tmp_path: Path) -> None:
    """Validate chunk FTS can be rebuilt from authoritative chunks."""
    install_docset(
        "http",
        str(tmp_path),
        http=FakeHttp(tmp_path),
        config=SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False)),
    )
    from smahtiepants.cache import read_docset_manifest
    from smahtiepants.embeddings.chunks import chunk_markdown_pages

    docset = read_docset_manifest(tmp_path, "http")
    assert docset is not None
    chunked = chunk_markdown_pages(docset, tmp_path)
    storage = open_embedding_storage(tmp_path)
    try:
        storage.replace_docset_chunks(chunked.docset, chunked.chunks, None, None)
        storage.conn.execute("DELETE FROM chunk_fts")
        storage.conn.commit()

        assert storage.keyword_chunks(["request"]) == []

        storage.rebuild_chunk_fts()

        assert storage.keyword_chunks(["request"])
    finally:
        storage.close()


def test_embedding_storage_resets_legacy_schema(tmp_path: Path) -> None:
    """Validate incompatible legacy embedding schemas are reset."""
    database = tmp_path / "embeddings" / "embeddings.sqlite"
    database.parent.mkdir(parents=True)
    conn = sqlite3.connect(database)
    try:
        conn.executescript(
            """
            CREATE TABLE docsets (
                slug TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE pages (
                docset_slug TEXT NOT NULL,
                page_id TEXT NOT NULL,
                page_title TEXT NOT NULL,
                page_name TEXT NOT NULL,
                page_path TEXT NOT NULL,
                PRIMARY KEY (docset_slug, page_id)
            );
            CREATE TABLE embeddings (
                chunk_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_encoding TEXT NOT NULL,
                vector BLOB NOT NULL,
                vector_hash TEXT NOT NULL,
                PRIMARY KEY (chunk_id, model, dimensions)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    storage = open_embedding_storage(tmp_path)
    try:
        assert {"id", "title", "name", "path"}.issubset(storage.table_columns("pages"))
        assert "page_id" not in storage.table_columns("pages")
        assert "vector_encoding" not in storage.table_columns("embeddings")
    finally:
        storage.close()


def test_multi_docset_update_defers_annoy_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate docs update rebuilds Annoy once for a multi-docset run."""
    http = MultiFakeHttp()
    disabled = SmahtiepantsConfig(embeddings=EmbeddingsConfig(enabled=False))
    install_docset("http", str(tmp_path), http=http, config=disabled)
    install_docset("css", str(tmp_path), http=http, config=disabled)

    class FakeEmbeddingClient:
        """Represent FakeEmbeddingClient."""

        def create_embeddings(self, input_):
            """Implement create embeddings."""
            if isinstance(input_, str):
                return [[1.0, 0.0]]
            return [[1.0, 0.0] for _item in input_]

    class CountingAnnoyIndexManager:
        """Represent CountingAnnoyIndexManager."""

        calls = 0

        def __init__(self, *_args, **_kwargs) -> None:
            """Implement init."""

        def ensure(self, _model: str) -> bool:
            """Implement ensure."""
            type(self).calls += 1
            return True

    monkeypatch.setattr("smahtiepants.install.AnnoyIndexManager", CountingAnnoyIndexManager)

    results = update_docsets(
        None,
        str(tmp_path),
        offline=True,
        config=SmahtiepantsConfig(
            embeddings=EmbeddingsConfig(enabled=True),
            openai=OpenAiConfig(embedding_model="embed"),
        ),
        embedding_client=FakeEmbeddingClient(),
    )

    assert CountingAnnoyIndexManager.calls == 1
    assert [result.slug for result in results] == ["css", "http"]
    assert all(result.annoy_indexed for result in results)
