from __future__ import annotations

import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ddserve.cache import ensure_embedding_db_path

SQLITE_BUSY_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class SearchChunk:
    """Represent SearchChunk."""

    id: int
    docset_slug: str
    docset_name: str
    page_id: str
    page_title: str
    page_path: str
    page_type: str | None
    page_file: str
    ordinal: int
    text: str
    metadata_json: str
    vector: list[float] | None = None


@dataclass(frozen=True)
class EmbeddingRow:
    """Represent EmbeddingRow."""

    chunk_id: int
    vector: list[float]


class EmbeddingStorage:
    """Represent EmbeddingStorage."""

    def __init__(self, path: str | Path) -> None:
        """Implement init."""
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path, timeout=SQLITE_BUSY_TIMEOUT_SECONDS)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        """Implement close."""
        self.conn.close()

    def _init_schema(self) -> None:
        """Implement init schema."""
        self._create_schema()
        self._migrate_schema()
        self.conn.commit()

    def _create_schema(self) -> None:
        """Create current embedding storage schema."""
        self.conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            PRAGMA busy_timeout = 30000;
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS docsets (
                slug TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                version TEXT,
                release TEXT,
                mtime INTEGER,
                db_size INTEGER,
                content_format TEXT NOT NULL,
                installed_at TEXT NOT NULL,
                manifest_updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pages (
                id TEXT NOT NULL,
                docset_slug TEXT NOT NULL REFERENCES docsets(slug) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                title TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                type TEXT,
                content_hash TEXT,
                PRIMARY KEY (docset_slug, id)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                docset_slug TEXT NOT NULL,
                page_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE (docset_slug, page_id, ordinal),
                FOREIGN KEY (docset_slug, page_id) REFERENCES pages(docset_slug, id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                content_hash TEXT NOT NULL,
                PRIMARY KEY (chunk_id, model)
            );
            CREATE TABLE IF NOT EXISTS annoy_indexes (
                model TEXT PRIMARY KEY,
                dimensions INTEGER NOT NULL,
                source_version TEXT NOT NULL,
                path TEXT NOT NULL,
                item_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS annoy_items (
                model TEXT NOT NULL,
                annoy_id INTEGER NOT NULL,
                chunk_id INTEGER NOT NULL,
                PRIMARY KEY (model, annoy_id)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id'
            );
            """
        )

    def _migrate_schema(self) -> None:
        """Migrate existing embedding databases."""
        if self.schema_requires_reset():
            self.reset_schema()
            return
        embedding_columns = self.table_columns("embeddings")
        if "content_hash" not in embedding_columns:
            self.conn.execute("ALTER TABLE embeddings ADD COLUMN content_hash TEXT")
            self.conn.execute(
                """
                UPDATE embeddings
                SET content_hash = (
                    SELECT chunks.content_hash
                    FROM chunks
                    WHERE chunks.id = embeddings.chunk_id
                )
                """
            )
            self.conn.execute("DELETE FROM annoy_indexes")
            self.conn.execute("DELETE FROM annoy_items")

    def schema_requires_reset(self) -> bool:
        """Return whether an old incompatible embedding schema must be rebuilt."""
        pages_columns = self.table_columns("pages")
        embeddings_columns = self.table_columns("embeddings")
        docsets_columns = self.table_columns("docsets")
        return (
            not {"id", "title", "name", "path"}.issubset(pages_columns)
            or {"page_id", "page_title", "page_name", "page_path"}.intersection(pages_columns)
            or {"vector_encoding", "vector_hash"}.intersection(embeddings_columns)
            or {"created_at", "updated_at"}.intersection(docsets_columns)
        )

    def reset_schema(self) -> None:
        """Reset incompatible rebuildable embedding tables to the current schema."""
        self.conn.executescript(
            """
            PRAGMA foreign_keys = OFF;
            DROP TABLE IF EXISTS chunk_fts;
            DROP TABLE IF EXISTS annoy_items;
            DROP TABLE IF EXISTS annoy_indexes;
            DROP TABLE IF EXISTS embeddings;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS pages;
            DROP TABLE IF EXISTS docsets;
            PRAGMA foreign_keys = ON;
            """
        )
        self._create_schema()
        self.conn.execute("VACUUM")

    def table_columns(self, table_name: str) -> set[str]:
        """Return table columns."""
        return {str(row["name"]) for row in self.conn.execute(f"PRAGMA table_info({table_name})")}

    def replace_docset_chunks(
        self,
        docset: dict[str, object],
        chunks: list[object],
        vectors: list[list[float]] | None,
        model: str | None,
    ) -> None:
        """Implement replace docset chunks."""
        try:
            self._replace_docset_chunks_once(docset, chunks, vectors, model)
        except sqlite3.DatabaseError as exc:
            if not is_sqlite_malformed_error(exc):
                raise
            self.rebuild_chunk_fts()
            self._replace_docset_chunks_once(docset, chunks, vectors, model)

    def _replace_docset_chunks_once(
        self,
        docset: dict[str, object],
        chunks: list[object],
        vectors: list[list[float]] | None,
        model: str | None,
    ) -> None:
        """Replace docset chunks without retrying FTS corruption."""
        slug = str(docset["slug"])
        with self.conn:
            self.conn.execute(
                """
                DELETE FROM chunk_fts
                WHERE rowid IN (SELECT id FROM chunks WHERE docset_slug = ?)
                """,
                (slug,),
            )
            self.conn.execute("DELETE FROM docsets WHERE slug = ?", (slug,))
            self.conn.execute(
                """
                INSERT INTO docsets
                (slug, name, source, version, release, mtime, db_size, content_format, installed_at, manifest_updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slug,
                    docset["name"],
                    docset["source"],
                    docset.get("version"),
                    docset.get("release"),
                    docset.get("mtime"),
                    docset.get("dbSize"),
                    docset["contentFormat"],
                    docset["installedAt"],
                    docset["manifestUpdatedAt"],
                ),
            )
            for index, chunk in enumerate(chunks):
                page = chunk.page
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO pages
                    (id, docset_slug, file_path, title, name, path, type, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        page["id"],
                        slug,
                        page["filePath"],
                        page["title"],
                        page["name"],
                        page["path"],
                        page.get("type"),
                        page.get("contentHash"),
                    ),
                )
                cursor = self.conn.execute(
                    """
                    INSERT INTO chunks
                    (docset_slug, page_id, ordinal, content_hash, source_hash, text, token_count, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        slug,
                        page["id"],
                        chunk.ordinal,
                        chunk.content_hash,
                        chunk.source_hash,
                        chunk.text,
                        chunk.token_count,
                        chunk.metadata_json,
                    ),
                )
                chunk_id = int(cursor.lastrowid)
                self.conn.execute(
                    "INSERT INTO chunk_fts(rowid, text) VALUES (?, ?)",
                    (chunk_id, chunk.text),
                )
                if vectors is not None and model is not None:
                    vector = vectors[index]
                    self.conn.execute(
                        "INSERT INTO embeddings(chunk_id, model, dimensions, vector, content_hash) VALUES (?, ?, ?, ?, ?)",
                        (chunk_id, model, len(vector), vector_to_blob(vector), chunk.content_hash),
                    )
            self.conn.execute("DELETE FROM annoy_indexes")
            self.conn.execute("DELETE FROM annoy_items")

    def rebuild_chunk_fts(self) -> None:
        """Rebuild the chunk FTS side table from authoritative chunks."""
        with self.conn:
            self.conn.execute("DROP TABLE IF EXISTS chunk_fts")
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE chunk_fts USING fts5(
                    text,
                    content='chunks',
                    content_rowid='id'
                )
                """
            )
            self.conn.execute(
                """
                INSERT INTO chunk_fts(rowid, text)
                SELECT id, text FROM chunks
                """
            )

    def docset_embeddings_current(self, slug: str, chunks: list[object], model: str) -> bool:
        """Return whether stored embeddings match prepared chunks."""
        expected = sorted(
            (
                str(chunk.page["id"]),
                int(chunk.ordinal),
                str(chunk.content_hash),
                str(chunk.source_hash),
            )
            for chunk in chunks
        )
        rows = self.conn.execute(
            """
            SELECT c.page_id, c.ordinal, c.content_hash, c.source_hash, e.content_hash AS embedding_hash
            FROM chunks c
            LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.model = ?
            WHERE c.docset_slug = ?
            ORDER BY c.page_id, c.ordinal
            """,
            (model, slug),
        )
        actual = []
        embedded = 0
        for row in rows:
            actual.append(
                (
                    str(row["page_id"]),
                    int(row["ordinal"]),
                    str(row["content_hash"]),
                    str(row["source_hash"]),
                )
            )
            if row["embedding_hash"] == row["content_hash"]:
                embedded += 1
        return actual == expected and embedded == len(expected)

    def chunks_with_vectors(
        self, model: str, slugs: set[str] | None = None, limit: int | None = None
    ) -> list[SearchChunk]:
        """Implement chunks with vectors."""
        if slugs is not None and not slugs:
            return []
        params: list[object] = [model]
        where = "WHERE e.model = ?"
        if slugs is not None:
            placeholders = ",".join("?" for _ in slugs)
            where += f" AND c.docset_slug IN ({placeholders})"
            params.extend(sorted(slugs))
        sql = f"""
            SELECT c.id, c.docset_slug, d.name AS docset_name, c.page_id, p.title AS page_title,
                   p.path AS page_path, p.type AS page_type, p.file_path AS page_file,
                   c.ordinal, c.text, c.metadata_json, e.vector
            FROM embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN pages p ON p.docset_slug = c.docset_slug AND p.id = c.page_id
            JOIN docsets d ON d.slug = c.docset_slug
            {where}
        """
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [
            row_to_search_chunk(row, vector_from_blob(row["vector"]))
            for row in self.conn.execute(sql, params)
        ]

    def chunks_by_ids(
        self, chunk_ids: Iterable[int], model: str, slugs: set[str] | None = None
    ) -> list[SearchChunk]:
        """Implement chunks by ids."""
        ids = list(dict.fromkeys(chunk_ids))
        if not ids or (slugs is not None and not slugs):
            return []
        placeholders = ",".join("?" for _ in ids)
        params: list[object] = [model, *ids]
        slug_where = ""
        if slugs is not None:
            slug_placeholders = ",".join("?" for _ in slugs)
            slug_where = f" AND c.docset_slug IN ({slug_placeholders})"
            params.extend(sorted(slugs))
        sql = f"""
            SELECT c.id, c.docset_slug, d.name AS docset_name, c.page_id, p.title AS page_title,
                   p.path AS page_path, p.type AS page_type, p.file_path AS page_file,
                   c.ordinal, c.text, c.metadata_json, e.vector
            FROM chunks c
            JOIN embeddings e ON e.chunk_id = c.id AND e.model = ?
            JOIN pages p ON p.docset_slug = c.docset_slug AND p.id = c.page_id
            JOIN docsets d ON d.slug = c.docset_slug
            WHERE c.id IN ({placeholders}) {slug_where}
        """
        rows = {
            int(row["id"]): row_to_search_chunk(row, vector_from_blob(row["vector"]))
            for row in self.conn.execute(sql, params)
        }
        return [rows[id_] for id_ in ids if id_ in rows]

    def keyword_chunks(
        self, terms: list[str], slugs: set[str] | None = None, limit: int = 50
    ) -> list[SearchChunk]:
        """Implement keyword chunks."""
        if not terms or (slugs is not None and not slugs):
            return []
        fts_query = " OR ".join(f"{term}*" for term in terms)
        params: list[object] = [fts_query]
        slug_join = ""
        if slugs is not None:
            placeholders = ",".join("?" for _ in slugs)
            slug_join = f" AND c.docset_slug IN ({placeholders})"
            params.extend(sorted(slugs))
        params.append(limit)
        sql = f"""
            SELECT c.id, c.docset_slug, d.name AS docset_name, c.page_id, p.title AS page_title,
                   p.path AS page_path, p.type AS page_type, p.file_path AS page_file,
                   c.ordinal, c.text, c.metadata_json, NULL AS vector
            FROM chunk_fts f
            JOIN chunks c ON c.id = f.rowid
            JOIN pages p ON p.docset_slug = c.docset_slug AND p.id = c.page_id
            JOIN docsets d ON d.slug = c.docset_slug
            WHERE chunk_fts MATCH ? {slug_join}
            ORDER BY bm25(chunk_fts)
            LIMIT ?
        """
        return [row_to_search_chunk(row, None) for row in self.conn.execute(sql, params)]

    def stats(self, slug: str | None = None) -> dict[str, int]:
        """Implement stats."""
        params: tuple[object, ...] = (slug,) if slug else ()
        docset_where = "WHERE slug = ?" if slug else ""
        chunk_where = "WHERE docset_slug = ?" if slug else ""
        return {
            "indexedDocsets": self.conn.execute(
                f"SELECT COUNT(*) FROM docsets {docset_where}", params
            ).fetchone()[0],
            "indexedPages": self.conn.execute(
                f"SELECT COUNT(*) FROM pages {chunk_where}", params
            ).fetchone()[0],
            "indexedChunks": self.conn.execute(
                f"SELECT COUNT(*) FROM chunks {chunk_where}", params
            ).fetchone()[0],
            "embeddedChunks": self.conn.execute(
                f"SELECT COUNT(*) FROM embeddings e JOIN chunks c ON c.id = e.chunk_id {chunk_where}",
                params,
            ).fetchone()[0],
        }

    def embedding_index_version(self, model: str) -> str:
        """Implement embedding index version."""
        rows = self.conn.execute(
            """
            SELECT e.chunk_id, e.content_hash, e.dimensions
            FROM embeddings e
            WHERE e.model = ?
            ORDER BY e.chunk_id
            """,
            (model,),
        )
        digest = __import__("hashlib").sha256()
        count = 0
        for row in rows:
            count += 1
            digest.update(f"{row['chunk_id']}:{row['content_hash']}:{row['dimensions']}\n".encode())
        digest.update(f"count:{count}".encode())
        return digest.hexdigest()

    def embedding_rows_for_model(self, model: str) -> list[EmbeddingRow]:
        """Implement embedding rows for model."""
        rows = self.conn.execute(
            """
            SELECT chunk_id, vector
            FROM embeddings
            WHERE model = ?
            ORDER BY chunk_id
            """,
            (model,),
        )
        return [
            EmbeddingRow(chunk_id=int(row["chunk_id"]), vector=vector_from_blob(row["vector"]))
            for row in rows
        ]

    def annoy_index_metadata(self, model: str) -> dict[str, object] | None:
        """Implement annoy index metadata."""
        row = self.conn.execute(
            """
            SELECT dimensions, source_version, path, item_count
            FROM annoy_indexes
            WHERE model = ?
            """,
            (model,),
        ).fetchone()
        if row is None:
            return None
        return {
            "dimensions": int(row["dimensions"]),
            "source_version": str(row["source_version"]),
            "path": str(row["path"]),
            "item_count": int(row["item_count"]),
        }

    def replace_annoy_mapping(
        self,
        model: str,
        dimensions: int,
        source_version: str,
        path: Path,
        chunk_ids: list[int],
    ) -> None:
        """Implement replace annoy mapping."""
        with self.conn:
            self.conn.execute("DELETE FROM annoy_indexes WHERE model = ?", (model,))
            self.conn.execute("DELETE FROM annoy_items WHERE model = ?", (model,))
            self.conn.execute(
                """
                INSERT INTO annoy_indexes(model, dimensions, source_version, path, item_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (model, dimensions, source_version, str(path), len(chunk_ids)),
            )
            self.conn.executemany(
                "INSERT INTO annoy_items(model, annoy_id, chunk_id) VALUES (?, ?, ?)",
                [(model, annoy_id, chunk_id) for annoy_id, chunk_id in enumerate(chunk_ids)],
            )

    def annoy_chunk_ids(self, model: str, annoy_ids: Iterable[int]) -> list[int]:
        """Implement annoy chunk ids."""
        ids = list(annoy_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            SELECT annoy_id, chunk_id
            FROM annoy_items
            WHERE model = ? AND annoy_id IN ({placeholders})
            """,
            [model, *ids],
        )
        by_annoy_id = {int(row["annoy_id"]): int(row["chunk_id"]) for row in rows}
        return [by_annoy_id[id_] for id_ in ids if id_ in by_annoy_id]


def open_embedding_storage(cache_root: str | Path) -> EmbeddingStorage:
    """Implement open embedding storage."""
    return EmbeddingStorage(ensure_embedding_db_path(cache_root))


def is_sqlite_malformed_error(exc: sqlite3.DatabaseError) -> bool:
    """Return whether SQLite reported corrupt storage."""
    return "malformed" in str(exc).lower()


def row_to_search_chunk(row: sqlite3.Row, vector: list[float] | None) -> SearchChunk:
    """Implement row to search chunk."""
    return SearchChunk(
        id=int(row["id"]),
        docset_slug=str(row["docset_slug"]),
        docset_name=str(row["docset_name"]),
        page_id=str(row["page_id"]),
        page_title=str(row["page_title"]),
        page_path=str(row["page_path"]),
        page_type=row["page_type"],
        page_file=str(row["page_file"]),
        ordinal=int(row["ordinal"]),
        text=str(row["text"]),
        metadata_json=str(row["metadata_json"]),
        vector=vector,
    )


def vector_to_blob(vector: list[float]) -> bytes:
    """Implement vector to blob."""
    return struct.pack(f"<{len(vector)}d", *vector)


def vector_from_blob(blob: bytes) -> list[float]:
    """Implement vector from blob."""
    if len(blob) % 8 != 0:
        return []
    return list(struct.unpack(f"<{len(blob) // 8}d", blob))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Implement cosine similarity."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
