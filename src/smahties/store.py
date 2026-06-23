from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable

from .models import (
    CodeUnit,
    FileError,
    IndexedItem,
    LeaseStatus,
    LexicalMatch,
    Priority,
    QueueStats,
    QueuedWork,
    StoreStats,
    StoredEmbeddingCandidate,
    StoredCodeUnitEmbedding,
)
from .vector import vector_from_blob, vector_norm, vector_to_blob

SQLITE_BUSY_TIMEOUT_SECONDS = 30


class Store:
    """Thread-safe SQLite store for indexing state, FTS, queueing, and Annoy metadata."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(
            path, timeout=SQLITE_BUSY_TIMEOUT_SECONDS, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._open()

    def _open(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                PRAGMA foreign_keys = ON;
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
                INSERT INTO schema_version (version)
                SELECT 2 WHERE NOT EXISTS (SELECT 1 FROM schema_version);

                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    hash TEXT NOT NULL,
                    parser_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS code_units (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    start_byte INTEGER NOT NULL,
                    end_byte INTEGER NOT NULL,
                    unit_type TEXT NOT NULL,
                    name TEXT,
                    source TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    language TEXT NOT NULL,
                    parser_key TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    unit_id TEXT NOT NULL REFERENCES code_units(id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    norm REAL,
                    vector BLOB NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (unit_id, model)
                );

                CREATE INDEX IF NOT EXISTS idx_code_units_file_path ON code_units(file_path);
                CREATE INDEX IF NOT EXISTS idx_code_units_language_file_path
                ON code_units(language, file_path);
                CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

                CREATE VIRTUAL TABLE IF NOT EXISTS code_units_fts USING fts5(
                    unit_id UNINDEXED,
                    file_path UNINDEXED,
                    language UNINDEXED,
                    unit_type,
                    name,
                    source
                );

                CREATE TABLE IF NOT EXISTS work_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    delete_path INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    claimed_by TEXT,
                    claimed_at INTEGER,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_work_queue_ready
                ON work_queue(status, priority DESC, id);

                CREATE TABLE IF NOT EXISTS process_leases (
                    name TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    expires_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS annoy_indexes (
                    model TEXT PRIMARY KEY,
                    dimensions INTEGER NOT NULL,
                    item_count INTEGER NOT NULL,
                    source_version TEXT NOT NULL,
                    path TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS annoy_items (
                    model TEXT NOT NULL,
                    annoy_id INTEGER NOT NULL,
                    unit_id TEXT NOT NULL,
                    PRIMARY KEY (model, annoy_id)
                );
                """
            )
            self._conn.commit()
            self.ensure_lexical_index_current()

    def ensure_lexical_index_current(self) -> None:
        """Repair the FTS table when it has drifted from code_units."""

        indexed_units = self._conn.execute("SELECT COUNT(*) FROM code_units").fetchone()[0]
        lexical_units = self._conn.execute("SELECT COUNT(*) FROM code_units_fts").fetchone()[0]
        if indexed_units == lexical_units:
            return
        self._conn.executescript(
            """
            DELETE FROM code_units_fts
            WHERE unit_id NOT IN (SELECT id FROM code_units);

            INSERT INTO code_units_fts (unit_id, file_path, language, unit_type, name, source)
            SELECT u.id, u.file_path, u.language, u.unit_type, u.name, u.source
            FROM code_units u
            WHERE NOT EXISTS (
                SELECT 1 FROM code_units_fts f WHERE f.unit_id = u.id
            );
            """
        )
        self._conn.commit()

    def file_complete_for_model(self, path: str, hash_: str, parser_key: str, model: str) -> bool:
        """Return whether a file is fully indexed for the parser and embedding model."""

        with self._lock:
            row = self._conn.execute(
                """
                SELECT f.hash, f.parser_key, COUNT(u.id) AS units, COUNT(e.unit_id) AS embeddings
                FROM files f
                LEFT JOIN code_units u ON u.file_path = f.path
                LEFT JOIN embeddings e ON e.unit_id = u.id AND e.model = ?
                WHERE f.path = ? AND f.status = 'indexed'
                GROUP BY f.path
                """,
                (model, path),
            ).fetchone()
        return bool(
            row
            and row["hash"] == hash_
            and row["parser_key"] == parser_key
            and row["units"] > 0
            and row["units"] == row["embeddings"]
        )

    def replace_file_units(
        self,
        path: str,
        hash_: str,
        parser_key: str,
        units: list[CodeUnit],
        model: str,
        embeddings: list[list[float]],
    ) -> None:
        """Replace all indexed units and embeddings for one source file."""

        if len(units) != len(embeddings):
            raise ValueError(
                f"embedding response count {len(embeddings)} did not match file unit count {len(units)}"
            )
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM code_units_fts WHERE file_path = ?", (path,))
            self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
            self._conn.execute(
                """
                INSERT INTO files (path, hash, parser_key, status, error, updated_at)
                VALUES (?, ?, ?, 'indexed', NULL, CURRENT_TIMESTAMP)
                """,
                (path, hash_, parser_key),
            )
            for unit, vector in zip(units, embeddings, strict=True):
                self._conn.execute(
                    """
                    INSERT INTO code_units (
                        id, file_path, start_line, end_line, start_byte, end_byte,
                        unit_type, name, source, source_hash, language, parser_key
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        unit.id,
                        unit.file_path,
                        unit.start_line,
                        unit.end_line,
                        unit.start_byte,
                        unit.end_byte,
                        unit.unit_type,
                        unit.name,
                        unit.source,
                        unit.source_hash,
                        unit.language,
                        unit.parser_key,
                    ),
                )
                self._conn.execute(
                    """
                    INSERT INTO embeddings (unit_id, model, dimensions, norm, vector, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (unit.id, model, len(vector), vector_norm(vector), vector_to_blob(vector)),
                )
                self._conn.execute(
                    """
                    INSERT INTO code_units_fts (unit_id, file_path, language, unit_type, name, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        unit.id,
                        unit.file_path,
                        unit.language,
                        unit.unit_type,
                        unit.name,
                        unit.source,
                    ),
                )

    def mark_error(self, path: str, error: str) -> None:
        """Persist the latest indexing error for a source path."""

        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO files (path, hash, parser_key, status, error, updated_at)
                VALUES (?, '', '', 'error', ?, CURRENT_TIMESTAMP)
                ON CONFLICT(path) DO UPDATE SET
                    status = 'error',
                    error = excluded.error,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (path, error),
            )

    def delete_file(self, path: str) -> None:
        """Delete indexed state for one source file."""

        with self._lock, self._conn:
            self._conn.execute("DELETE FROM code_units_fts WHERE file_path = ?", (path,))
            self._conn.execute("DELETE FROM files WHERE path = ?", (path,))

    def delete_path_prefix(self, prefix: str) -> None:
        """Delete indexed state for a path and all children under it."""

        with self._lock, self._conn:
            self._conn.execute(
                f"DELETE FROM code_units_fts WHERE {path_prefix_clause('file_path')}",
                path_prefix_params(prefix),
            )
            self._conn.execute(
                f"DELETE FROM files WHERE {path_prefix_clause('path')}",
                path_prefix_params(prefix),
            )

    def delete_path_part(self, part: str) -> None:
        """Delete indexed state for files with a matching path component."""

        with self._lock, self._conn:
            self._conn.execute(
                f"DELETE FROM code_units_fts WHERE {path_part_clause('file_path')}",
                path_part_params(part),
            )
            self._conn.execute(
                f"DELETE FROM files WHERE {path_part_clause('path')}",
                path_part_params(part),
            )

    def delete_file_name(self, file_name: str) -> None:
        """Delete indexed state for files matching an excluded file name."""

        with self._lock, self._conn:
            self._conn.execute(
                f"DELETE FROM code_units_fts WHERE {path_terminal_part_clause('file_path')}",
                path_terminal_part_params(file_name),
            )
            self._conn.execute(
                f"DELETE FROM files WHERE {path_terminal_part_clause('path')}",
                path_terminal_part_params(file_name),
            )

    def enqueue_work(self, path: Path, priority: Priority, delete: bool) -> None:
        """Insert or reprioritize pending indexing or delete work."""

        now = unix_now()
        path_text = str(path)
        delete_path = int(delete)
        with self._lock, self._conn:
            changed = self._conn.execute(
                """
                UPDATE work_queue
                SET priority = MAX(priority, ?), updated_at = ?
                WHERE path = ? AND delete_path = ? AND status = 'pending'
                """,
                (priority.value, now, path_text, delete_path),
            ).rowcount
            if changed:
                return
            self._conn.execute(
                """
                INSERT INTO work_queue (path, priority, delete_path, status, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (path_text, priority.value, delete_path, now, now),
            )

    def claim_next_work(self, owner: str, stale_after_seconds: int) -> QueuedWork | None:
        """Claim the next pending queue item, reclaiming stale work first."""

        now = unix_now()
        stale_cutoff = now - stale_after_seconds
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE work_queue
                SET status = 'pending', claimed_by = NULL, claimed_at = NULL, updated_at = ?
                WHERE status = 'in_progress' AND claimed_at IS NOT NULL AND claimed_at <= ?
                """,
                (now, stale_cutoff),
            )
            row = self._conn.execute(
                """
                SELECT id, path, priority, delete_path
                FROM work_queue
                WHERE status = 'pending'
                ORDER BY priority DESC, id ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                """
                UPDATE work_queue
                SET status = 'in_progress',
                    attempts = attempts + 1,
                    claimed_by = ?,
                    claimed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (owner, now, now, row["id"]),
            )
        return QueuedWork(
            id=row["id"],
            path=Path(row["path"]),
            priority=Priority.from_int(row["priority"]),
            delete=bool(row["delete_path"]),
        )

    def complete_work_for_owner(self, id_: int, owner: str) -> None:
        """Delete a queue item when it is still owned by the caller."""

        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM work_queue WHERE id = ? AND claimed_by = ?", (id_, owner)
            )

    def fail_work_for_owner(self, id_: int, owner: str, error: str) -> bool:
        """Return owned work to pending state and record the failure reason."""

        now = unix_now()
        with self._lock, self._conn:
            changed = self._conn.execute(
                """
                UPDATE work_queue
                SET status = 'pending',
                    error = ?,
                    claimed_by = NULL,
                    claimed_at = NULL,
                    updated_at = ?
                WHERE id = ? AND claimed_by = ?
                """,
                (error, now, id_, owner),
            ).rowcount
        return changed > 0

    def requeue_work_for_owner(self, owner: str, error: str) -> int:
        """Return all work claimed by an owner to pending state."""

        now = unix_now()
        with self._lock, self._conn:
            return self._conn.execute(
                """
                UPDATE work_queue
                SET status = 'pending',
                    error = ?,
                    claimed_by = NULL,
                    claimed_at = NULL,
                    updated_at = ?
                WHERE status = 'in_progress' AND claimed_by = ?
                """,
                (error, now, owner),
            ).rowcount

    def renew_work_for_owner(self, owner: str) -> int:
        """Refresh claimed-work timestamps for a live owner."""

        now = unix_now()
        with self._lock, self._conn:
            return self._conn.execute(
                """
                UPDATE work_queue
                SET claimed_at = ?, updated_at = ?
                WHERE status = 'in_progress' AND claimed_by = ?
                """,
                (now, now, owner),
            ).rowcount

    def queue_stats(self) -> QueueStats:
        """Return counts for pending high/low priority and in-progress work."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT status, priority, COUNT(*) AS count FROM work_queue GROUP BY status, priority"
            ).fetchall()
        high = low = in_progress = 0
        for row in rows:
            if row["status"] == "in_progress":
                in_progress += row["count"]
            elif row["priority"] >= Priority.HIGH.value:
                high += row["count"]
            else:
                low += row["count"]
        return QueueStats(high, low, in_progress)

    def acquire_lease(self, name: str, owner: str, ttl_seconds: int) -> bool:
        """Acquire or renew a named lease if it is free or expired."""

        now = unix_now()
        expires_at = now + ttl_seconds
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT owner, expires_at FROM process_leases WHERE name = ?", (name,)
            ).fetchone()
            if row and row["owner"] != owner and row["expires_at"] > now:
                return False
            self._conn.execute(
                """
                INSERT INTO process_leases (name, owner, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET owner = excluded.owner, expires_at = excluded.expires_at
                """,
                (name, owner, expires_at),
            )
        return True

    def lease_status(self, name: str, owner: str) -> LeaseStatus:
        """Return the current status of a named process lease."""

        with self._lock:
            row = self._conn.execute(
                "SELECT owner, expires_at FROM process_leases WHERE name = ?", (name,)
            ).fetchone()
        if not row or row["expires_at"] <= unix_now():
            return LeaseStatus(None, None, False)
        return LeaseStatus(row["owner"], row["expires_at"], row["owner"] == owner)

    def stats(self) -> StoreStats:
        """Return aggregate store counts and recent file errors."""

        with self._lock:
            indexed_files = self._conn.execute(
                "SELECT COUNT(*) FROM files WHERE status = 'indexed'"
            ).fetchone()[0]
            indexed_units = self._conn.execute("SELECT COUNT(*) FROM code_units").fetchone()[0]
            embedded_units = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            lexical_units = self._conn.execute("SELECT COUNT(*) FROM code_units_fts").fetchone()[0]
            errors = self._conn.execute(
                """
                SELECT path, error FROM files
                WHERE status = 'error' AND error IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 10
                """
            ).fetchall()
        return StoreStats(
            indexed_files,
            indexed_units,
            embedded_units,
            lexical_units,
            [FileError(row["path"], row["error"]) for row in errors],
        )

    def file_paths(self) -> list[str]:
        """Return indexed file paths tracked by the store."""

        with self._lock:
            rows = self._conn.execute("SELECT path FROM files ORDER BY path").fetchall()
        return [row["path"] for row in rows]

    def file_paths_under(self, prefix: str | None) -> list[str]:
        """Return tracked file paths under an optional directory prefix."""

        if prefix is None:
            return self.file_paths()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT path FROM files WHERE {path_prefix_clause('path')} ORDER BY path",
                path_prefix_params(prefix),
            ).fetchall()
        return [row["path"] for row in rows]

    def lexical_search(
        self,
        query: str,
        path_prefix: str | None,
        language: str | None,
        limit: int,
    ) -> list[LexicalMatch]:
        """Search the FTS index with optional path and language filters."""

        clauses = ["code_units_fts MATCH ?"]
        params: list[object] = [query]
        if path_prefix:
            clauses.append(path_prefix_clause("f.file_path"))
            params.extend(path_prefix_params(path_prefix))
        if language:
            clauses.append("f.language = ?")
            params.append(language)
        params.append(limit)
        sql = f"""
            SELECT u.*, bm25(code_units_fts) AS rank
            FROM code_units_fts f
            JOIN code_units u ON u.id = f.unit_id
            WHERE {" AND ".join(clauses)}
            ORDER BY rank
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [LexicalMatch(_code_unit_from_row(row), float(row["rank"])) for row in rows]

    def code_units_by_ids(self, ids: Iterable[str]) -> list[CodeUnit]:
        """Load code units for IDs, preserving database row contents."""

        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM code_units WHERE id IN ({placeholders})", ids
            ).fetchall()
        return [_code_unit_from_row(row) for row in rows]

    def embedding_candidates_by_ids(
        self, model: str, ids: Iterable[str]
    ) -> list[StoredEmbeddingCandidate]:
        """Load stored embedding vectors for exact candidate scoring."""

        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT unit_id, vector, norm FROM embeddings
                WHERE model = ? AND unit_id IN ({placeholders})
                """,
                [model, *ids],
            ).fetchall()
        return [
            StoredEmbeddingCandidate(
                row["unit_id"], vector_from_blob(row["vector"]), float(row["norm"])
            )
            for row in rows
        ]

    def embedding_rows_for_model(self, model: str) -> list[StoredEmbeddingCandidate]:
        """Load all embeddings for a model in stable order for Annoy rebuilds."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT unit_id, vector, norm FROM embeddings WHERE model = ? ORDER BY unit_id",
                (model,),
            ).fetchall()
        return [
            StoredEmbeddingCandidate(
                row["unit_id"], vector_from_blob(row["vector"]), float(row["norm"])
            )
            for row in rows
        ]

    def code_unit_embeddings_for_model(
        self,
        model: str,
        path_prefixes: Iterable[str | None],
        language: str | None,
    ) -> list[StoredCodeUnitEmbedding]:
        """Load code units and embeddings for duplicate detection."""

        clauses = ["e.model = ?"]
        params: list[object] = [model]
        prefixes = [prefix for prefix in dict.fromkeys(path_prefixes) if prefix]
        if prefixes:
            prefix_clauses = []
            for prefix in prefixes:
                prefix_clauses.append(path_prefix_clause("u.file_path"))
                params.extend(path_prefix_params(prefix))
            clauses.append(f"({' OR '.join(prefix_clauses)})")
        if language:
            clauses.append("u.language = ?")
            params.append(language)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT
                    u.id, u.file_path, u.start_line, u.end_line, u.start_byte, u.end_byte,
                    u.unit_type, u.name, u.source, u.source_hash, u.language, u.parser_key,
                    e.vector, e.norm
                FROM code_units u
                JOIN embeddings e ON e.unit_id = u.id
                WHERE {" AND ".join(clauses)}
                ORDER BY u.file_path, u.start_line, u.end_line, u.id
                """,
                params,
            ).fetchall()
        return [
            StoredCodeUnitEmbedding(
                _code_unit_from_row(row), vector_from_blob(row["vector"]), float(row["norm"])
            )
            for row in rows
        ]

    def embedding_index_version(self, model: str) -> str:
        """Return a compact version string for a model's embedding set."""

        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS count, COALESCE(MAX(updated_at), '') AS updated FROM embeddings WHERE model = ?",
                (model,),
            ).fetchone()
        return f"{row['count']}:{row['updated']}"

    def annoy_index_metadata(self, model: str) -> sqlite3.Row | None:
        """Return persisted Annoy sidecar metadata for a model."""

        with self._lock:
            return self._conn.execute(
                "SELECT * FROM annoy_indexes WHERE model = ?", (model,)
            ).fetchone()

    def replace_annoy_mapping(
        self,
        model: str,
        dimensions: int,
        source_version: str,
        path: Path,
        unit_ids: list[str],
    ) -> None:
        """Replace Annoy integer-ID mappings and sidecar metadata for a model."""

        with self._lock, self._conn:
            self._conn.execute("DELETE FROM annoy_items WHERE model = ?", (model,))
            for annoy_id, unit_id in enumerate(unit_ids):
                self._conn.execute(
                    "INSERT INTO annoy_items (model, annoy_id, unit_id) VALUES (?, ?, ?)",
                    (model, annoy_id, unit_id),
                )
            self._conn.execute(
                """
                INSERT INTO annoy_indexes (model, dimensions, item_count, source_version, path, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(model) DO UPDATE SET
                    dimensions = excluded.dimensions,
                    item_count = excluded.item_count,
                    source_version = excluded.source_version,
                    path = excluded.path,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (model, dimensions, len(unit_ids), source_version, str(path)),
            )

    def annoy_unit_ids(self, model: str, annoy_ids: Iterable[int]) -> list[str]:
        """Translate Annoy integer IDs to code unit IDs in input order."""

        ids = list(annoy_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT annoy_id, unit_id FROM annoy_items
                WHERE model = ? AND annoy_id IN ({placeholders})
                """,
                [model, *ids],
            ).fetchall()
        mapping = {row["annoy_id"]: row["unit_id"] for row in rows}
        return [mapping[id_] for id_ in ids if id_ in mapping]

    def list_indexed_units(
        self,
        path_prefix: str | None,
        language: str | None,
        limit: int,
        offset: int,
        include_source: bool,
    ) -> list[IndexedItem]:
        """List indexed units with optional filters, pagination, and source."""

        clauses: list[str] = []
        params: list[object] = []
        if path_prefix:
            clauses.append(path_prefix_clause("file_path"))
            params.extend(path_prefix_params(path_prefix))
        if language:
            clauses.append("language = ?")
            params.append(language)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        source_expr = "source" if include_source else "NULL AS source"
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT file_path, language, unit_type, name, start_line, end_line, {source_expr}
                FROM code_units
                {where}
                ORDER BY file_path, start_line, id
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [
            IndexedItem(
                row["file_path"],
                row["language"],
                row["unit_type"],
                row["name"],
                row["start_line"],
                row["end_line"],
                row["source"],
            )
            for row in rows
        ]


def _code_unit_from_row(row: sqlite3.Row) -> CodeUnit:
    return CodeUnit(
        id=row["id"],
        file_path=row["file_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        start_byte=row["start_byte"],
        end_byte=row["end_byte"],
        unit_type=row["unit_type"],
        name=row["name"],
        source=row["source"],
        source_hash=row["source_hash"],
        language=row["language"],
        parser_key=row["parser_key"],
    )


def path_prefix_clause(column: str) -> str:
    """Return a SQL fragment for literal path-prefix matching."""

    return f"({column} = ? OR substr({column}, 1, length(?) + 1) = ? || '/')"


def path_prefix_params(prefix: str) -> tuple[str, str, str]:
    """Return repeated parameters for path_prefix_clause."""

    return (prefix, prefix, prefix)


def path_part_clause(column: str) -> str:
    """Return a SQL fragment for literal path component matching."""

    return (
        f"({column} = ? "
        f"OR substr({column}, 1, length(?) + 1) = ? || '/' "
        f"OR substr({column}, -length(?) - 1) = '/' || ? "
        f"OR instr({column}, '/' || ? || '/') > 0)"
    )


def path_part_params(part: str) -> tuple[str, str, str, str, str, str]:
    """Return repeated parameters for path_part_clause."""

    return (part, part, part, part, part, part)


def path_terminal_part_clause(column: str) -> str:
    """Return a SQL fragment for matching a literal final path component."""

    return f"({column} = ? OR substr({column}, -length(?) - 1) = '/' || ?)"


def path_terminal_part_params(part: str) -> tuple[str, str, str]:
    """Return repeated parameters for path_terminal_part_clause."""

    return (part, part, part)


def unix_now() -> int:
    """Return current Unix time in seconds."""

    return int(time.time())
