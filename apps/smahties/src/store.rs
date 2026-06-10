use std::{
    fs, io,
    path::Path,
    sync::{Mutex, MutexGuard},
};

use rusqlite::{Connection, OptionalExtension, params, types::Type};

use crate::{
    Result, SmahtiesError,
    model::{
        CodeUnit, FileError, IndexedItem, LeaseStatus, LexicalMatch, Priority, QueueStats,
        QueuedWork, StoreStats, StoredEmbedding,
    },
    vector::{vector_from_blob, vector_to_blob},
};

pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;
        conn.execute_batch(
            r#"
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );

            INSERT INTO schema_version (version)
            SELECT 1
            WHERE NOT EXISTS (SELECT 1 FROM schema_version);

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
                vector BLOB NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (unit_id, model)
            );

            CREATE INDEX IF NOT EXISTS idx_code_units_file_path ON code_units(file_path);
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
            "#,
        )?;
        Self::repair_lexical_index_conn(&conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn repair_lexical_index_conn(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            DELETE FROM code_units_fts
            WHERE unit_id NOT IN (SELECT id FROM code_units);

            INSERT INTO code_units_fts (
                unit_id, file_path, language, unit_type, name, source
            )
            SELECT u.id, u.file_path, u.language, u.unit_type, u.name, u.source
            FROM code_units u
            WHERE NOT EXISTS (
                SELECT 1
                FROM code_units_fts f
                WHERE f.unit_id = u.id
            );
            "#,
        )?;
        Ok(())
    }

    pub fn ensure_lexical_index_current(&self) -> Result<()> {
        let conn = self.lock()?;
        let indexed_units = conn.query_row("SELECT COUNT(*) FROM code_units", [], |row| {
            row.get::<_, u64>(0)
        })?;
        let lexical_units = conn.query_row("SELECT COUNT(*) FROM code_units_fts", [], |row| {
            row.get::<_, u64>(0)
        })?;
        if indexed_units != lexical_units {
            Self::repair_lexical_index_conn(&conn)?;
        }
        Ok(())
    }

    pub fn file_complete_for_model(
        &self,
        path: &str,
        hash: &str,
        parser_key: &str,
        model: &str,
    ) -> Result<bool> {
        let conn = self.lock()?;
        let row = conn
            .query_row(
                r#"
                SELECT f.hash, f.parser_key, COUNT(u.id), COUNT(e.unit_id)
                FROM files f
                LEFT JOIN code_units u ON u.file_path = f.path
                LEFT JOIN embeddings e ON e.unit_id = u.id AND e.model = ?2
                WHERE f.path = ?1 AND f.status = 'indexed'
                GROUP BY f.path
                "#,
                params![path, model],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, u64>(2)?,
                        row.get::<_, u64>(3)?,
                    ))
                },
            )
            .optional()?;

        Ok(
            row.is_some_and(|(stored_hash, stored_parser_key, units, embeddings)| {
                stored_hash == hash
                    && stored_parser_key == parser_key
                    && units > 0
                    && units == embeddings
            }),
        )
    }

    pub fn replace_file_units(
        &self,
        path: &str,
        hash: &str,
        parser_key: &str,
        units: &[CodeUnit],
        model: &str,
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM code_units_fts WHERE file_path = ?1",
            params![path],
        )?;
        tx.execute("DELETE FROM files WHERE path = ?1", params![path])?;
        tx.execute(
            r#"
            INSERT INTO files (path, hash, parser_key, status, error, updated_at)
            VALUES (?1, ?2, ?3, 'indexed', NULL, CURRENT_TIMESTAMP)
            "#,
            params![path, hash, parser_key],
        )?;

        for (unit, vector) in units.iter().zip(embeddings) {
            tx.execute(
                r#"
                INSERT INTO code_units (
                    id, file_path, start_line, end_line, start_byte, end_byte,
                    unit_type, name, source, source_hash, language, parser_key
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
                "#,
                params![
                    unit.id,
                    unit.file_path,
                    unit.start_line,
                    unit.end_line,
                    unit.start_byte as u64,
                    unit.end_byte as u64,
                    unit.unit_type,
                    unit.name,
                    unit.source,
                    unit.source_hash,
                    unit.language,
                    unit.parser_key,
                ],
            )?;
            tx.execute(
                r#"
                INSERT INTO embeddings (unit_id, model, dimensions, vector, updated_at)
                VALUES (?1, ?2, ?3, ?4, CURRENT_TIMESTAMP)
                "#,
                params![unit.id, model, vector.len() as u64, vector_to_blob(vector)],
            )?;
            tx.execute(
                r#"
                INSERT INTO code_units_fts (
                    unit_id, file_path, language, unit_type, name, source
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                "#,
                params![
                    unit.id,
                    unit.file_path,
                    unit.language,
                    unit.unit_type,
                    unit.name,
                    unit.source,
                ],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    pub fn mark_error(&self, path: &str, error: &str) -> Result<()> {
        let conn = self.lock()?;
        conn.execute(
            r#"
            INSERT INTO files (path, hash, parser_key, status, error, updated_at)
            VALUES (?1, '', '', 'error', ?2, CURRENT_TIMESTAMP)
            ON CONFLICT(path) DO UPDATE SET
                status = 'error',
                error = excluded.error,
                updated_at = CURRENT_TIMESTAMP
            "#,
            params![path, error],
        )?;
        Ok(())
    }

    pub fn delete_file(&self, path: &str) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM code_units_fts WHERE file_path = ?1",
            params![path],
        )?;
        tx.execute("DELETE FROM files WHERE path = ?1", params![path])?;
        tx.commit()?;
        Ok(())
    }

    pub fn delete_path_prefix(&self, prefix: &str) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        let prefix = prefix.trim_end_matches('/');
        let prefix_like = format!("{prefix}/%");
        tx.execute(
            "DELETE FROM code_units_fts WHERE file_path = ?1 OR file_path LIKE ?2",
            params![prefix, prefix_like],
        )?;
        tx.execute(
            "DELETE FROM files WHERE path = ?1 OR path LIKE ?2",
            params![prefix, prefix_like],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn delete_file_name(&self, file_name: &str) -> Result<()> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        let file_like = format!("%/{file_name}");
        tx.execute(
            "DELETE FROM code_units_fts WHERE file_path = ?1 OR file_path LIKE ?2",
            params![file_name, file_like],
        )?;
        tx.execute(
            "DELETE FROM files WHERE path = ?1 OR path LIKE ?2",
            params![file_name, file_like],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn enqueue_work(&self, path: &Path, priority: Priority, delete: bool) -> Result<()> {
        let conn = self.lock()?;
        let now = Self::unix_now();
        let path = path.to_string_lossy().into_owned();
        let priority = priority.as_i64();
        let delete_path = i64::from(delete);
        let changed = conn.execute(
            r#"
            UPDATE work_queue
            SET priority = MAX(priority, ?2),
                updated_at = ?4
            WHERE path = ?1
              AND delete_path = ?3
              AND status = 'pending'
            "#,
            params![path, priority, delete_path, now],
        )?;
        if changed > 0 {
            return Ok(());
        }

        conn.execute(
            r#"
            INSERT INTO work_queue (
                path, priority, delete_path, status, created_at, updated_at
            )
            VALUES (?1, ?2, ?3, 'pending', ?4, ?4)
            "#,
            params![path, priority, delete_path, now],
        )?;
        Ok(())
    }

    pub fn claim_next_work(
        &self,
        owner: &str,
        stale_after_seconds: i64,
    ) -> Result<Option<QueuedWork>> {
        let mut conn = self.lock()?;
        let tx = conn.transaction()?;
        let now = Self::unix_now();
        let stale_cutoff = now.saturating_sub(stale_after_seconds);
        tx.execute(
            r#"
            UPDATE work_queue
            SET status = 'pending',
                claimed_by = NULL,
                claimed_at = NULL,
                updated_at = ?1
            WHERE status = 'in_progress'
              AND claimed_at IS NOT NULL
              AND claimed_at <= ?2
            "#,
            params![now, stale_cutoff],
        )?;
        let row = tx
            .query_row(
                r#"
                SELECT id, path, priority, delete_path
                FROM work_queue
                WHERE status = 'pending'
                ORDER BY priority DESC, id ASC
                LIMIT 1
                "#,
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;

        let Some((id, path, priority, delete_path)) = row else {
            tx.commit()?;
            return Ok(None);
        };

        tx.execute(
            r#"
            UPDATE work_queue
            SET status = 'in_progress',
                attempts = attempts + 1,
                claimed_by = ?1,
                claimed_at = ?2,
                updated_at = ?2
            WHERE id = ?3
            "#,
            params![owner, now, id],
        )?;
        tx.commit()?;

        Ok(Some(QueuedWork {
            id,
            path: path.into(),
            priority: Priority::from_i64(priority),
            delete: delete_path != 0,
        }))
    }

    pub fn complete_work(&self, id: i64) -> Result<()> {
        let conn = self.lock()?;
        conn.execute("DELETE FROM work_queue WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn complete_work_for_owner(&self, id: i64, owner: &str) -> Result<()> {
        let conn = self.lock()?;
        conn.execute(
            "DELETE FROM work_queue WHERE id = ?1 AND claimed_by = ?2",
            params![id, owner],
        )?;
        Ok(())
    }

    pub fn fail_work_for_owner(&self, id: i64, owner: &str, error: &str) -> Result<bool> {
        let conn = self.lock()?;
        let now = Self::unix_now();
        let changed = conn.execute(
            r#"
            UPDATE work_queue
            SET status = 'pending',
                error = ?1,
                claimed_by = NULL,
                claimed_at = NULL,
                updated_at = ?2
            WHERE id = ?3 AND claimed_by = ?4
            "#,
            params![error, now, id, owner],
        )?;
        Ok(changed > 0)
    }

    pub fn touch_work_claim(&self, id: i64, owner: &str) -> Result<()> {
        let conn = self.lock()?;
        let now = Self::unix_now();
        conn.execute(
            r#"
            UPDATE work_queue
            SET claimed_at = ?1,
                updated_at = ?1
            WHERE id = ?2 AND claimed_by = ?3 AND status = 'in_progress'
            "#,
            params![now, id, owner],
        )?;
        Ok(())
    }

    pub fn queue_stats(&self) -> Result<QueueStats> {
        let conn = self.lock()?;
        let high_priority = conn.query_row(
            "SELECT COUNT(*) FROM work_queue WHERE status = 'pending' AND priority >= 100",
            [],
            |row| row.get(0),
        )?;
        let low_priority = conn.query_row(
            "SELECT COUNT(*) FROM work_queue WHERE status = 'pending' AND priority < 100",
            [],
            |row| row.get(0),
        )?;
        let in_progress = conn.query_row(
            "SELECT COUNT(*) FROM work_queue WHERE status = 'in_progress'",
            [],
            |row| row.get(0),
        )?;

        Ok(QueueStats {
            high_priority,
            low_priority,
            in_progress,
        })
    }

    pub fn acquire_lease(&self, lease_name: &str, owner: &str, ttl_seconds: i64) -> Result<bool> {
        let conn = self.lock()?;
        let now = Self::unix_now();
        let expires_at = now + ttl_seconds;
        conn.execute(
            r#"
            INSERT INTO process_leases (name, owner, expires_at)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(name) DO UPDATE SET
                owner = excluded.owner,
                expires_at = excluded.expires_at
            WHERE process_leases.owner = ?2 OR process_leases.expires_at <= ?4
            "#,
            params![lease_name, owner, expires_at, now],
        )?;

        let current_owner: Option<String> = conn
            .query_row(
                "SELECT owner FROM process_leases WHERE name = ?1",
                params![lease_name],
                |row| row.get(0),
            )
            .optional()?;
        Ok(current_owner.as_deref() == Some(owner))
    }

    pub fn lease_status(&self, lease_name: &str, owner: &str) -> Result<LeaseStatus> {
        let conn = self.lock()?;
        let row = conn
            .query_row(
                "SELECT owner, expires_at FROM process_leases WHERE name = ?1",
                params![lease_name],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
            )
            .optional()?;

        Ok(match row {
            Some((lease_owner, expires_at)) => LeaseStatus {
                held_by_this_process: lease_owner == owner && expires_at > Self::unix_now(),
                owner: Some(lease_owner),
                expires_at_unix: Some(expires_at),
            },
            None => LeaseStatus {
                owner: None,
                expires_at_unix: None,
                held_by_this_process: false,
            },
        })
    }

    pub fn embeddings_for_model(
        &self,
        model: &str,
        path_prefix: Option<&str>,
        language: Option<&str>,
    ) -> Result<Vec<StoredEmbedding>> {
        let conn = self.lock()?;
        let path_like = path_prefix.map(|prefix| format!("{prefix}%"));
        let mut stmt = conn.prepare(
            r#"
            SELECT
                u.id, u.file_path, u.start_line, u.end_line, u.start_byte, u.end_byte,
                u.unit_type, u.name, u.source, u.source_hash, u.language, u.parser_key,
                e.vector
            FROM code_units u
            JOIN embeddings e ON e.unit_id = u.id
            WHERE e.model = ?1
              AND (?2 IS NULL OR u.file_path LIKE ?2)
              AND (?3 IS NULL OR u.language = ?3)
            "#,
        )?;

        let rows = stmt.query_map(params![model, path_like, language], |row| {
            let blob: Vec<u8> = row.get(12)?;
            let vector = vector_from_blob(&blob).map_err(|message| {
                rusqlite::Error::FromSqlConversionFailure(
                    12,
                    Type::Blob,
                    Box::new(io::Error::new(io::ErrorKind::InvalidData, message)),
                )
            })?;

            Ok(StoredEmbedding {
                unit: CodeUnit {
                    id: row.get(0)?,
                    file_path: row.get(1)?,
                    start_line: row.get(2)?,
                    end_line: row.get(3)?,
                    start_byte: row.get::<_, u64>(4)? as usize,
                    end_byte: row.get::<_, u64>(5)? as usize,
                    unit_type: row.get(6)?,
                    name: row.get(7)?,
                    source: row.get(8)?,
                    source_hash: row.get(9)?,
                    language: row.get(10)?,
                    parser_key: row.get(11)?,
                },
                vector,
            })
        })?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn lexical_search(
        &self,
        fts_query: &str,
        path_prefix: Option<&str>,
        language: Option<&str>,
        limit: usize,
    ) -> Result<Vec<LexicalMatch>> {
        let conn = self.lock()?;
        let path_like = path_prefix.map(|prefix| format!("{prefix}%"));
        let mut stmt = conn.prepare(
            r#"
            SELECT
                u.id, u.file_path, u.start_line, u.end_line, u.start_byte, u.end_byte,
                u.unit_type, u.name, u.source, u.source_hash, u.language, u.parser_key,
                code_units_fts.rank
            FROM code_units_fts
            JOIN code_units u ON u.id = code_units_fts.unit_id
            WHERE code_units_fts MATCH ?1
              AND (?2 IS NULL OR u.file_path LIKE ?2)
              AND (?3 IS NULL OR u.language = ?3)
            ORDER BY code_units_fts.rank
            LIMIT ?4
            "#,
        )?;

        let rows = stmt.query_map(
            params![fts_query, path_like, language, limit as u64],
            |row| {
                Ok(LexicalMatch {
                    unit: CodeUnit {
                        id: row.get(0)?,
                        file_path: row.get(1)?,
                        start_line: row.get(2)?,
                        end_line: row.get(3)?,
                        start_byte: row.get::<_, u64>(4)? as usize,
                        end_byte: row.get::<_, u64>(5)? as usize,
                        unit_type: row.get(6)?,
                        name: row.get(7)?,
                        source: row.get(8)?,
                        source_hash: row.get(9)?,
                        language: row.get(10)?,
                        parser_key: row.get(11)?,
                    },
                    rank: row.get(12)?,
                })
            },
        )?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn list_indexed_units(
        &self,
        path_prefix: Option<&str>,
        language: Option<&str>,
        limit: usize,
        offset: usize,
        include_source: bool,
    ) -> Result<Vec<IndexedItem>> {
        let conn = self.lock()?;
        let path_like = path_prefix.map(|prefix| format!("{prefix}%"));
        let mut stmt = conn.prepare(
            r#"
            SELECT file_path, language, unit_type, name, start_line, end_line, source
            FROM code_units
            WHERE (?1 IS NULL OR file_path LIKE ?1)
              AND (?2 IS NULL OR language = ?2)
            ORDER BY file_path, start_line
            LIMIT ?3 OFFSET ?4
            "#,
        )?;

        let rows = stmt.query_map(
            params![path_like, language, limit as u64, offset as u64],
            |row| {
                Ok(IndexedItem {
                    file_path: row.get(0)?,
                    language: row.get(1)?,
                    unit_type: row.get(2)?,
                    name: row.get(3)?,
                    start_line: row.get(4)?,
                    end_line: row.get(5)?,
                    source: if include_source {
                        Some(row.get(6)?)
                    } else {
                        None
                    },
                })
            },
        )?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub fn stats(&self) -> Result<StoreStats> {
        let conn = self.lock()?;
        let indexed_files = conn.query_row(
            "SELECT COUNT(*) FROM files WHERE status = 'indexed'",
            [],
            |row| row.get(0),
        )?;
        let indexed_units =
            conn.query_row("SELECT COUNT(*) FROM code_units", [], |row| row.get(0))?;
        let embedded_units =
            conn.query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))?;
        let lexical_units =
            conn.query_row("SELECT COUNT(*) FROM code_units_fts", [], |row| row.get(0))?;

        let mut stmt = conn.prepare(
            r#"
            SELECT path, error
            FROM files
            WHERE status = 'error' AND error IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT 10
            "#,
        )?;
        let recent_errors = stmt
            .query_map([], |row| {
                Ok(FileError {
                    path: row.get(0)?,
                    error: row.get(1)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(StoreStats {
            indexed_files,
            indexed_units,
            embedded_units,
            lexical_units,
            recent_errors,
        })
    }

    fn unix_now() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    fn lock(&self) -> Result<MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|_| SmahtiesError::InvalidRequest("database mutex is poisoned".into()))
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn store_opens_and_reports_empty_stats() {
        let dir = tempdir().unwrap();
        let store = Store::open(dir.path().join("smahties.sqlite")).unwrap();

        let stats = store.stats().unwrap();

        assert_eq!(stats.indexed_files, 0);
        assert_eq!(stats.indexed_units, 0);
        assert_eq!(stats.embedded_units, 0);
        assert_eq!(stats.lexical_units, 0);
    }

    #[test]
    fn replace_file_units_updates_lexical_index() {
        let dir = tempdir().unwrap();
        let store = Store::open(dir.path().join("smahties.sqlite")).unwrap();
        let unit = code_unit("src/lib.rs", "unit-1");

        store
            .replace_file_units(
                "src/lib.rs",
                "hash",
                "parser",
                &[unit],
                "model",
                &[vec![1.0]],
            )
            .unwrap();

        let matches = store
            .lexical_search("load_config*", None, Some("text"), 10)
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].unit.id, "unit-1");

        let stats = store.stats().unwrap();
        assert_eq!(stats.indexed_files, 1);
        assert_eq!(stats.indexed_units, 1);
        assert_eq!(stats.embedded_units, 1);
        assert_eq!(stats.lexical_units, 1);
    }

    #[test]
    fn ensure_lexical_index_repairs_existing_empty_fts_rows() {
        let dir = tempdir().unwrap();
        let store = Store::open(dir.path().join("smahties.sqlite")).unwrap();
        let unit = code_unit("src/lib.rs", "unit-1");

        store
            .replace_file_units(
                "src/lib.rs",
                "hash",
                "parser",
                &[unit],
                "model",
                &[vec![1.0]],
            )
            .unwrap();
        store
            .lock()
            .unwrap()
            .execute("DELETE FROM code_units_fts", [])
            .unwrap();
        assert_eq!(store.stats().unwrap().lexical_units, 0);

        store.ensure_lexical_index_current().unwrap();

        let matches = store
            .lexical_search("load_config*", None, Some("text"), 10)
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(store.stats().unwrap().lexical_units, 1);
    }

    #[test]
    fn delete_path_prefix_removes_nested_indexed_files_and_lexical_rows() {
        let dir = tempdir().unwrap();
        let store = Store::open(dir.path().join("smahties.sqlite")).unwrap();
        let unit = code_unit("src/old/lib.rs", "unit-1");

        store
            .replace_file_units(
                "src/old/lib.rs",
                "hash",
                "parser",
                &[unit],
                "model",
                &[vec![1.0]],
            )
            .unwrap();
        store.delete_path_prefix("src/old").unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.indexed_files, 0);
        assert_eq!(stats.indexed_units, 0);
        assert_eq!(stats.embedded_units, 0);
        assert_eq!(stats.lexical_units, 0);
    }

    #[test]
    fn stale_work_can_be_reclaimed_but_owned_updates_are_guarded() {
        let dir = tempdir().unwrap();
        let store = Store::open(dir.path().join("smahties.sqlite")).unwrap();
        let path = dir.path().join("src/lib.rs");

        store.enqueue_work(&path, Priority::Low, false).unwrap();
        let first = store.claim_next_work("owner-1", 300).unwrap().unwrap();
        assert!(store.claim_next_work("owner-2", 300).unwrap().is_none());

        let reclaimed = store.claim_next_work("owner-2", 0).unwrap().unwrap();
        assert_eq!(reclaimed.id, first.id);
        assert!(
            !store
                .fail_work_for_owner(first.id, "owner-1", "stale")
                .unwrap()
        );
        assert!(
            store
                .fail_work_for_owner(first.id, "owner-2", "retry")
                .unwrap()
        );
    }

    #[test]
    fn enqueue_work_updates_existing_pending_item_but_preserves_in_progress_follow_up() {
        let dir = tempdir().unwrap();
        let store = Store::open(dir.path().join("smahties.sqlite")).unwrap();
        let path = dir.path().join("src/lib.rs");

        store.enqueue_work(&path, Priority::Low, false).unwrap();
        store.enqueue_work(&path, Priority::High, false).unwrap();

        let stats = store.queue_stats().unwrap();
        assert_eq!(stats.high_priority, 1);
        assert_eq!(stats.low_priority, 0);

        let claimed = store.claim_next_work("owner", 300).unwrap().unwrap();
        assert_eq!(claimed.priority, Priority::High);
        store.enqueue_work(&path, Priority::Low, false).unwrap();

        let stats = store.queue_stats().unwrap();
        assert_eq!(stats.in_progress, 1);
        assert_eq!(stats.low_priority, 1);
        assert!(store.claim_next_work("other-owner", 300).unwrap().is_some());
    }

    fn code_unit(path: &str, id: &str) -> CodeUnit {
        CodeUnit {
            id: id.into(),
            file_path: path.into(),
            start_line: 1,
            end_line: 1,
            start_byte: 0,
            end_byte: 1,
            unit_type: "file".into(),
            name: Some("load_config".into()),
            source: "fn load_config() {}".into(),
            source_hash: "hash".into(),
            language: "text".into(),
            parser_key: "parser".into(),
        }
    }
}
