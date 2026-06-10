use std::{
    future::Future,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use tokio::sync::Notify;
use tracing::{debug, warn};

use crate::{
    Result,
    embedding::OpenAiEmbedder,
    model::{LeaseStatus, Priority, QueueStats, QueuedWork},
    parser::ParserRegistry,
    scanner::Scanner,
    store::Store,
};

const INDEXER_LEASE_NAME: &str = "indexer";
const INDEXER_LEASE_TTL_SECONDS: i64 = 15;
const WORK_STALE_AFTER_SECONDS: i64 = 300;

#[derive(Clone)]
pub struct Indexer {
    inner: Arc<IndexerInner>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct IndexRunSummary {
    pub completed: u64,
    pub requeued: u64,
    pub failed: u64,
}

#[derive(Clone, Copy, Debug)]
pub enum IndexRunOutcome {
    Complete(IndexRunSummary),
    Interrupted(IndexRunSummary),
}

struct IndexerInner {
    scanner: Scanner,
    parser: ParserRegistry,
    store: Arc<Store>,
    embedder: OpenAiEmbedder,
    notify: Notify,
    owner: String,
}

impl Indexer {
    pub fn new(
        scanner: Scanner,
        parser: ParserRegistry,
        store: Arc<Store>,
        embedder: OpenAiEmbedder,
    ) -> Self {
        Self {
            inner: Arc::new(IndexerInner {
                scanner,
                parser,
                store,
                embedder,
                notify: Notify::new(),
                owner: format!("{}:{}", std::process::id(), unix_now()),
            }),
        }
    }

    pub fn root(&self) -> &Path {
        self.inner.scanner.root()
    }

    pub fn model(&self) -> &str {
        self.inner.embedder.model()
    }

    pub async fn enqueue_requested_path(&self, requested: &str) -> Result<()> {
        let path = self.inner.scanner.resolve_existing_under_root(requested)?;
        self.enqueue_path(path, Priority::High).await;
        Ok(())
    }

    pub async fn enqueue_requested_path_under(&self, requested: &str, base: &Path) -> Result<()> {
        let path = self.inner.scanner.resolve_existing_under(base, requested)?;
        self.enqueue_path(path, Priority::High).await;
        Ok(())
    }

    pub async fn enqueue_path(&self, path: PathBuf, priority: Priority) {
        if let Err(error) = self.inner.store.enqueue_work(&path, priority, false) {
            warn!(path = %path.display(), error = %error, "failed to enqueue indexing work");
        }
        self.inner.notify.notify_one();
    }

    pub async fn enqueue_delete(&self, path: PathBuf) {
        if let Err(error) = self.inner.store.enqueue_work(&path, Priority::High, true) {
            warn!(path = %path.display(), error = %error, "failed to enqueue delete work");
        }
        self.inner.notify.notify_one();
    }

    pub fn spawn_worker(&self) {
        let indexer = self.clone();
        tokio::spawn(async move { indexer.worker_loop().await });
    }

    pub async fn run_until_idle_or_interrupt(
        &self,
        interrupt: impl Future<Output = std::io::Result<()>>,
    ) -> Result<IndexRunOutcome> {
        tokio::pin!(interrupt);
        let mut summary = IndexRunSummary::default();

        loop {
            if !self.acquire_lease()? {
                tokio::select! {
                    interrupt_result = &mut interrupt => {
                        interrupt_result?;
                        return Ok(IndexRunOutcome::Interrupted(summary));
                    }
                    () = tokio::time::sleep(Duration::from_secs(2)) => {}
                }
                continue;
            }

            let next_work = self.process_next_work();
            tokio::pin!(next_work);
            let outcome = tokio::select! {
                interrupt_result = &mut interrupt => {
                    interrupt_result?;
                    return Ok(IndexRunOutcome::Interrupted(summary));
                }
                outcome = &mut next_work => outcome?,
            };

            match outcome {
                ProcessNextOutcome::Idle => return Ok(IndexRunOutcome::Complete(summary)),
                ProcessNextOutcome::Completed => summary.completed += 1,
                ProcessNextOutcome::Requeued => summary.requeued += 1,
                ProcessNextOutcome::Failed => summary.failed += 1,
            }
        }
    }

    pub async fn queue_stats(&self) -> QueueStats {
        self.inner.store.queue_stats().unwrap_or(QueueStats {
            high_priority: 0,
            low_priority: 0,
            in_progress: 0,
        })
    }

    pub fn lease_status(&self) -> LeaseStatus {
        self.inner
            .store
            .lease_status(INDEXER_LEASE_NAME, &self.inner.owner)
            .unwrap_or(LeaseStatus {
                owner: None,
                expires_at_unix: None,
                held_by_this_process: false,
            })
    }

    async fn worker_loop(self) {
        loop {
            let has_lease = self.acquire_lease().unwrap_or(false);
            if !has_lease {
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }

            match self.process_next_work().await {
                Ok(ProcessNextOutcome::Idle) => {
                    tokio::select! {
                        () = self.inner.notify.notified() => {}
                        () = tokio::time::sleep(Duration::from_secs(1)) => {}
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    warn!(error = %error, "failed to claim indexing work");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    fn acquire_lease(&self) -> Result<bool> {
        self.inner.store.acquire_lease(
            INDEXER_LEASE_NAME,
            &self.inner.owner,
            INDEXER_LEASE_TTL_SECONDS,
        )
    }

    async fn process_next_work(&self) -> Result<ProcessNextOutcome> {
        let Some(item) = self
            .inner
            .store
            .claim_next_work(&self.inner.owner, WORK_STALE_AFTER_SECONDS)?
        else {
            return Ok(ProcessNextOutcome::Idle);
        };
        let claim = WorkClaim::new(
            Arc::clone(&self.inner.store),
            item.id,
            self.inner.owner.clone(),
        );

        match self.handle_item(item.clone()).await {
            Ok(WorkOutcome::Complete) => {
                if let Err(error) = claim.complete() {
                    warn!(work_id = item.id, error = %error, "failed to complete work item");
                }
                Ok(ProcessNextOutcome::Completed)
            }
            Ok(WorkOutcome::Requeue(reason)) => {
                if let Err(error) = claim.requeue(&reason) {
                    warn!(work_id = item.id, error = %error, "failed to requeue work item");
                }
                Ok(ProcessNextOutcome::Requeued)
            }
            Err(error) => {
                warn!(path = %item.path.display(), error = %error, "failed to index path");
                match claim.requeue(&error.to_string()) {
                    Ok(true) => {
                        let rel = self.inner.scanner.relative_path(&item.path);
                        if let Err(mark_error) =
                            self.inner.store.mark_error(&rel, &error.to_string())
                        {
                            warn!(path = %rel, error = %mark_error, "failed to persist indexing error");
                        }
                    }
                    Ok(false) => {
                        debug!(
                            work_id = item.id,
                            "skipping error persistence for work item no longer owned by this process"
                        );
                    }
                    Err(fail_error) => {
                        warn!(work_id = item.id, error = %fail_error, "failed to release failed work item");
                    }
                }
                Ok(ProcessNextOutcome::Failed)
            }
        }
    }

    async fn handle_item(&self, item: QueuedWork) -> Result<WorkOutcome> {
        if item.delete || !item.path.exists() {
            let rel = self.inner.scanner.relative_path(&item.path);
            self.inner.store.delete_path_prefix(&rel)?;
            return Ok(WorkOutcome::Complete);
        }

        if item.path.is_dir() {
            for path in self.inner.scanner.discover_files(&item.path)? {
                self.enqueue_path(path, item.priority).await;
            }
            return Ok(WorkOutcome::Complete);
        }

        if !self.inner.scanner.is_discoverable_file(&item.path)? {
            let rel = self.inner.scanner.relative_path(&item.path);
            self.inner.store.delete_file(&rel)?;
            return Ok(WorkOutcome::Complete);
        }

        let Some(source_file) = self.inner.scanner.read_source(&item.path)? else {
            let rel = self.inner.scanner.relative_path(&item.path);
            self.inner.store.delete_file(&rel)?;
            return Ok(WorkOutcome::Complete);
        };

        let parser_key = self
            .inner
            .parser
            .cache_key_for_path(&source_file.absolute_path);
        if self.inner.store.file_complete_for_model(
            &source_file.relative_path,
            &source_file.hash,
            &parser_key,
            self.inner.embedder.model(),
        )? {
            debug!(path = %source_file.relative_path, "file already indexed");
            return Ok(WorkOutcome::Complete);
        }

        let units = self.inner.parser.parse(&source_file)?;
        self.inner
            .store
            .touch_work_claim(item.id, &self.inner.owner)?;
        let texts = units
            .iter()
            .map(|unit| unit.source.clone())
            .collect::<Vec<_>>();
        let embeddings = self.inner.embedder.embed_texts(&texts).await?;
        if embeddings.len() != units.len() {
            return Err(crate::SmahtiesError::InvalidRequest(format!(
                "embedding response count {} did not match unit count {}",
                embeddings.len(),
                units.len()
            )));
        }

        let Some(current_source) = self.inner.scanner.read_source(&item.path)? else {
            let rel = self.inner.scanner.relative_path(&item.path);
            self.inner.store.delete_file(&rel)?;
            return Ok(WorkOutcome::Complete);
        };
        if current_source.hash != source_file.hash {
            return Ok(WorkOutcome::Requeue("source changed while indexing".into()));
        }
        if !self.inner.store.acquire_lease(
            INDEXER_LEASE_NAME,
            &self.inner.owner,
            INDEXER_LEASE_TTL_SECONDS,
        )? {
            return Ok(WorkOutcome::Requeue(
                "indexer lease expired before commit".into(),
            ));
        }
        self.inner
            .store
            .touch_work_claim(item.id, &self.inner.owner)?;
        self.inner.store.replace_file_units(
            &source_file.relative_path,
            &source_file.hash,
            &parser_key,
            &units,
            self.inner.embedder.model(),
            &embeddings,
        )?;
        Ok(WorkOutcome::Complete)
    }
}

enum ProcessNextOutcome {
    Idle,
    Completed,
    Requeued,
    Failed,
}

enum WorkOutcome {
    Complete,
    Requeue(String),
}

struct WorkClaim {
    store: Arc<Store>,
    id: i64,
    owner: String,
    active: bool,
}

impl WorkClaim {
    fn new(store: Arc<Store>, id: i64, owner: String) -> Self {
        Self {
            store,
            id,
            owner,
            active: true,
        }
    }

    fn complete(mut self) -> Result<()> {
        self.store.complete_work_for_owner(self.id, &self.owner)?;
        self.active = false;
        Ok(())
    }

    fn requeue(mut self, reason: &str) -> Result<bool> {
        let changed = self
            .store
            .fail_work_for_owner(self.id, &self.owner, reason)?;
        self.active = false;
        Ok(changed)
    }
}

impl Drop for WorkClaim {
    fn drop(&mut self) {
        if self.active {
            let _ = self
                .store
                .fail_work_for_owner(self.id, &self.owner, "indexing interrupted");
        }
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
