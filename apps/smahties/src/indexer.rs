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
    model::{CodeUnit, LeaseStatus, Priority, QueueStats, QueuedWork, SourceFile},
    parser::ParserRegistry,
    scanner::Scanner,
    store::Store,
};

const INDEXER_LEASE_NAME: &str = "indexer";
const INDEXER_LEASE_TTL_SECONDS: i64 = 15;
const WORK_STALE_AFTER_SECONDS: i64 = 300;
const MAX_INDEXER_BATCH_WORK_ITEMS: usize = 128;

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
                ProcessNextOutcome::Processed(processed) => summary.add(processed),
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
        let mut summary = IndexRunSummary::default();
        let mut batch = Vec::new();
        let mut claimed = 0;

        while claimed < MAX_INDEXER_BATCH_WORK_ITEMS {
            let Some(work) = self.claim_next_work()? else {
                break;
            };
            claimed += 1;

            let item = work.item;
            let claim = work.claim;
            match self.prepare_item(&item).await {
                Ok(WorkPreparation::Complete) => {
                    self.complete_claimed_work(&item, claim, &mut summary);
                }
                Ok(WorkPreparation::Embed(prepared)) => {
                    batch.push(PreparedEmbeddingWork {
                        item,
                        claim,
                        source_file: prepared.source_file,
                        parser_key: prepared.parser_key,
                        units: prepared.units,
                    });
                }
                Err(error) => {
                    self.fail_claimed_work(&item, claim, &error, &mut summary);
                }
            }
        }

        if claimed == 0 {
            return Ok(ProcessNextOutcome::Idle);
        }

        self.embed_and_commit_batch(batch, &mut summary).await;
        Ok(ProcessNextOutcome::Processed(summary))
    }

    fn claim_next_work(&self) -> Result<Option<ClaimedWork>> {
        let Some(item) = self
            .inner
            .store
            .claim_next_work(&self.inner.owner, WORK_STALE_AFTER_SECONDS)?
        else {
            return Ok(None);
        };
        let claim = WorkClaim::new(
            Arc::clone(&self.inner.store),
            item.id,
            self.inner.owner.clone(),
        );

        Ok(Some(ClaimedWork { item, claim }))
    }

    async fn prepare_item(&self, item: &QueuedWork) -> Result<WorkPreparation> {
        if item.delete || !item.path.exists() {
            let rel = self.inner.scanner.relative_path(&item.path);
            self.inner.store.delete_path_prefix(&rel)?;
            return Ok(WorkPreparation::Complete);
        }

        if item.path.is_dir() {
            for path in self.inner.scanner.discover_files(&item.path)? {
                self.enqueue_path(path, item.priority).await;
            }
            return Ok(WorkPreparation::Complete);
        }

        if !self.inner.scanner.is_discoverable_file(&item.path)? {
            let rel = self.inner.scanner.relative_path(&item.path);
            self.inner.store.delete_file(&rel)?;
            return Ok(WorkPreparation::Complete);
        }

        let Some(source_file) = self.inner.scanner.read_source(&item.path)? else {
            let rel = self.inner.scanner.relative_path(&item.path);
            self.inner.store.delete_file(&rel)?;
            return Ok(WorkPreparation::Complete);
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
            return Ok(WorkPreparation::Complete);
        }

        let units = self.inner.parser.parse(&source_file)?;
        self.inner
            .store
            .touch_work_claim(item.id, &self.inner.owner)?;

        Ok(WorkPreparation::Embed(PreparedFile {
            source_file,
            parser_key,
            units,
        }))
    }

    async fn embed_and_commit_batch(
        &self,
        batch: Vec<PreparedEmbeddingWork>,
        summary: &mut IndexRunSummary,
    ) {
        if batch.is_empty() {
            return;
        }

        let texts = batch
            .iter()
            .flat_map(|work| work.units.iter())
            .map(|unit| unit.source.clone())
            .collect::<Vec<_>>();

        let embeddings = match self.inner.embedder.embed_texts(&texts).await {
            Ok(embeddings) if embeddings.len() == texts.len() => embeddings,
            Ok(embeddings) => {
                let error = crate::SmahtiesError::InvalidRequest(format!(
                    "embedding response count {} did not match unit count {}",
                    embeddings.len(),
                    texts.len()
                ));
                for work in batch {
                    self.fail_claimed_work(&work.item, work.claim, &error, summary);
                }
                return;
            }
            Err(error) => {
                for work in batch {
                    self.fail_claimed_work(&work.item, work.claim, &error, summary);
                }
                return;
            }
        };

        let mut offset = 0;
        for work in batch {
            let PreparedEmbeddingWork {
                item,
                claim,
                source_file,
                parser_key,
                units,
            } = work;
            let end = offset + units.len();
            let file_embeddings = &embeddings[offset..end];
            offset = end;

            match self.commit_embedded_file(
                &item,
                &source_file,
                &parser_key,
                &units,
                file_embeddings,
            ) {
                Ok(WorkOutcome::Complete) => {
                    self.complete_claimed_work(&item, claim, summary);
                }
                Ok(WorkOutcome::Requeue(reason)) => {
                    self.requeue_claimed_work(&item, claim, &reason, summary);
                }
                Err(error) => {
                    self.fail_claimed_work(&item, claim, &error, summary);
                }
            }
        }
    }

    fn commit_embedded_file(
        &self,
        item: &QueuedWork,
        source_file: &SourceFile,
        parser_key: &str,
        units: &[CodeUnit],
        embeddings: &[Vec<f32>],
    ) -> Result<WorkOutcome> {
        if embeddings.len() != units.len() {
            return Err(crate::SmahtiesError::InvalidRequest(format!(
                "embedding response count {} did not match file unit count {}",
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
            parser_key,
            units,
            self.inner.embedder.model(),
            embeddings,
        )?;
        Ok(WorkOutcome::Complete)
    }

    fn complete_claimed_work(
        &self,
        item: &QueuedWork,
        claim: WorkClaim,
        summary: &mut IndexRunSummary,
    ) {
        if let Err(error) = claim.complete() {
            warn!(work_id = item.id, error = %error, "failed to complete work item");
        }
        summary.completed += 1;
    }

    fn requeue_claimed_work(
        &self,
        item: &QueuedWork,
        claim: WorkClaim,
        reason: &str,
        summary: &mut IndexRunSummary,
    ) {
        if let Err(error) = claim.requeue(reason) {
            warn!(work_id = item.id, error = %error, "failed to requeue work item");
        }
        summary.requeued += 1;
    }

    fn fail_claimed_work(
        &self,
        item: &QueuedWork,
        claim: WorkClaim,
        error: &dyn std::fmt::Display,
        summary: &mut IndexRunSummary,
    ) {
        let error = error.to_string();
        warn!(path = %item.path.display(), error = %error, "failed to index path");
        match claim.requeue(&error) {
            Ok(true) => {
                let rel = self.inner.scanner.relative_path(&item.path);
                if let Err(mark_error) = self.inner.store.mark_error(&rel, &error) {
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
        summary.failed += 1;
    }
}

enum ProcessNextOutcome {
    Idle,
    Processed(IndexRunSummary),
}

impl IndexRunSummary {
    fn add(&mut self, other: Self) {
        self.completed += other.completed;
        self.requeued += other.requeued;
        self.failed += other.failed;
    }
}

enum WorkOutcome {
    Complete,
    Requeue(String),
}

enum WorkPreparation {
    Complete,
    Embed(PreparedFile),
}

struct ClaimedWork {
    item: QueuedWork,
    claim: WorkClaim,
}

struct PreparedFile {
    source_file: SourceFile,
    parser_key: String,
    units: Vec<CodeUnit>,
}

struct PreparedEmbeddingWork {
    item: QueuedWork,
    claim: WorkClaim,
    source_file: SourceFile,
    parser_key: String,
    units: Vec<CodeUnit>,
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
