use std::{path::Path, time::Duration};

use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::runtime::Handle;

use crate::{
    Result,
    indexer::Indexer,
    model::Priority,
    scanner::{self},
};

pub fn start(root: &Path, indexer: Indexer) -> Result<RecommendedWatcher> {
    let runtime = Handle::current();
    let watched_root = root.to_path_buf();
    let event_root = watched_root.clone();
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        let Ok(event) = event else {
            return;
        };
        let delete = matches!(event.kind, EventKind::Remove(_));
        for path in event.paths {
            if scanner::is_excluded_path(&event_root, &path) {
                continue;
            }
            let indexer = indexer.clone();
            runtime.spawn(async move {
                tokio::time::sleep(Duration::from_millis(200)).await;
                if delete {
                    indexer.enqueue_delete(path).await;
                } else {
                    indexer.enqueue_path(path, Priority::High).await;
                }
            });
        }
    })?;
    watcher.watch(&watched_root, RecursiveMode::Recursive)?;
    Ok(watcher)
}
