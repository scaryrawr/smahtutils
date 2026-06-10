use std::path::{Component, Path, PathBuf};

use crate::{Result, SmahtiesError, scanner::ensure_state_dir};

#[derive(Clone, Debug)]
pub struct RuntimeContext {
    repository_root: Option<PathBuf>,
    storage_root: PathBuf,
    runtime_root: PathBuf,
    scope_prefix: Option<String>,
    auto_indexing_enabled: bool,
}

impl RuntimeContext {
    pub fn resolve(runtime_root: PathBuf) -> Result<Self> {
        let runtime_root = runtime_root.canonicalize()?;
        let repository_root = find_git_root(&runtime_root);
        let storage_root = repository_root
            .clone()
            .unwrap_or_else(|| runtime_root.clone());
        let scope_prefix = repository_root
            .as_deref()
            .map(|root| relative_path_string(root, &runtime_root))
            .transpose()?
            .flatten();
        let auto_indexing_enabled = repository_root.is_some();

        Ok(Self {
            repository_root,
            storage_root,
            runtime_root,
            scope_prefix,
            auto_indexing_enabled,
        })
    }

    pub fn repository_root(&self) -> Option<&Path> {
        self.repository_root.as_deref()
    }

    pub fn storage_root(&self) -> &Path {
        &self.storage_root
    }

    pub fn runtime_root(&self) -> &Path {
        &self.runtime_root
    }

    pub fn scope_prefix(&self) -> Option<&str> {
        self.scope_prefix.as_deref()
    }

    pub fn auto_indexing_enabled(&self) -> bool {
        self.auto_indexing_enabled
    }

    pub fn auto_index_root(&self) -> Option<&Path> {
        self.auto_indexing_enabled.then_some(self.runtime_root())
    }

    pub fn state_dir(&self) -> Result<PathBuf> {
        ensure_state_dir(&self.storage_root)
    }

    pub fn scoped_path_prefix(&self, requested: Option<&str>) -> Result<Option<String>> {
        let requested = requested
            .map(normalize_relative_prefix)
            .transpose()?
            .flatten();
        match (self.scope_prefix(), requested) {
            (Some(scope), Some(prefix)) if path_prefix_contains(scope, &prefix) => Ok(Some(prefix)),
            (Some(scope), Some(prefix)) => Ok(Some(join_path_prefix(scope, &prefix))),
            (Some(scope), None) => Ok(Some(scope.to_string())),
            (None, prefix) => Ok(prefix),
        }
    }
}

fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut current = if start.is_file() {
        start.parent()?.to_path_buf()
    } else {
        start.to_path_buf()
    };

    loop {
        let git_marker = current.join(".git");
        if git_marker.is_dir() || git_marker.is_file() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

fn relative_path_string(root: &Path, path: &Path) -> Result<Option<String>> {
    let relative = path.strip_prefix(root).map_err(|_| {
        SmahtiesError::InvalidRequest(format!(
            "runtime root is outside repository root: {}",
            path.display()
        ))
    })?;
    let value = relative.to_string_lossy().replace('\\', "/");
    Ok((!value.is_empty()).then_some(value))
}

fn normalize_relative_prefix(value: &str) -> Result<Option<String>> {
    let normalized = value.replace('\\', "/");
    let path = Path::new(&normalized);
    if path.is_absolute() {
        return Err(SmahtiesError::InvalidRequest(format!(
            "path_prefix must be relative to the active smahties scope: {value}"
        )));
    }

    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(SmahtiesError::InvalidRequest(format!(
                    "path_prefix must not escape the active smahties scope: {value}"
                )));
            }
        }
    }

    Ok((!parts.is_empty()).then(|| parts.join("/")))
}

fn path_prefix_contains(scope: &str, prefix: &str) -> bool {
    prefix == scope
        || prefix
            .strip_prefix(scope)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn join_path_prefix(scope: &str, prefix: &str) -> String {
    if scope.is_empty() {
        prefix.to_string()
    } else {
        format!("{scope}/{prefix}")
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn repo_context_uses_repository_state_and_runtime_scope() {
        let dir = tempdir().unwrap();
        let repo = dir.path().join("repo");
        let app = repo.join("apps/api");
        fs::create_dir_all(repo.join(".git")).unwrap();
        fs::create_dir_all(&app).unwrap();

        let context = RuntimeContext::resolve(app.clone()).unwrap();

        let repo = repo.canonicalize().unwrap();
        let app = app.canonicalize().unwrap();

        assert_eq!(context.repository_root(), Some(repo.as_path()));
        assert_eq!(context.storage_root(), repo.as_path());
        assert_eq!(context.runtime_root(), app.as_path());
        assert_eq!(context.scope_prefix(), Some("apps/api"));
        assert!(context.auto_indexing_enabled());
        assert_eq!(context.auto_index_root(), Some(app.as_path()));
        assert_eq!(context.state_dir().unwrap(), repo.join(".smahties"));
    }

    #[test]
    fn repo_context_accepts_git_file_markers_for_worktrees() {
        let dir = tempdir().unwrap();
        let repo = dir.path().join("worktree");
        let app = repo.join("packages/app");
        fs::create_dir_all(&app).unwrap();
        fs::write(repo.join(".git"), "gitdir: ../main/.git/worktrees/app\n").unwrap();

        let context = RuntimeContext::resolve(app).unwrap();

        let repo = repo.canonicalize().unwrap();

        assert_eq!(context.repository_root(), Some(repo.as_path()));
        assert_eq!(context.scope_prefix(), Some("packages/app"));
    }

    #[test]
    fn non_repo_context_disables_auto_indexing() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("loose");
        fs::create_dir_all(&root).unwrap();

        let context = RuntimeContext::resolve(root.clone()).unwrap();

        assert_eq!(context.repository_root(), None);
        let root = root.canonicalize().unwrap();

        assert_eq!(context.storage_root(), root.as_path());
        assert_eq!(context.scope_prefix(), None);
        assert!(!context.auto_indexing_enabled());
        assert_eq!(context.auto_index_root(), None);
    }

    #[test]
    fn scoped_path_prefixes_stay_under_runtime_scope() {
        let dir = tempdir().unwrap();
        let repo = dir.path().join("repo");
        let app = repo.join("apps/api");
        fs::create_dir_all(repo.join(".git")).unwrap();
        fs::create_dir_all(&app).unwrap();
        let context = RuntimeContext::resolve(app).unwrap();

        assert_eq!(
            context.scoped_path_prefix(None).unwrap(),
            Some("apps/api".into())
        );
        assert_eq!(
            context.scoped_path_prefix(Some("src")).unwrap(),
            Some("apps/api/src".into())
        );
        assert_eq!(
            context.scoped_path_prefix(Some("apps/api/src")).unwrap(),
            Some("apps/api/src".into())
        );
        assert_eq!(
            context
                .scoped_path_prefix(Some("../other"))
                .unwrap_err()
                .to_string(),
            "path_prefix must not escape the active smahties scope: ../other"
        );
    }
}
