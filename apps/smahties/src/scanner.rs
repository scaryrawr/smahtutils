use std::{
    ffi::OsStr,
    fs,
    path::{Component, Path, PathBuf},
};

use ignore::WalkBuilder;
use sha2::{Digest, Sha256};

use crate::{Result, model::SourceFile};

const DEFAULT_MAX_FILE_BYTES: u64 = 512 * 1024;
pub const EXCLUDED_DIR_NAMES: &[&str] = &[
    ".git",
    ".smahties",
    "target",
    "node_modules",
    ".next",
    ".turbo",
];
pub const EXCLUDED_FILE_NAMES: &[&str] =
    &[".gitignore", ".ignore", ".gitattributes", ".gitmodules"];

#[derive(Clone, Debug)]
pub struct Scanner {
    root: PathBuf,
    max_file_bytes: u64,
}

impl Scanner {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn discover_files(&self, path: &Path) -> Result<Vec<PathBuf>> {
        if is_excluded_path(&self.root, path) {
            return Ok(Vec::new());
        }

        if path.is_file() {
            return Ok(self
                .discover_files(path.parent().unwrap_or_else(|| self.root()))
                .unwrap_or_default()
                .into_iter()
                .filter(|candidate| candidate == path)
                .collect());
        }

        if !path.exists() {
            return Ok(Vec::new());
        }

        let mut builder = WalkBuilder::new(path);
        builder
            .hidden(false)
            .parents(true)
            .ignore(true)
            .git_ignore(true)
            .git_exclude(true)
            .follow_links(false);

        let mut files = Vec::new();
        for entry in builder.build() {
            let Ok(entry) = entry else {
                continue;
            };
            let entry_path = entry.path();
            if entry
                .file_type()
                .is_some_and(|file_type| file_type.is_file())
                && self.is_indexable_path(entry_path)
            {
                files.push(entry_path.to_path_buf());
            }
        }

        Ok(files)
    }

    pub fn read_source(&self, path: &Path) -> Result<Option<SourceFile>> {
        if !self.is_indexable_path(path) {
            return Ok(None);
        }

        let bytes = fs::read(path)?;
        if bytes.contains(&0) {
            return Ok(None);
        }

        let Ok(contents) = String::from_utf8(bytes) else {
            return Ok(None);
        };
        let hash = sha256_hex(contents.as_bytes());

        Ok(Some(SourceFile {
            absolute_path: path.to_path_buf(),
            relative_path: self.relative_path(path),
            contents,
            hash,
        }))
    }

    pub fn is_discoverable_file(&self, path: &Path) -> Result<bool> {
        Ok(self
            .discover_files(path)?
            .into_iter()
            .any(|candidate| candidate == path))
    }

    pub fn relative_path(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    }

    pub fn resolve_existing_under_root(&self, requested: &str) -> Result<PathBuf> {
        let path = Path::new(requested);
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };

        let root = self.root.canonicalize()?;
        let canonical = absolute.canonicalize()?;
        if !canonical.starts_with(root) {
            return Err(crate::SmahtiesError::InvalidRequest(format!(
                "path is outside the indexed root: {}",
                absolute.display()
            )));
        }

        Ok(canonical)
    }

    fn is_indexable_path(&self, path: &Path) -> bool {
        if !path.starts_with(&self.root) || is_excluded_path(&self.root, path) {
            return false;
        }

        let Ok(metadata) = fs::metadata(path) else {
            return false;
        };

        metadata.is_file() && metadata.len() <= self.max_file_bytes
    }
}

pub fn ensure_state_dir(root: &Path) -> Result<PathBuf> {
    let state_dir = root.join(".smahties");
    fs::create_dir_all(&state_dir)?;
    fs::write(state_dir.join(".gitignore"), "*\n")?;
    Ok(state_dir)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn is_excluded_path(root: &Path, path: &Path) -> bool {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .any(|component| {
            matches!(
                component,
                Component::Normal(name) if EXCLUDED_DIR_NAMES
                    .iter()
                    .chain(EXCLUDED_FILE_NAMES.iter())
                    .any(|excluded| name == OsStr::new(excluded))
            )
        })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn state_dir_contains_gitignore() {
        let dir = tempdir().unwrap();

        let state_dir = ensure_state_dir(dir.path()).unwrap();

        assert_eq!(
            fs::read_to_string(state_dir.join(".gitignore")).unwrap(),
            "*\n"
        );
    }

    #[test]
    fn discover_files_skips_target_even_without_gitignore() {
        let dir = tempdir().unwrap();
        let src_dir = dir.path().join("src");
        let target_dir = dir.path().join("target/debug/.fingerprint/package");
        fs::create_dir_all(&src_dir).unwrap();
        fs::create_dir_all(&target_dir).unwrap();
        let source_path = src_dir.join("lib.rs");
        let target_path = target_dir.join("invoked.timestamp");
        fs::write(&source_path, "fn main() {}\n").unwrap();
        fs::write(
            &target_path,
            "This file has an mtime of when this was started.",
        )
        .unwrap();

        let scanner = Scanner::new(dir.path().to_path_buf());
        let files = scanner.discover_files(dir.path()).unwrap();

        assert!(files.contains(&source_path));
        assert!(!files.contains(&target_path));
        assert!(!scanner.is_discoverable_file(&target_path).unwrap());
    }

    #[test]
    fn discover_files_skips_gitignore_control_files() {
        let dir = tempdir().unwrap();
        let source_path = dir.path().join("src.rs");
        let gitignore_path = dir.path().join(".gitignore");
        fs::write(&source_path, "fn main() {}\n").unwrap();
        fs::write(&gitignore_path, "target\n").unwrap();

        let scanner = Scanner::new(dir.path().to_path_buf());
        let files = scanner.discover_files(dir.path()).unwrap();

        assert!(files.contains(&source_path));
        assert!(!files.contains(&gitignore_path));
        assert!(!scanner.is_discoverable_file(&gitignore_path).unwrap());
    }
}
