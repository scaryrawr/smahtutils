use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, rmcp::schemars::JsonSchema, Serialize)]
pub struct CodeUnit {
    pub id: String,
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub start_byte: usize,
    pub end_byte: usize,
    pub unit_type: String,
    pub name: Option<String>,
    pub source: String,
    pub source_hash: String,
    pub language: String,
    pub parser_key: String,
}

#[derive(Clone, Debug)]
pub struct SourceFile {
    pub absolute_path: std::path::PathBuf,
    pub relative_path: String,
    pub contents: String,
    pub hash: String,
}

#[derive(Clone, Debug)]
pub struct StoredEmbeddingCandidate {
    pub unit_id: String,
    pub vector: Vec<f32>,
    pub norm: f32,
}

#[derive(Clone, Debug)]
pub struct LexicalMatch {
    pub unit: CodeUnit,
    pub rank: f64,
}

#[derive(Clone, Debug, rmcp::schemars::JsonSchema, Serialize)]
pub struct StoreStats {
    pub indexed_files: u64,
    pub indexed_units: u64,
    pub embedded_units: u64,
    pub lexical_units: u64,
    pub recent_errors: Vec<FileError>,
}

#[derive(Clone, Debug, rmcp::schemars::JsonSchema, Serialize)]
pub struct FileError {
    pub path: String,
    pub error: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, rmcp::schemars::JsonSchema, Serialize)]
pub enum Priority {
    High,
    Low,
}

impl Priority {
    pub fn as_i64(self) -> i64 {
        match self {
            Self::High => 100,
            Self::Low => 0,
        }
    }

    pub fn from_i64(value: i64) -> Self {
        if value >= 100 { Self::High } else { Self::Low }
    }
}

#[derive(Clone, Copy, Debug, rmcp::schemars::JsonSchema, Serialize)]
pub struct QueueStats {
    pub high_priority: u64,
    pub low_priority: u64,
    pub in_progress: u64,
}

#[derive(Clone, Debug)]
pub struct QueuedWork {
    pub id: i64,
    pub path: std::path::PathBuf,
    pub priority: Priority,
    pub delete: bool,
}

#[derive(Clone, Debug, rmcp::schemars::JsonSchema, Serialize)]
pub struct LeaseStatus {
    pub owner: Option<String>,
    pub expires_at_unix: Option<i64>,
    pub held_by_this_process: bool,
}

#[derive(Clone, Debug, rmcp::schemars::JsonSchema, Serialize)]
pub struct ServiceStatus {
    pub root: String,
    pub repository_root: Option<String>,
    pub runtime_root: String,
    pub scope_prefix: Option<String>,
    pub auto_indexing_enabled: bool,
    pub model: String,
    pub queue: QueueStats,
    pub store: StoreStats,
    pub lease: LeaseStatus,
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, rmcp::schemars::JsonSchema, Serialize,
)]
#[serde(rename_all = "snake_case")]
pub enum QueryMode {
    Semantic,
    Keyword,
    #[default]
    Hybrid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, rmcp::schemars::JsonSchema, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryMatchKind {
    Semantic,
    Keyword,
    Hybrid,
}

#[derive(Clone, Debug, rmcp::schemars::JsonSchema, Serialize)]
pub struct QueryMatch {
    pub score: f32,
    pub semantic_score: Option<f32>,
    pub lexical_score: Option<f32>,
    pub match_kind: QueryMatchKind,
    pub unit: CodeUnit,
}

#[derive(Clone, Debug, rmcp::schemars::JsonSchema, Serialize)]
pub struct QueryResponse {
    pub matches: Vec<QueryMatch>,
}

#[derive(Clone, Debug, rmcp::schemars::JsonSchema, Serialize)]
pub struct IndexResponse {
    pub queued: bool,
    pub path: String,
}

#[derive(Clone, Debug, rmcp::schemars::JsonSchema, Serialize)]
pub struct IndexedItem {
    pub file_path: String,
    pub language: String,
    pub unit_type: String,
    pub name: Option<String>,
    pub start_line: u32,
    pub end_line: u32,
    pub source: Option<String>,
}

#[derive(Clone, Debug, rmcp::schemars::JsonSchema, Serialize)]
pub struct IndexedListResponse {
    pub items: Vec<IndexedItem>,
    pub limit: usize,
    pub offset: usize,
}
