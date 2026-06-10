use std::{cmp::Ordering, collections::HashMap, sync::Arc};

use serde::Deserialize;

use crate::{
    Result, SmahtiesError,
    embedding::OpenAiEmbedder,
    indexer::Indexer,
    model::{
        CodeUnit, IndexResponse, IndexedListResponse, LexicalMatch, QueryMatch, QueryMatchKind,
        QueryMode, QueryResponse, ServiceStatus,
    },
    store::Store,
    vector::cosine_similarity,
};

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<Store>,
    pub indexer: Indexer,
    pub embedder: OpenAiEmbedder,
}

#[derive(Clone, Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct QueryRequest {
    #[serde(alias = "q")]
    pub query: String,
    pub limit: Option<usize>,
    pub mode: Option<QueryMode>,
    pub path_prefix: Option<String>,
    pub language: Option<String>,
}

#[derive(Clone, Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct IndexRequest {
    pub path: String,
}

#[derive(Clone, Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct ListIndexedRequest {
    pub path_prefix: Option<String>,
    pub language: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub include_source: Option<bool>,
}

pub async fn status(state: &AppState) -> Result<ServiceStatus> {
    state.store.ensure_lexical_index_current()?;
    Ok(ServiceStatus {
        root: state.indexer.root().display().to_string(),
        model: state.indexer.model().to_string(),
        queue: state.indexer.queue_stats().await,
        store: state.store.stats()?,
        lease: state.indexer.lease_status(),
    })
}

pub async fn index_path(state: &AppState, request: IndexRequest) -> Result<IndexResponse> {
    state.indexer.enqueue_requested_path(&request.path).await?;
    Ok(IndexResponse {
        queued: true,
        path: request.path,
    })
}

pub async fn query_code(state: &AppState, request: QueryRequest) -> Result<QueryResponse> {
    let limit = request.limit.unwrap_or(10).clamp(1, 100);
    let mode = request.mode.unwrap_or_default();
    let fts_query = build_fts_query(&request.query);
    let lexical_limit = limit.saturating_mul(20).clamp(50, 500);

    let lexical_matches = if matches!(mode, QueryMode::Keyword | QueryMode::Hybrid) {
        match fts_query {
            Some(ref query) => {
                state.store.ensure_lexical_index_current()?;
                state.store.lexical_search(
                    query,
                    request.path_prefix.as_deref(),
                    request.language.as_deref(),
                    lexical_limit,
                )?
            }
            None => Vec::new(),
        }
    } else {
        Vec::new()
    };

    let semantic_matches = if matches!(mode, QueryMode::Semantic | QueryMode::Hybrid) {
        let query_embedding = state
            .embedder
            .embed_texts(std::slice::from_ref(&request.query))
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| SmahtiesError::InvalidRequest("embedding response was empty".into()))?;

        state
            .store
            .embeddings_for_model(
                state.embedder.model(),
                request.path_prefix.as_deref(),
                request.language.as_deref(),
            )?
            .into_iter()
            .filter_map(|entry| {
                cosine_similarity(&query_embedding, &entry.vector).map(|score| (entry.unit, score))
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let mut matches = merge_matches(mode, semantic_matches, lexical_matches);

    matches.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
    matches.truncate(limit);

    Ok(QueryResponse { matches })
}

fn merge_matches(
    mode: QueryMode,
    semantic_matches: Vec<(CodeUnit, f32)>,
    lexical_matches: Vec<LexicalMatch>,
) -> Vec<QueryMatch> {
    let lexical_scores = normalize_lexical_scores(&lexical_matches);
    let mut candidates = HashMap::<String, CandidateMatch>::new();

    for (unit, semantic_score) in semantic_matches {
        let candidate = candidates
            .entry(unit.id.clone())
            .or_insert_with(|| CandidateMatch::new(unit));
        candidate.semantic_score = Some(semantic_score);
    }

    for (lexical_match, lexical_score) in lexical_matches.into_iter().zip(lexical_scores) {
        let candidate = candidates
            .entry(lexical_match.unit.id.clone())
            .or_insert_with(|| CandidateMatch::new(lexical_match.unit));
        candidate.lexical_score = Some(lexical_score);
    }

    candidates
        .into_values()
        .map(|candidate| candidate.into_query_match(mode))
        .collect()
}

struct CandidateMatch {
    unit: CodeUnit,
    semantic_score: Option<f32>,
    lexical_score: Option<f32>,
}

impl CandidateMatch {
    fn new(unit: CodeUnit) -> Self {
        Self {
            unit,
            semantic_score: None,
            lexical_score: None,
        }
    }

    fn into_query_match(self, mode: QueryMode) -> QueryMatch {
        let semantic_normalized = self
            .semantic_score
            .map(|score| ((score + 1.0) / 2.0).clamp(0.0, 1.0));
        let match_kind = match (self.semantic_score, self.lexical_score) {
            (Some(_), Some(_)) => QueryMatchKind::Hybrid,
            (Some(_), None) => QueryMatchKind::Semantic,
            (None, Some(_)) => QueryMatchKind::Keyword,
            (None, None) => QueryMatchKind::Semantic,
        };
        let score = match mode {
            QueryMode::Semantic => self.semantic_score.unwrap_or(0.0),
            QueryMode::Keyword => self.lexical_score.unwrap_or(0.0),
            QueryMode::Hybrid => match (semantic_normalized, self.lexical_score) {
                (Some(semantic), Some(lexical)) => (0.7 * semantic) + (0.3 * lexical),
                (Some(semantic), None) => semantic,
                (None, Some(lexical)) => lexical * 0.85,
                (None, None) => 0.0,
            },
        };

        QueryMatch {
            score,
            semantic_score: self.semantic_score,
            lexical_score: self.lexical_score,
            match_kind,
            unit: self.unit,
        }
    }
}

fn normalize_lexical_scores(matches: &[LexicalMatch]) -> Vec<f32> {
    if matches.is_empty() {
        return Vec::new();
    }

    let min = matches
        .iter()
        .map(|item| item.rank)
        .fold(f64::INFINITY, f64::min);
    let max = matches
        .iter()
        .map(|item| item.rank)
        .fold(f64::NEG_INFINITY, f64::max);
    if !min.is_finite() || !max.is_finite() || (max - min).abs() < f64::EPSILON {
        return vec![1.0; matches.len()];
    }

    matches
        .iter()
        .map(|item| (1.0 - ((item.rank - min) / (max - min))) as f32)
        .map(|score| score.clamp(0.0, 1.0))
        .collect()
}

fn build_fts_query(query: &str) -> Option<String> {
    let mut tokens = Vec::<String>::new();
    for raw_token in query
        .split(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
        .filter(|token| token.len() >= 2)
    {
        let token = raw_token.to_lowercase();
        if !tokens.iter().any(|existing| existing == &token) {
            tokens.push(token);
        }
        if tokens.len() >= 12 {
            break;
        }
    }

    (!tokens.is_empty()).then(|| {
        tokens
            .into_iter()
            .map(|token| format!("{token}*"))
            .collect::<Vec<_>>()
            .join(" OR ")
    })
}

pub fn list_indexed(state: &AppState, request: ListIndexedRequest) -> Result<IndexedListResponse> {
    let limit = request.limit.unwrap_or(50).clamp(1, 200);
    let offset = request.offset.unwrap_or(0);
    let include_source = request.include_source.unwrap_or(false) && limit <= 20;
    let items = state.store.list_indexed_units(
        request.path_prefix.as_deref(),
        request.language.as_deref(),
        limit,
        offset,
        include_source,
    )?;

    Ok(IndexedListResponse {
        items,
        limit,
        offset,
    })
}

#[cfg(test)]
mod tests {
    use crate::model::{CodeUnit, LexicalMatch, QueryMatchKind, QueryMode};

    use super::{build_fts_query, merge_matches};

    #[test]
    fn fts_query_uses_safe_prefix_tokens() {
        assert_eq!(
            build_fts_query("Find config loader, config loader!"),
            Some("find* OR config* OR loader*".into())
        );
        assert_eq!(build_fts_query("! ? a"), None);
    }

    #[test]
    fn hybrid_merge_combines_semantic_and_lexical_scores() {
        let unit = code_unit("one");
        let matches = merge_matches(
            QueryMode::Hybrid,
            vec![(unit.clone(), 0.8)],
            vec![LexicalMatch { unit, rank: -2.0 }],
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].match_kind, QueryMatchKind::Hybrid);
        assert_eq!(matches[0].semantic_score, Some(0.8));
        assert_eq!(matches[0].lexical_score, Some(1.0));
        assert!(matches[0].score > 0.9);
    }

    fn code_unit(id: &str) -> CodeUnit {
        CodeUnit {
            id: id.into(),
            file_path: "src/lib.rs".into(),
            start_line: 1,
            end_line: 1,
            start_byte: 0,
            end_byte: 1,
            unit_type: "function".into(),
            name: Some("load_config".into()),
            source: "fn load_config() {}".into(),
            source_hash: "hash".into(),
            language: "rust".into(),
            parser_key: "parser".into(),
        }
    }
}
