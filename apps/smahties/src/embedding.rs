use async_openai::{
    Client,
    config::OpenAIConfig,
    types::embeddings::{CreateEmbeddingRequestArgs, Embedding},
};
use std::ops::Range;

use crate::Result;

const DEFAULT_MAX_EMBEDDING_BATCH_INPUTS: usize = 128;
const DEFAULT_MAX_EMBEDDING_BATCH_BYTES: usize = 256 * 1024;

#[derive(Clone)]
pub struct OpenAiEmbedder {
    client: Client<OpenAIConfig>,
    model: String,
    batch_limits: EmbeddingBatchLimits,
}

#[derive(Clone, Copy, Debug)]
struct EmbeddingBatchLimits {
    max_inputs: usize,
    max_request_bytes: usize,
}

impl Default for EmbeddingBatchLimits {
    fn default() -> Self {
        Self {
            max_inputs: DEFAULT_MAX_EMBEDDING_BATCH_INPUTS,
            max_request_bytes: DEFAULT_MAX_EMBEDDING_BATCH_BYTES,
        }
    }
}

impl OpenAiEmbedder {
    pub fn new(base_url: &str, model: String) -> Self {
        let config = OpenAIConfig::new().with_api_base(base_url);
        Self {
            client: Client::with_config(config),
            model,
            batch_limits: EmbeddingBatchLimits::default(),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut embeddings = Vec::with_capacity(texts.len());
        for range in embedding_batch_ranges(texts, self.batch_limits)? {
            let mut batch_embeddings = self.embed_text_batch(&texts[range]).await?;
            embeddings.append(&mut batch_embeddings);
        }

        Ok(embeddings)
    }

    async fn embed_text_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(texts.to_vec())
            .build()?;
        let response = self.client.embeddings().create(request).await?;

        let embeddings = ordered_embeddings(response.data)?;
        if embeddings.len() != texts.len() {
            return Err(crate::SmahtiesError::InvalidRequest(format!(
                "embedding response count {} did not match batch input count {}",
                embeddings.len(),
                texts.len()
            )));
        }
        Ok(embeddings)
    }
}

fn embedding_batch_ranges(
    texts: &[String],
    limits: EmbeddingBatchLimits,
) -> Result<Vec<Range<usize>>> {
    if limits.max_inputs == 0 || limits.max_request_bytes == 0 {
        return Err(crate::SmahtiesError::InvalidRequest(
            "embedding batch limits must be greater than zero".into(),
        ));
    }

    let mut ranges = Vec::new();
    let mut start = 0;
    let mut request_bytes = 0;

    for (index, text) in texts.iter().enumerate() {
        let batch_inputs = index - start;
        let would_exceed_inputs = batch_inputs >= limits.max_inputs;
        let would_exceed_bytes =
            batch_inputs > 0 && request_bytes + text.len() > limits.max_request_bytes;

        if would_exceed_inputs || would_exceed_bytes {
            ranges.push(start..index);
            start = index;
            request_bytes = 0;
        }

        request_bytes += text.len();
    }

    if start < texts.len() {
        ranges.push(start..texts.len());
    }

    Ok(ranges)
}

fn ordered_embeddings(mut embeddings: Vec<Embedding>) -> Result<Vec<Vec<f32>>> {
    embeddings.sort_by_key(|embedding| embedding.index);
    for (expected, embedding) in embeddings.iter().enumerate() {
        if embedding.index as usize != expected {
            return Err(crate::SmahtiesError::InvalidRequest(format!(
                "embedding response index {} did not match expected index {expected}",
                embedding.index
            )));
        }
    }

    Ok(embeddings
        .into_iter()
        .map(|embedding| embedding.embedding)
        .collect())
}

#[cfg(test)]
mod tests {
    use async_openai::types::embeddings::Embedding;

    use super::{EmbeddingBatchLimits, embedding_batch_ranges, ordered_embeddings};

    #[test]
    fn embeddings_are_returned_in_request_order() {
        let embeddings = ordered_embeddings(vec![
            Embedding {
                index: 1,
                object: "embedding".into(),
                embedding: vec![1.0],
            },
            Embedding {
                index: 0,
                object: "embedding".into(),
                embedding: vec![0.0],
            },
        ])
        .unwrap();

        assert_eq!(embeddings, vec![vec![0.0], vec![1.0]]);
    }

    #[test]
    fn embeddings_reject_missing_response_indexes() {
        let error = ordered_embeddings(vec![Embedding {
            index: 1,
            object: "embedding".into(),
            embedding: vec![1.0],
        }])
        .unwrap_err();

        assert!(error.to_string().contains("expected index 0"));
    }

    #[test]
    fn embedding_batches_are_limited_by_input_count() {
        let texts = ["one", "two", "three", "four", "five"].map(String::from);
        let ranges = embedding_batch_ranges(
            &texts,
            EmbeddingBatchLimits {
                max_inputs: 2,
                max_request_bytes: usize::MAX,
            },
        )
        .unwrap();

        assert_eq!(ranges, vec![0..2, 2..4, 4..5]);
    }

    #[test]
    fn embedding_batches_are_limited_by_request_bytes() {
        let texts = ["aa", "bb", "cc"].map(String::from);
        let ranges = embedding_batch_ranges(
            &texts,
            EmbeddingBatchLimits {
                max_inputs: 10,
                max_request_bytes: 5,
            },
        )
        .unwrap();

        assert_eq!(ranges, vec![0..2, 2..3]);
    }

    #[test]
    fn oversized_single_inputs_are_sent_alone() {
        let texts = ["abcdef", "g"].map(String::from);
        let ranges = embedding_batch_ranges(
            &texts,
            EmbeddingBatchLimits {
                max_inputs: 10,
                max_request_bytes: 5,
            },
        )
        .unwrap();

        assert_eq!(ranges, vec![0..1, 1..2]);
    }

    #[test]
    fn embedding_batches_reject_zero_limits() {
        let texts = ["one"].map(String::from);
        let error = embedding_batch_ranges(
            &texts,
            EmbeddingBatchLimits {
                max_inputs: 0,
                max_request_bytes: 1,
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("greater than zero"));
    }
}
