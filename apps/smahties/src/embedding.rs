use async_openai::{
    Client,
    config::OpenAIConfig,
    types::embeddings::{CreateEmbeddingRequestArgs, Embedding},
};

use crate::Result;

#[derive(Clone)]
pub struct OpenAiEmbedder {
    client: Client<OpenAIConfig>,
    model: String,
}

impl OpenAiEmbedder {
    pub fn new(base_url: &str, model: String) -> Self {
        let config = OpenAIConfig::new().with_api_base(base_url);
        Self {
            client: Client::with_config(config),
            model,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(texts.to_vec())
            .build()?;
        let response = self.client.embeddings().create(request).await?;

        ordered_embeddings(response.data)
    }
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

    use super::ordered_embeddings;

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
}
