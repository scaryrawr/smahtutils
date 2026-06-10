pub fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(size_of_val(vector));
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

pub fn vector_from_blob(blob: &[u8]) -> Result<Vec<f32>, String> {
    if !blob.len().is_multiple_of(size_of::<f32>()) {
        return Err("embedding vector blob length is not a multiple of f32 size".into());
    }

    Ok(blob
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }

    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;

    for (left, right) in left.iter().zip(right) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }

    if left_norm == 0.0 || right_norm == 0.0 {
        return None;
    }

    Some(dot / (left_norm.sqrt() * right_norm.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_blob_round_trips() {
        let vector = vec![1.0, -2.5, 3.25];

        assert_eq!(vector_from_blob(&vector_to_blob(&vector)).unwrap(), vector);
    }

    #[test]
    fn cosine_similarity_scores_identical_vectors_as_one() {
        let score = cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap();

        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn cosine_similarity_rejects_dimension_mismatch() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), None);
    }
}
