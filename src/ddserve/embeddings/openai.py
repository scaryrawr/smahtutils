from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ddserve.config import DdserveConfig, resolve_openai_api_key
from ddserve.errors import DdserveError
from ddserve.text import remove_unpaired_surrogates

INTERNAL_API_KEY_PLACEHOLDER = "ddserve-local-openai-compatible-endpoint"
EmbeddingInput = str | Sequence[str]
EmbeddingVector = list[float]


class EmbeddingClient(Protocol):
    """Represent EmbeddingClient."""

    def create_embeddings(self, input: EmbeddingInput) -> list[EmbeddingVector]:
        """Create embedding vectors for one or more text inputs."""
        ...


def create_openai_embedding_client(
    config: DdserveConfig, env: dict[str, str] | None = None
) -> EmbeddingClient:
    """Implement create openai embedding client."""
    if config.openai is None:
        raise DdserveError("OpenAI embeddings are not configured")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise DdserveError("official `openai` package is required to create embeddings") from exc
    api_key = (
        resolve_openai_api_key(config, env) or ("placeholder", INTERNAL_API_KEY_PLACEHOLDER)
    )[1]
    client = OpenAI(api_key=api_key, base_url=config.openai.base_url)
    return OpenAiEmbeddingClient(client, config.openai.embedding_model)


class OpenAiEmbeddingClient:
    """Represent OpenAiEmbeddingClient."""

    def __init__(self, client: object, model: str) -> None:
        """Implement init."""
        self.client = client
        self.model = model

    def create_embeddings(self, input: EmbeddingInput) -> list[EmbeddingVector]:
        """Implement create embeddings."""
        normalized = normalize_embedding_input(input)
        try:
            response = self.client.embeddings.create(model=self.model, input=normalized)
        except Exception as exc:
            raise DdserveError(f"OpenAI embedding request failed: {exc}") from exc
        data = getattr(response, "data", None)
        return extract_embedding_vectors(data, expected_embedding_count(normalized))


def normalize_embedding_input(input: EmbeddingInput) -> str | list[str]:
    """Normalize embedding input."""
    if isinstance(input, str):
        return remove_unpaired_surrogates(input)
    values = [remove_unpaired_surrogates(value) for value in input]
    if not values:
        raise DdserveError("Embedding input must include at least one text value")
    return values


def expected_embedding_count(input: str | list[str]) -> int:
    """Implement expected embedding count."""
    return 1 if isinstance(input, str) else len(input)


def extract_embedding_vectors(data: object, expected_count: int) -> list[EmbeddingVector]:
    """Implement extract embedding vectors."""
    if not isinstance(data, list):
        raise DdserveError("OpenAI embedding response was invalid: expected a data array")
    if len(data) != expected_count:
        raise DdserveError(
            f"OpenAI embedding response was invalid: expected {expected_count} embeddings, received {len(data)}"
        )
    vectors: list[EmbeddingVector | None] = [None] * expected_count
    dimensions: int | None = None
    for item in data:
        index = getattr(item, "index", None)
        embedding = getattr(item, "embedding", None)
        if not isinstance(index, int) or index < 0 or index >= expected_count:
            raise DdserveError(
                "OpenAI embedding response was invalid: embedding index was out of range"
            )
        if vectors[index] is not None:
            raise DdserveError("OpenAI embedding response was invalid: duplicate embedding index")
        if not isinstance(embedding, list):
            raise DdserveError(
                "OpenAI embedding response was invalid: embedding vector was not an array"
            )
        vector = validate_embedding_vector(embedding, index, dimensions)
        dimensions = dimensions or len(vector)
        vectors[index] = vector
    if any(vector is None for vector in vectors):
        raise DdserveError("OpenAI embedding response was invalid: missing embedding vector")
    return [vector for vector in vectors if vector is not None]


def validate_embedding_vector(
    vector: list[object], index: int, dimensions: int | None
) -> EmbeddingVector:
    """Validate embedding vector."""
    if not vector:
        raise DdserveError(
            f"OpenAI embedding response was invalid: embedding at index {index} was empty"
        )
    if dimensions is not None and len(vector) != dimensions:
        raise DdserveError(
            f"OpenAI embedding response dimensions mismatch: expected {dimensions}, received {len(vector)} at index {index}"
        )
    out: list[float] = []
    for value in vector:
        if not isinstance(value, int | float):
            raise DdserveError(
                f"OpenAI embedding response was invalid: embedding at index {index} contained non-numeric values"
            )
        out.append(float(value))
    return out
