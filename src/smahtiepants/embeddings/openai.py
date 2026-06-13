from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from smahtiepants.config import SmahtiepantsConfig, resolve_openai_api_key
from smahtiepants.errors import SmahtiepantsError
from smahtiepants.text import remove_unpaired_surrogates

INTERNAL_API_KEY_PLACEHOLDER = "smahtiepants-local-openai-compatible-endpoint"
EmbeddingInput = str | Sequence[str]
EmbeddingVector = list[float]


class EmbeddingClient(Protocol):
    """Represent EmbeddingClient."""

    def create_embeddings(self, input: EmbeddingInput) -> list[EmbeddingVector]:
        """Create embedding vectors for one or more text inputs."""
        ...


class AsyncEmbeddingClient(Protocol):
    """Represent AsyncEmbeddingClient."""

    async def create_embeddings(self, input: EmbeddingInput) -> list[EmbeddingVector]:
        """Create embedding vectors for one or more text inputs."""
        ...


class AsyncClosableEmbeddingClient(AsyncEmbeddingClient, Protocol):
    """Represent an async embedding client with explicit resource cleanup."""

    async def aclose(self) -> None:
        """Close any async resources owned by the client."""
        ...


@dataclass(frozen=True)
class EmbeddingBatchLimits:
    """Embedding request size limits."""

    max_inputs: int
    max_request_bytes: int


def create_openai_embedding_client(
    config: SmahtiepantsConfig, env: dict[str, str] | None = None
) -> EmbeddingClient:
    """Implement create openai embedding client."""
    if config.openai is None:
        raise SmahtiepantsError("OpenAI embeddings are not configured")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SmahtiepantsError(
            "official `openai` package is required to create embeddings"
        ) from exc
    api_key = (
        resolve_openai_api_key(config, env) or ("placeholder", INTERNAL_API_KEY_PLACEHOLDER)
    )[1]
    client = OpenAI(api_key=api_key, base_url=config.openai.base_url)
    return OpenAiEmbeddingClient(client, config.openai.embedding_model)


def create_openai_async_embedding_client(
    config: SmahtiepantsConfig, env: dict[str, str] | None = None
) -> AsyncClosableEmbeddingClient:
    """Create an async OpenAI-compatible embedding client."""
    if config.openai is None:
        raise SmahtiepantsError("OpenAI embeddings are not configured")
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SmahtiepantsError(
            "official `openai` package is required to create embeddings"
        ) from exc
    api_key = (
        resolve_openai_api_key(config, env) or ("placeholder", INTERNAL_API_KEY_PLACEHOLDER)
    )[1]
    client = AsyncOpenAI(api_key=api_key, base_url=config.openai.base_url)
    return AsyncOpenAiEmbeddingClient(client, config.openai.embedding_model)


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
            raise SmahtiepantsError(f"OpenAI embedding request failed: {exc}") from exc
        data = getattr(response, "data", None)
        return extract_embedding_vectors(data, expected_embedding_count(normalized))


class AsyncOpenAiEmbeddingClient:
    """Represent AsyncOpenAiEmbeddingClient."""

    def __init__(self, client: object, model: str) -> None:
        """Implement init."""
        self.client = client
        self.model = model

    async def create_embeddings(self, input: EmbeddingInput) -> list[EmbeddingVector]:
        """Implement create embeddings."""
        normalized = normalize_embedding_input(input)
        try:
            response = await self.client.embeddings.create(model=self.model, input=normalized)
        except Exception as exc:
            raise SmahtiepantsError(f"OpenAI embedding request failed: {exc}") from exc
        data = getattr(response, "data", None)
        return extract_embedding_vectors(data, expected_embedding_count(normalized))

    async def aclose(self) -> None:
        """Close the underlying async OpenAI client."""
        await self.client.close()


def normalize_embedding_input(input: EmbeddingInput) -> str | list[str]:
    """Normalize embedding input."""
    if isinstance(input, str):
        return remove_unpaired_surrogates(input)
    values = [remove_unpaired_surrogates(value) for value in input]
    if not values:
        raise SmahtiepantsError("Embedding input must include at least one text value")
    return values


def expected_embedding_count(input: str | list[str]) -> int:
    """Implement expected embedding count."""
    return 1 if isinstance(input, str) else len(input)


def embedding_batch_ranges(
    texts: list[str],
    limits: EmbeddingBatchLimits,
) -> list[tuple[int, int]]:
    """Return half-open ranges that fit embedding request limits."""
    if limits.max_inputs <= 0 or limits.max_request_bytes <= 0:
        raise SmahtiepantsError("Embedding batch limits must be greater than zero")
    ranges: list[tuple[int, int]] = []
    start = 0
    request_bytes = 0
    for index, text in enumerate(texts):
        input_bytes = len(text.encode("utf-8"))
        batch_inputs = index - start
        would_exceed_inputs = batch_inputs >= limits.max_inputs
        would_exceed_bytes = (
            batch_inputs > 0 and request_bytes + input_bytes > limits.max_request_bytes
        )
        if would_exceed_inputs or would_exceed_bytes:
            ranges.append((start, index))
            start = index
            request_bytes = 0
        request_bytes += input_bytes
    if start < len(texts):
        ranges.append((start, len(texts)))
    return ranges


def extract_embedding_vectors(data: object, expected_count: int) -> list[EmbeddingVector]:
    """Implement extract embedding vectors."""
    if not isinstance(data, list):
        raise SmahtiepantsError("OpenAI embedding response was invalid: expected a data array")
    if len(data) != expected_count:
        raise SmahtiepantsError(
            f"OpenAI embedding response was invalid: expected {expected_count} embeddings, received {len(data)}"
        )
    vectors: list[EmbeddingVector | None] = [None] * expected_count
    dimensions: int | None = None
    for item in data:
        index = getattr(item, "index", None)
        embedding = getattr(item, "embedding", None)
        if not isinstance(index, int) or index < 0 or index >= expected_count:
            raise SmahtiepantsError(
                "OpenAI embedding response was invalid: embedding index was out of range"
            )
        if vectors[index] is not None:
            raise SmahtiepantsError(
                "OpenAI embedding response was invalid: duplicate embedding index"
            )
        if not isinstance(embedding, list):
            raise SmahtiepantsError(
                "OpenAI embedding response was invalid: embedding vector was not an array"
            )
        vector = validate_embedding_vector(embedding, index, dimensions)
        dimensions = dimensions or len(vector)
        vectors[index] = vector
    if any(vector is None for vector in vectors):
        raise SmahtiepantsError("OpenAI embedding response was invalid: missing embedding vector")
    return [vector for vector in vectors if vector is not None]


def validate_embedding_vector(
    vector: list[object], index: int, dimensions: int | None
) -> EmbeddingVector:
    """Validate embedding vector."""
    if not vector:
        raise SmahtiepantsError(
            f"OpenAI embedding response was invalid: embedding at index {index} was empty"
        )
    if dimensions is not None and len(vector) != dimensions:
        raise SmahtiepantsError(
            f"OpenAI embedding response dimensions mismatch: expected {dimensions}, received {len(vector)} at index {index}"
        )
    out: list[float] = []
    for value in vector:
        if not isinstance(value, int | float):
            raise SmahtiepantsError(
                f"OpenAI embedding response was invalid: embedding at index {index} contained non-numeric values"
            )
        out.append(float(value))
    return out
