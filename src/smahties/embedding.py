from __future__ import annotations

import os
from dataclasses import dataclass

from openai import AsyncOpenAI

DEFAULT_MAX_EMBEDDING_BATCH_INPUTS = 128
DEFAULT_MAX_EMBEDDING_BATCH_BYTES = 256 * 1024


@dataclass(frozen=True)
class EmbeddingBatchLimits:
    """Maximum OpenAI embedding request size limits."""

    max_inputs: int = DEFAULT_MAX_EMBEDDING_BATCH_INPUTS
    max_request_bytes: int = DEFAULT_MAX_EMBEDDING_BATCH_BYTES


class OpenAiEmbedder:
    """OpenAI-compatible embedding client with deterministic batching."""

    def __init__(
        self, base_url: str, model: str, limits: EmbeddingBatchLimits | None = None
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        )
        self.limits = limits or EmbeddingBatchLimits()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in request-sized batches and preserve input order."""

        embeddings: list[list[float]] = []
        for start, end in embedding_batch_ranges(texts, self.limits):
            response = await self.client.embeddings.create(model=self.model, input=texts[start:end])
            batch = sorted(response.data, key=lambda item: item.index)
            if len(batch) != end - start:
                raise ValueError(
                    f"embedding response count {len(batch)} did not match batch input count {end - start}"
                )
            for expected, item in enumerate(batch):
                if item.index != expected:
                    raise ValueError(
                        f"embedding response index {item.index} did not match expected index {expected}"
                    )
                embeddings.append(list(item.embedding))
        return embeddings


def embedding_batch_ranges(
    texts: list[str],
    limits: EmbeddingBatchLimits,
) -> list[tuple[int, int]]:
    """Return half-open input ranges that fit embedding request limits."""

    if limits.max_inputs <= 0 or limits.max_request_bytes <= 0:
        raise ValueError("embedding batch limits must be greater than zero")
    ranges: list[tuple[int, int]] = []
    start = 0
    request_bytes = 0
    for index, text in enumerate(texts):
        batch_inputs = index - start
        would_exceed_inputs = batch_inputs >= limits.max_inputs
        would_exceed_bytes = (
            batch_inputs > 0 and request_bytes + len(text) > limits.max_request_bytes
        )
        if would_exceed_inputs or would_exceed_bytes:
            ranges.append((start, index))
            start = index
            request_bytes = 0
        request_bytes += len(text)
    if start < len(texts):
        ranges.append((start, len(texts)))
    return ranges
