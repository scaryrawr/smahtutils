from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass

from openai import AsyncOpenAI

DEFAULT_MAX_EMBEDDING_BATCH_INPUTS = 128
DEFAULT_MAX_EMBEDDING_BATCH_BYTES = 256 * 1024
DEFAULT_MAX_EMBEDDING_CONCURRENCY = 2


@dataclass(frozen=True)
class EmbeddingBatchLimits:
    """Maximum OpenAI embedding request size limits."""

    max_inputs: int = DEFAULT_MAX_EMBEDDING_BATCH_INPUTS
    max_request_bytes: int = DEFAULT_MAX_EMBEDDING_BATCH_BYTES
    max_concurrent_requests: int = DEFAULT_MAX_EMBEDDING_CONCURRENCY


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

        ranges = embedding_batch_ranges(texts, self.limits)
        if not ranges:
            return []
        if self.limits.max_concurrent_requests == 1 or len(ranges) == 1:
            embeddings: list[list[float]] = []
            for start, end in ranges:
                embeddings.extend(await self._embed_range(texts, start, end))
            return embeddings

        results: list[list[list[float]] | None] = [None] * len(ranges)
        next_range = 0

        async def worker() -> None:
            nonlocal next_range
            while next_range < len(ranges):
                range_index = next_range
                next_range += 1
                start, end = ranges[range_index]
                results[range_index] = await self._embed_range(texts, start, end)

        worker_count = min(self.limits.max_concurrent_requests, len(ranges))
        await asyncio.gather(*(worker() for _ in range(worker_count)))

        embeddings: list[list[float]] = []
        for result in results:
            if result is None:
                raise ValueError("embedding worker did not produce a result")
            embeddings.extend(result)
        return embeddings

    async def _embed_range(self, texts: list[str], start: int, end: int) -> list[list[float]]:
        """Embed one pre-sized input range and validate response ordering."""

        response = await self.client.embeddings.create(model=self.model, input=texts[start:end])
        batch = sorted(response.data, key=lambda item: item.index)
        if len(batch) != end - start:
            raise ValueError(
                f"embedding response count {len(batch)} did not match batch input count {end - start}"
            )
        embeddings: list[list[float]] = []
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

    if (
        limits.max_inputs <= 0
        or limits.max_request_bytes <= 0
        or limits.max_concurrent_requests <= 0
    ):
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
