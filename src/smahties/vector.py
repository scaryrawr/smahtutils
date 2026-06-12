from __future__ import annotations

import math
import struct


def vector_to_blob(vector: list[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in vector)


def vector_from_blob(blob: bytes) -> list[float]:
    if len(blob) % 4 != 0:
        raise ValueError("embedding vector blob length is not a multiple of f32 size")
    return [item[0] for item in struct.iter_unpack("<f", blob)]


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
    return cosine_similarity_with_norms(left, right, vector_norm(left), vector_norm(right))


def cosine_similarity_with_norms(
    left: list[float],
    right: list[float],
    left_norm: float,
    right_norm: float,
) -> float | None:
    if len(left) != len(right) or not left or left_norm == 0.0 or right_norm == 0.0:
        return None
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
