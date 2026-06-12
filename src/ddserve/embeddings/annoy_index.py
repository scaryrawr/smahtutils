from __future__ import annotations

import hashlib
import os
from pathlib import Path

from annoy import AnnoyIndex

from ddserve.cache import cache_paths

from .storage import EmbeddingStorage


class AnnoyIndexManager:
    """Build, load, and query rebuildable ddserve embedding sidecar indexes."""

    def __init__(self, cache_root: str | Path, storage: EmbeddingStorage, trees: int = 10) -> None:
        """Implement init."""
        self.cache_root = Path(cache_root)
        self.storage = storage
        self.trees = trees
        self.index_dir = cache_paths(cache_root).embeddings_root / "annoy"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, tuple[str, AnnoyIndex]] = {}

    def search(self, model: str, query_embedding: list[float], n: int) -> list[int]:
        """Implement search."""
        index = self._ensure_loaded(model)
        if index is None:
            return []
        _, annoy = index
        search_k = max(n * self.trees * 10, n)
        annoy_ids = annoy.get_nns_by_vector(query_embedding, n, search_k=search_k)
        chunk_ids = self.storage.annoy_chunk_ids(model, annoy_ids)
        if len(chunk_ids) >= n:
            return chunk_ids

        metadata = self.storage.annoy_index_metadata(model)
        if metadata is None:
            return chunk_ids
        for chunk_id in self.storage.annoy_chunk_ids(model, range(metadata["item_count"])):
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
            if len(chunk_ids) >= n:
                break
        return chunk_ids

    def ensure(self, model: str) -> bool:
        """Ensure the sidecar index exists for a model."""
        return self._ensure_loaded(model) is not None

    def _ensure_loaded(self, model: str) -> tuple[str, AnnoyIndex] | None:
        """Implement ensure loaded."""
        source_version = self.storage.embedding_index_version(model)
        cached = self._loaded.get(model)
        if cached and cached[0] == source_version:
            return cached

        metadata = self.storage.annoy_index_metadata(model)
        if (
            metadata
            and metadata["source_version"] == source_version
            and Path(metadata["path"]).is_file()
        ):
            annoy = AnnoyIndex(metadata["dimensions"], "angular")
            annoy.load(metadata["path"])
            self._loaded[model] = (source_version, annoy)
            return self._loaded[model]

        return self._rebuild(model, source_version)

    def _rebuild(self, model: str, source_version: str) -> tuple[str, AnnoyIndex] | None:
        """Implement rebuild."""
        rows = self.storage.embedding_rows_for_model(model)
        if not rows:
            return None
        dimensions = len(rows[0].vector)
        if any(len(row.vector) != dimensions for row in rows):
            raise ValueError(f"stored embeddings for model {model} have mixed dimensions")

        safe_name = hashlib.sha256(model.encode("utf-8")).hexdigest()
        path = self.index_dir / f"{safe_name}-{dimensions}.ann"
        tmp_path = path.with_suffix(".ann.tmp")
        annoy = AnnoyIndex(dimensions, "angular")
        chunk_ids: list[int] = []
        for annoy_id, row in enumerate(rows):
            annoy.add_item(annoy_id, row.vector)
            chunk_ids.append(row.chunk_id)
        annoy.build(self.trees)
        annoy.save(str(tmp_path))
        os.replace(tmp_path, path)
        self.storage.replace_annoy_mapping(model, dimensions, source_version, path, chunk_ids)

        loaded = AnnoyIndex(dimensions, "angular")
        loaded.load(str(path))
        self._loaded[model] = (source_version, loaded)
        return self._loaded[model]
