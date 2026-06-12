from __future__ import annotations

import hashlib
import os
from pathlib import Path

from annoy import AnnoyIndex

from .store import Store


class AnnoyIndexManager:
    """Build, load, and query rebuildable Annoy sidecar indexes."""

    def __init__(self, state_dir: Path, store: Store, trees: int = 10) -> None:
        self.state_dir = state_dir
        self.store = store
        self.trees = trees
        self.index_dir = state_dir / "annoy"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, tuple[str, AnnoyIndex]] = {}

    def search(self, model: str, query_embedding: list[float], n: int) -> list[str]:
        """Return candidate code unit IDs for a query embedding."""

        index = self._ensure_loaded(model)
        if index is None:
            return []
        _, annoy = index
        search_k = max(n * self.trees * 10, n)
        annoy_ids = annoy.get_nns_by_vector(query_embedding, n, search_k=search_k)
        unit_ids = self.store.annoy_unit_ids(model, annoy_ids)
        if len(unit_ids) >= n:
            return unit_ids

        # Annoy can return fewer items for very small or newly rebuilt indexes.
        # Fill from the SQLite mapping so callers can exact-score a bounded candidate set.
        metadata = self.store.annoy_index_metadata(model)
        if metadata is None:
            return unit_ids
        fallback_ids = range(metadata["item_count"])
        for unit_id in self.store.annoy_unit_ids(model, fallback_ids):
            if unit_id not in unit_ids:
                unit_ids.append(unit_id)
            if len(unit_ids) >= n:
                break
        return unit_ids

    def _ensure_loaded(self, model: str) -> tuple[str, AnnoyIndex] | None:
        source_version = self.store.embedding_index_version(model)
        cached = self._loaded.get(model)
        if cached and cached[0] == source_version:
            return cached

        metadata = self.store.annoy_index_metadata(model)
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
        rows = self.store.embedding_rows_for_model(model)
        if not rows:
            return None
        dimensions = len(rows[0].vector)
        if any(len(row.vector) != dimensions for row in rows):
            raise ValueError(f"stored embeddings for model {model} have mixed dimensions")

        safe_name = hashlib.sha256(model.encode("utf-8")).hexdigest()
        path = self.index_dir / f"{safe_name}-{dimensions}.ann"
        tmp_path = path.with_suffix(".ann.tmp")
        annoy = AnnoyIndex(dimensions, "angular")
        unit_ids: list[str] = []
        for annoy_id, row in enumerate(rows):
            annoy.add_item(annoy_id, row.vector)
            unit_ids.append(row.unit_id)
        annoy.build(self.trees)
        annoy.save(str(tmp_path))
        os.replace(tmp_path, path)
        self.store.replace_annoy_mapping(model, dimensions, source_version, path, unit_ids)

        loaded = AnnoyIndex(dimensions, "angular")
        loaded.load(str(path))
        self._loaded[model] = (source_version, loaded)
        return self._loaded[model]
