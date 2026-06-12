from __future__ import annotations

from pathlib import Path

from smahties.annoy_index import AnnoyIndexManager
from smahties.models import CodeUnit, Priority
from smahties.store import Store


def test_store_indexes_units_and_annoy_returns_candidates(tmp_path: Path) -> None:
    store = Store(tmp_path / "smahties.sqlite")
    unit_a = code_unit("a", "fn apple() {}", "apple")
    unit_b = code_unit("b", "fn banana() {}", "banana")
    store.replace_file_units(
        "src/lib.rs",
        "hash",
        "parser",
        [unit_a, unit_b],
        "model",
        [[1.0, 0.0], [0.0, 1.0]],
    )

    manager = AnnoyIndexManager(tmp_path, store, trees=2)
    ids = manager.search("model", [1.0, 0.0], 2)

    assert "a" in ids
    assert store.list_indexed_units("src", "rust", 10, 0, True)[0].source == "fn apple() {}"


def test_work_queue_reclaims_stale_in_progress(tmp_path: Path) -> None:
    store = Store(tmp_path / "smahties.sqlite")
    store.enqueue_work(tmp_path / "src.rs", Priority.HIGH, False)
    work = store.claim_next_work("first", 300)
    assert work is not None

    reclaimed = store.claim_next_work("second", -1)

    assert reclaimed is not None
    assert reclaimed.id == work.id


def code_unit(id_: str, source: str, name: str) -> CodeUnit:
    return CodeUnit(
        id=id_,
        file_path="src/lib.rs",
        start_line=1,
        end_line=1,
        start_byte=0,
        end_byte=len(source),
        unit_type="function",
        name=name,
        source=source,
        source_hash=f"hash-{id_}",
        language="rust",
        parser_key="parser",
    )
