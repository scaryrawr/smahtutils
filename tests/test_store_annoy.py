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


def test_path_prefix_filters_treat_wildcards_literally(tmp_path: Path) -> None:
    store = Store(tmp_path / "smahties.sqlite")
    underscored = code_unit("underscore", "fn needle_under() {}", "needle", "apps/api_v1/lib.rs")
    similar = code_unit("similar", "fn needle_other() {}", "needle", "apps/apiXv1/lib.rs")
    store.replace_file_units(
        underscored.file_path,
        "hash-under",
        "parser",
        [underscored],
        "model",
        [[1.0, 0.0]],
    )
    store.replace_file_units(
        similar.file_path,
        "hash-similar",
        "parser",
        [similar],
        "model",
        [[0.0, 1.0]],
    )

    listed = store.list_indexed_units("apps/api_v1", None, 10, 0, False)
    lexical = store.lexical_search("needle*", "apps/api_v1", None, 10)

    assert [item.file_path for item in listed] == ["apps/api_v1/lib.rs"]
    assert [match.unit.file_path for match in lexical] == ["apps/api_v1/lib.rs"]

    store.delete_path_prefix("apps/api_v1")

    assert store.code_units_by_ids(["underscore"]) == []
    assert [unit.file_path for unit in store.code_units_by_ids(["similar"])] == [
        "apps/apiXv1/lib.rs"
    ]


def code_unit(id_: str, source: str, name: str, file_path: str = "src/lib.rs") -> CodeUnit:
    return CodeUnit(
        id=id_,
        file_path=file_path,
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
