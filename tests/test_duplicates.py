from __future__ import annotations

import asyncio
from pathlib import Path

from smahties.cli import build_parser
from smahties.duplicates import (
    ComparisonLevel,
    DEFAULT_DUPLICATE_THRESHOLD,
    DuplicateEntry,
    DuplicateSearchOptions,
    cluster_duplicates,
    create_file_unit,
    find_duplicate_pairs,
    find_duplicate_pairs_annoy,
    find_duplicate_pairs_prefer_annoy,
    parse_comparison_levels,
    parse_threshold,
    stored_unit_matches_levels,
    transient_file_entries,
)
from smahties.models import CodeUnit
from smahties.parser import ParserRegistry
from smahties.scanner import Scanner
from smahties.store import Store
from smahties.vector import vector_norm


def test_parse_comparison_levels_accepts_repeated_and_comma_separated_values() -> None:
    assert parse_comparison_levels(["function,class", "file"]) == (
        ComparisonLevel.FUNCTION,
        ComparisonLevel.CLASS,
        ComparisonLevel.FILE,
    )
    assert parse_comparison_levels(None) == (ComparisonLevel.FUNCTION,)


def test_parse_threshold_rejects_values_outside_similarity_range() -> None:
    assert parse_threshold("0.85") == 0.85
    for value in ("", "nope", "-0.1", "1.1"):
        try:
            parse_threshold(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected threshold {value!r} to be rejected")


def test_stored_unit_level_filtering_excludes_file_units() -> None:
    function = code_unit("function", "function_definition")
    klass = code_unit("class", "type_declaration")
    file_unit = code_unit("file", "file")

    assert stored_unit_matches_levels(function, [ComparisonLevel.FUNCTION])
    assert stored_unit_matches_levels(klass, [ComparisonLevel.CLASS])
    assert not stored_unit_matches_levels(file_unit, [ComparisonLevel.FILE])


def test_duplicate_pairs_and_clusters_use_cosine_threshold() -> None:
    first = entry("first", [1.0, 0.0])
    second = entry("second", [0.99, 0.01])
    third = entry("third", [0.0, 1.0])

    pairs = find_duplicate_pairs([first, second, third], 0.9)
    clusters = cluster_duplicates(pairs)

    assert [(pair.unitA.id, pair.unitB.id) for pair in pairs] == [("first", "second")]
    assert len(clusters) == 1
    assert [unit.id for unit in clusters[0].units] == ["first", "second"]
    assert clusters[0].pairs[0].unitIdA == "first"
    assert clusters[0].pairs[0].unitIdB == "second"


def test_bounded_duplicate_search_finds_signature_candidates() -> None:
    entries = [entry(f"item-{index}", [1.0, 0.0, 0.0]) for index in range(4)]

    pairs = find_duplicate_pairs(
        entries,
        0.99,
        DuplicateSearchOptions(exhaustive_search_limit=0, max_candidates_per_unit=2),
    )

    assert pairs
    assert all(pair.similarity >= 0.99 for pair in pairs)


def test_annoy_duplicate_search_exact_scores_candidate_pairs() -> None:
    first = entry("first", [1.0, 0.0])
    second = entry("second", [0.99, 0.01])
    third = entry("third", [0.0, 1.0])
    annoy = FakeAnnoy({"first": ["first", "second"], "second": ["second", "first"]}, item_count=3)

    pairs = find_duplicate_pairs_annoy([first, second, third], 0.9, annoy, "model")

    assert [(pair.unitA.id, pair.unitB.id) for pair in pairs] == [("first", "second")]
    assert annoy.calls == [
        ("model", [1.0, 0.0], 3),
        ("model", [0.99, 0.01], 3),
        ("model", [0.0, 1.0], 3),
    ]


def test_preferred_search_uses_annoy_for_large_persisted_sets() -> None:
    first = entry("first", [1.0, 0.0])
    second = entry("second", [0.99, 0.01])
    annoy = FakeAnnoy({"first": ["second"], "second": ["first"]}, item_count=2)

    pairs = find_duplicate_pairs_prefer_annoy(
        [first, second],
        [],
        0.9,
        annoy,
        "model",
        DuplicateSearchOptions(exhaustive_search_limit=0),
    )

    assert [(pair.unitA.id, pair.unitB.id) for pair in pairs] == [("first", "second")]
    assert annoy.calls


def test_preferred_search_exact_scores_transient_entries() -> None:
    persisted = entry("persisted", [1.0, 0.0])
    transient = entry("transient", [0.99, 0.01])

    pairs = find_duplicate_pairs_prefer_annoy(
        [persisted],
        [transient],
        0.9,
        FakeAnnoy({}),
        "model",
        DuplicateSearchOptions(exhaustive_search_limit=0),
    )

    assert [(pair.unitA.id, pair.unitB.id) for pair in pairs] == [("transient", "persisted")]


def test_annoy_duplicate_search_overfetches_when_index_contains_out_of_scope_items() -> None:
    first = entry("first", [1.0, 0.0])
    second = entry("second", [0.99, 0.01])
    annoy = FakeAnnoy(
        {
            "first": ["outside-1", "outside-2", "second"],
            "second": ["outside-1", "outside-2", "first"],
        },
        item_count=4,
    )

    pairs = find_duplicate_pairs_annoy(
        [first, second],
        0.9,
        annoy,
        "model",
        DuplicateSearchOptions(exhaustive_search_limit=0, max_candidates_per_unit=1),
    )

    assert [(pair.unitA.id, pair.unitB.id) for pair in pairs] == [("first", "second")]
    assert annoy.calls == [("model", [1.0, 0.0], 4), ("model", [0.99, 0.01], 4)]


def test_transient_file_entries_do_not_match_persisted_units_from_same_file() -> None:
    persisted_same_file = entry("persisted-same", [1.0, 0.0], file_path="src/main.py")
    persisted_other_file = entry("persisted-other", [1.0, 0.0], file_path="src/other.py")
    transient = entry("transient", [1.0, 0.0], file_path="src/main.py", unit_type="file")

    pairs = find_duplicate_pairs_prefer_annoy(
        [persisted_same_file, persisted_other_file],
        [transient],
        0.9,
        FakeAnnoy({}, item_count=0),
        "model",
        DuplicateSearchOptions(exhaustive_search_limit=0),
    )

    pair_ids = {frozenset((pair.unitA.id, pair.unitB.id)) for pair in pairs}
    assert frozenset(("transient", "persisted-same")) not in pair_ids
    assert frozenset(("transient", "persisted-other")) in pair_ids


def test_store_loads_scoped_duplicate_candidates(tmp_path: Path) -> None:
    store = Store(tmp_path / "smahties.sqlite")
    kept = code_unit("kept", "function_definition", "apps/api/src/main.py", "python")
    skipped_path = code_unit(
        "skipped-path", "function_definition", "libs/tool/src/main.py", "python"
    )
    skipped_language = code_unit(
        "skipped-language", "function_definition", "apps/api/src/lib.rs", "rust"
    )
    for unit, vector in (
        (kept, [1.0, 0.0]),
        (skipped_path, [0.0, 1.0]),
        (skipped_language, [1.0, 1.0]),
    ):
        store.replace_file_units(
            unit.file_path, f"hash-{unit.id}", "parser", [unit], "model", [vector]
        )

    candidates = store.code_unit_embeddings_for_model("model", ["apps/api"], "python")

    assert [candidate.unit.id for candidate in candidates] == ["kept"]
    assert candidates[0].vector == [1.0, 0.0]


def test_transient_file_entries_create_unpersisted_file_units(tmp_path: Path) -> None:
    source = tmp_path / "main.py"
    source.write_text("def main():\n    return 1\n", encoding="utf-8")

    entries = asyncio.run(
        transient_file_entries(
            Scanner(tmp_path),
            ParserRegistry(),
            FakeEmbedder(),
            [source],
            "python",
        )
    )

    assert len(entries) == 1
    assert entries[0].unit.unit_type == "file"
    assert entries[0].unit.name == "main.py"
    assert entries[0].vector == [1.0, 0.0]


def test_file_unit_uses_parser_language(tmp_path: Path) -> None:
    source = tmp_path / "main.py"
    source.write_text("def main():\n    return 1\n", encoding="utf-8")
    source_file = Scanner(tmp_path).read_source(source)

    unit = create_file_unit(source_file, ParserRegistry())

    assert unit.language == "python"
    assert unit.unit_type == "file"


def test_duplicates_cli_arguments_are_registered() -> None:
    args = build_parser().parse_args(
        ["duplicates", "src", "tests", "--threshold", "0.82", "--level", "function,class"]
    )

    assert args.command == "duplicates"
    assert args.path == ["src", "tests"]
    assert args.threshold == "0.82"
    assert args.level == ["function,class"]


def test_duplicates_cli_default_threshold_is_conservative() -> None:
    args = build_parser().parse_args(["duplicates"])

    assert args.threshold == str(DEFAULT_DUPLICATE_THRESHOLD)
    assert DEFAULT_DUPLICATE_THRESHOLD == 0.92


def test_default_threshold_filters_related_but_not_high_confidence_pairs() -> None:
    related = entry("related", [0.901, (1 - 0.901**2) ** 0.5, 0.0])
    high_confidence = entry("high-confidence", [0.925, 0.0, (1 - 0.925**2) ** 0.5])
    anchor = entry("anchor", [1.0, 0.0, 0.0])

    pairs = find_duplicate_pairs([anchor, related, high_confidence], DEFAULT_DUPLICATE_THRESHOLD)

    assert [(pair.unitA.id, pair.unitB.id) for pair in pairs] == [("anchor", "high-confidence")]


class FakeEmbedder:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]


class FakeAnnoy:
    def __init__(self, neighbors: dict[str, list[str]], item_count: int = 0) -> None:
        self.neighbors = neighbors
        self._item_count = item_count
        self.calls: list[tuple[str, list[float], int]] = []

    def search(self, model: str, query_embedding: list[float], n: int) -> list[str]:
        self.calls.append((model, query_embedding, n))
        for id_, vector in {
            "first": [1.0, 0.0],
            "second": [0.99, 0.01],
            "third": [0.0, 1.0],
            "persisted": [1.0, 0.0],
            "transient": [0.99, 0.01],
        }.items():
            if vector == query_embedding:
                return self.neighbors.get(id_, [])
        return []

    def item_count(self, model: str) -> int:
        return self._item_count


def entry(
    id_: str,
    vector: list[float],
    file_path: str = "src/main.py",
    unit_type: str = "function_definition",
) -> DuplicateEntry:
    return DuplicateEntry(
        code_unit(id_, unit_type=unit_type, file_path=file_path), vector, vector_norm(vector)
    )


def code_unit(
    id_: str,
    unit_type: str = "function_definition",
    file_path: str = "src/main.py",
    language: str = "python",
) -> CodeUnit:
    return CodeUnit(
        id=id_,
        file_path=file_path,
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=10,
        unit_type=unit_type,
        name=id_,
        source=f"def {id_.replace('-', '_')}():\n    pass\n",
        source_hash=f"hash-{id_}",
        language=language,
        parser_key="parser",
    )
