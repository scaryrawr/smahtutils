from __future__ import annotations

from pathlib import Path

from smahties.context import RuntimeContext
from smahties.embedding import EmbeddingBatchLimits, embedding_batch_ranges
from smahties.models import CodeUnit, LexicalMatch, QueryMatchKind, QueryMode
from smahties.parser import ParserRegistry
from smahties.scanner import Scanner, ensure_state_dir
from smahties.service import build_fts_query, merge_matches
from smahties.vector import cosine_similarity, vector_from_blob, vector_to_blob


def test_vector_blob_round_trips() -> None:
    vector = [1.0, -2.5, 3.25]
    assert vector_from_blob(vector_to_blob(vector)) == vector
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == 0.9999999999999998


def test_embedding_batches_are_limited() -> None:
    ranges = embedding_batch_ranges(
        ["one", "two", "three", "four", "five"],
        EmbeddingBatchLimits(max_inputs=2, max_request_bytes=100),
    )
    assert ranges == [(0, 2), (2, 4), (4, 5)]


def test_scanner_skips_excluded_paths(tmp_path: Path) -> None:
    src = tmp_path / "src"
    target = tmp_path / "target"
    src.mkdir()
    target.mkdir()
    source = src / "lib.rs"
    ignored = target / "generated.rs"
    source.write_text("fn main() {}\n", encoding="utf-8")
    ignored.write_text("fn generated() {}\n", encoding="utf-8")

    scanner = Scanner(tmp_path)
    assert source in scanner.discover_files(tmp_path)
    assert ignored not in scanner.discover_files(tmp_path)


def test_state_dir_contains_gitignore(tmp_path: Path) -> None:
    state_dir = ensure_state_dir(tmp_path)
    assert (state_dir / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_runtime_context_scopes_repo_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    app = repo / "apps" / "api"
    (repo / ".git").mkdir(parents=True)
    app.mkdir(parents=True)

    context = RuntimeContext.resolve(app)

    assert context.repository_root == repo.resolve()
    assert context.storage_root == repo.resolve()
    assert context.scope_prefix == "apps/api"
    assert context.scoped_path_prefix("src") == "apps/api/src"


def test_parser_extracts_python_units(tmp_path: Path) -> None:
    path = tmp_path / "app.py"
    path.write_text("class Greeter:\n    def hello(self):\n        return 'hi'\n", encoding="utf-8")
    source = Scanner(tmp_path).read_source(path)

    units = ParserRegistry().parse(source)

    assert [unit.name for unit in units] == ["Greeter", "hello"]
    assert units[0].language == "python"


def test_query_merge_combines_scores() -> None:
    unit = code_unit("one")
    matches = merge_matches(
        QueryMode.HYBRID,
        [(unit, 0.8)],
        [LexicalMatch(unit, -2.0)],
    )

    assert len(matches) == 1
    assert matches[0].match_kind == QueryMatchKind.HYBRID
    assert matches[0].semantic_score == 0.8
    assert matches[0].lexical_score == 1.0
    assert matches[0].score > 0.9


def test_fts_query_uses_safe_prefix_tokens() -> None:
    assert build_fts_query("Find config loader, config loader!") == "find* OR config* OR loader*"
    assert build_fts_query("! ? a") is None


def code_unit(id_: str) -> CodeUnit:
    return CodeUnit(
        id=id_,
        file_path="src/lib.rs",
        start_line=1,
        end_line=1,
        start_byte=0,
        end_byte=1,
        unit_type="function",
        name="load_config",
        source="fn load_config() {}",
        source_hash="hash",
        language="rust",
        parser_key="parser",
    )
