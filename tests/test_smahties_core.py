from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from smahties.app import purge_non_indexable_files
from smahties.context import RuntimeContext
from smahties.embedding import EmbeddingBatchLimits, embedding_batch_ranges
from smahties.models import CodeUnit, LexicalMatch, QueryMatchKind, QueryMode
from smahties.parser import ParserRegistry
from smahties.scanner import Scanner, ensure_state_dir
from smahties.service import build_fts_query, merge_matches
from smahties.store import Store
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
    venv = tmp_path / ".venv"
    cache = src / "__pycache__"
    egg_info = src / "smahtutils.egg-info"
    src.mkdir()
    target.mkdir()
    venv.mkdir()
    cache.mkdir()
    egg_info.mkdir()
    source = src / "lib.rs"
    ignored = target / "generated.rs"
    dependency = venv / "dependency.py"
    bytecode = cache / "lib.pyc"
    package_metadata = egg_info / "PKG-INFO"
    source.write_text("fn main() {}\n", encoding="utf-8")
    ignored.write_text("fn generated() {}\n", encoding="utf-8")
    dependency.write_text("def dependency(): pass\n", encoding="utf-8")
    bytecode.write_text("cache\n", encoding="utf-8")
    package_metadata.write_text("Name: smahtutils\n", encoding="utf-8")

    scanner = Scanner(tmp_path)
    discovered = scanner.discover_files(tmp_path)

    assert source in discovered
    assert ignored not in discovered
    assert dependency not in discovered
    assert bytecode not in discovered
    assert package_metadata not in discovered


def test_scanner_skips_gitignored_paths(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is required for gitignore filtering")

    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("ignored-dir/\n*.log\n", encoding="utf-8")
    source = tmp_path / "src.py"
    ignored_source = tmp_path / "ignored-dir" / "generated.py"
    ignored_log = tmp_path / "debug.log"
    ignored_source.parent.mkdir()
    source.write_text("def main(): pass\n", encoding="utf-8")
    ignored_source.write_text("def generated(): pass\n", encoding="utf-8")
    ignored_log.write_text("debug\n", encoding="utf-8")

    scanner = Scanner(tmp_path)
    discovered = scanner.discover_files(tmp_path)

    assert source in discovered
    assert ignored_source not in discovered
    assert ignored_log not in discovered
    assert scanner.read_source(ignored_source) is None


def test_scanner_treats_gitignore_as_indexing_exclude_for_tracked_files(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is required for gitignore filtering")

    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    tracked_ignored = tmp_path / "generated" / "tracked.py"
    tracked_ignored.parent.mkdir()
    tracked_ignored.write_text("def generated(): pass\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", tracked_ignored.relative_to(tmp_path)],
        check=True,
    )
    (tmp_path / ".gitignore").write_text("generated/*.py\n", encoding="utf-8")

    scanner = Scanner(tmp_path)

    assert tracked_ignored not in scanner.discover_files(tmp_path)
    assert scanner.read_source(tracked_ignored) is None


def test_scanner_respects_nested_gitignore_when_scanning_scope(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is required for gitignore filtering")

    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    app = tmp_path / "apps" / "api"
    source = app / "src" / "main.py"
    ignored_source = app / "local-only" / "generated.py"
    ignored_cache = app / "debug.cache"
    source.parent.mkdir(parents=True)
    ignored_source.parent.mkdir()
    source.write_text("def main(): pass\n", encoding="utf-8")
    ignored_source.write_text("def generated(): pass\n", encoding="utf-8")
    ignored_cache.write_text("debug\n", encoding="utf-8")
    (app / ".gitignore").write_text("local-only/\n*.cache\n", encoding="utf-8")

    scanner = Scanner(tmp_path)
    discovered = scanner.discover_files(app)

    assert source in discovered
    assert ignored_source not in discovered
    assert ignored_cache not in discovered


def test_purge_non_indexable_files_removes_gitignored_store_entries(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is required for gitignore filtering")

    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    ignored_file = tmp_path / "generated" / "old.py"
    kept_file = tmp_path / "src" / "main.py"
    ignored_file.parent.mkdir()
    kept_file.parent.mkdir()
    ignored_file.write_text("def old(): pass\n", encoding="utf-8")
    kept_file.write_text("def main(): pass\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("generated/*.py\n", encoding="utf-8")
    store = Store(tmp_path / "smahties.sqlite")
    ignored_unit = code_unit("ignored", file_path="generated/old.py")
    kept_unit = code_unit("kept", file_path="src/main.py")
    store.replace_file_units(
        ignored_unit.file_path,
        "hash-ignored",
        "parser",
        [ignored_unit],
        "model",
        [[1.0]],
    )
    store.replace_file_units(
        kept_unit.file_path,
        "hash-kept",
        "parser",
        [kept_unit],
        "model",
        [[1.0]],
    )

    purge_non_indexable_files(store, Scanner(tmp_path))

    assert store.code_units_by_ids(["ignored"]) == []
    assert [unit.file_path for unit in store.code_units_by_ids(["kept"])] == ["src/main.py"]


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


def test_runtime_context_has_no_scope_at_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)

    context = RuntimeContext.resolve(repo)

    assert context.scope_prefix is None
    assert context.scoped_path_prefix(None) is None
    assert context.scoped_path_prefix("src") == "src"


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


def code_unit(id_: str, file_path: str = "src/lib.rs") -> CodeUnit:
    return CodeUnit(
        id=id_,
        file_path=file_path,
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
