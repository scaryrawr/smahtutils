from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .models import CodeUnit, SourceFile
from .scanner import sha256_hex


@dataclass(frozen=True)
class LanguageSpec:
    """Tree-sitter language metadata used for code-unit extraction."""

    language_name: str
    extensions: tuple[str, ...]
    module_name: str
    unit_kinds: tuple[str, ...]
    cache_key: str


TYPESCRIPT_KINDS = (
    "function_declaration",
    "method_definition",
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "generator_function_declaration",
)
C_FAMILY_KINDS = (
    "function_definition",
    "class_specifier",
    "struct_specifier",
    "enum_specifier",
    "namespace_definition",
)
SPECS = (
    LanguageSpec(
        "typescript",
        ("ts", "mts", "cts"),
        "tree_sitter_typescript",
        TYPESCRIPT_KINDS,
        "tree-sitter-typescript:v2",
    ),
    LanguageSpec(
        "tsx",
        ("tsx",),
        "tree_sitter_typescript",
        TYPESCRIPT_KINDS,
        "tree-sitter-tsx:v2",
    ),
    LanguageSpec(
        "rust",
        ("rs",),
        "tree_sitter_rust",
        ("function_item", "impl_item", "struct_item", "enum_item", "trait_item", "mod_item"),
        "tree-sitter-rust:v2",
    ),
    LanguageSpec(
        "cpp",
        ("cpp", "cxx", "cc", "c++", "hpp", "hxx", "hh", "h++"),
        "tree_sitter_cpp",
        C_FAMILY_KINDS,
        "tree-sitter-cpp:v2",
    ),
    LanguageSpec("c", ("c", "h"), "tree_sitter_c", C_FAMILY_KINDS, "tree-sitter-c:v2"),
    LanguageSpec(
        "csharp",
        ("cs",),
        "tree_sitter_c_sharp",
        (
            "class_declaration",
            "struct_declaration",
            "record_declaration",
            "method_declaration",
            "constructor_declaration",
        ),
        "tree-sitter-c-sharp:v1",
    ),
    LanguageSpec(
        "go",
        ("go",),
        "tree_sitter_go",
        ("function_declaration", "method_declaration", "type_declaration"),
        "tree-sitter-go:v2",
    ),
    LanguageSpec(
        "bash",
        ("sh", "bash", "zsh"),
        "tree_sitter_bash",
        ("function_definition",),
        "tree-sitter-bash:v2",
    ),
    LanguageSpec(
        "css",
        ("css",),
        "tree_sitter_css",
        ("rule_set", "media_statement"),
        "tree-sitter-css:v2",
    ),
    LanguageSpec(
        "java",
        ("java",),
        "tree_sitter_java",
        (
            "method_declaration",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "constructor_declaration",
        ),
        "tree-sitter-java:v2",
    ),
    LanguageSpec(
        "ruby",
        ("rb",),
        "tree_sitter_ruby",
        ("method", "singleton_method", "class", "module"),
        "tree-sitter-ruby:v2",
    ),
)


class ParserRegistry:
    """Parser facade that returns code units for supported and fallback files."""

    def parse(self, source_file: SourceFile) -> list[CodeUnit]:
        """Parse a source file into indexable code units."""

        if source_file.absolute_path.suffix == ".py":
            units = parse_python_ast(source_file)
            if units:
                return units

        spec = spec_for_path(source_file.absolute_path)
        if spec is None:
            return [fallback_unit(source_file, "text", "file")]

        units = parse_tree_sitter(source_file, spec)
        return units or [fallback_unit(source_file, spec.language_name, "file")]

    def cache_key_for_path(self, path: Path) -> str:
        """Return the parser cache key that invalidates stale indexed units."""

        if path.suffix == ".py":
            return "python-ast:v1"
        spec = spec_for_path(path)
        return spec.cache_key if spec else "fallback-text:v1"


def spec_for_path(path: Path) -> LanguageSpec | None:
    """Return the language spec matching a file extension."""

    extension = path.suffix.removeprefix(".")
    return next((spec for spec in SPECS if extension in spec.extensions), None)


def parse_python_ast(source_file: SourceFile) -> list[CodeUnit]:
    """Extract Python class and function units using the stdlib AST parser."""

    try:
        tree = ast.parse(source_file.contents)
    except SyntaxError:
        return []

    lines = source_file.contents.splitlines(keepends=True)
    offsets = line_offsets(lines)
    units: list[CodeUnit] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        end_lineno = getattr(node, "end_lineno", None)
        if end_lineno is None:
            continue
        start_line = node.lineno
        end_line = int(end_lineno)
        start_byte = offsets[start_line - 1] + node.col_offset
        end_byte = offsets[end_line - 1] + len(lines[end_line - 1].encode("utf-8"))
        source = "".join(lines[start_line - 1 : end_line])
        unit_type = {
            ast.FunctionDef: "function_definition",
            ast.AsyncFunctionDef: "function_definition",
            ast.ClassDef: "class_definition",
        }[type(node)]
        units.append(
            build_unit(
                source_file,
                source,
                start_line,
                end_line,
                start_byte,
                end_byte,
                unit_type,
                node.name,
                "python",
                "python-ast:v1",
            )
        )
    return units


def parse_tree_sitter(source_file: SourceFile, spec: LanguageSpec) -> list[CodeUnit]:
    """Extract code units using a tree-sitter grammar package."""

    try:
        from tree_sitter import Language, Parser

        module = importlib.import_module(spec.module_name)
        language_fn: Callable[[], object]
        if spec.language_name == "tsx" and hasattr(module, "language_tsx"):
            language_fn = module.language_tsx
        elif spec.language_name == "typescript" and hasattr(module, "language_typescript"):
            language_fn = module.language_typescript
        else:
            language_fn = module.language
        language = Language(language_fn())
        parser = Parser(language)
        tree = parser.parse(source_file.contents.encode("utf-8"))
    except Exception:
        return []

    units: list[CodeUnit] = []
    collect_units(tree.root_node, source_file, spec, source_file.contents.encode("utf-8"), units)
    return units


def collect_units(
    node: object,
    source_file: SourceFile,
    spec: LanguageSpec,
    source_bytes: bytes,
    units: list[CodeUnit],
) -> None:
    """Recursively collect tree-sitter nodes matching the language unit kinds."""

    kind = node.type
    if kind in spec.unit_kinds:
        source = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        units.append(
            build_unit(
                source_file,
                source,
                node.start_point[0] + 1,
                node.end_point[0] + 1,
                node.start_byte,
                node.end_byte,
                kind,
                node_name(node, source_bytes),
                spec.language_name,
                spec.cache_key,
            )
        )

    for child in node.named_children:
        collect_units(child, source_file, spec, source_bytes, units)


def node_name(node: object, source_bytes: bytes) -> str | None:
    """Extract a best-effort identifier from a tree-sitter node."""

    for child in node.named_children:
        if child.type in {"identifier", "type_identifier", "property_identifier"}:
            return source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
    return None


def fallback_unit(source_file: SourceFile, language: str, unit_type: str) -> CodeUnit:
    """Create one whole-file unit when structured parsing is unavailable."""

    end_line = max(1, len(source_file.contents.splitlines()))
    return build_unit(
        source_file,
        source_file.contents,
        1,
        end_line,
        0,
        len(source_file.contents.encode("utf-8")),
        unit_type,
        None,
        language,
        "fallback-text:v1" if language == "text" else f"{language}:fallback:v1",
    )


def build_unit(
    source_file: SourceFile,
    source: str,
    start_line: int,
    end_line: int,
    start_byte: int,
    end_byte: int,
    unit_type: str,
    name: str | None,
    language: str,
    parser_key: str,
) -> CodeUnit:
    """Build a CodeUnit with stable content-derived identity."""

    source_hash = sha256_hex(source.encode("utf-8"))
    return CodeUnit(
        id=unit_id(source_file.relative_path, start_byte, end_byte, unit_type, source_hash),
        file_path=source_file.relative_path,
        start_line=start_line,
        end_line=end_line,
        start_byte=start_byte,
        end_byte=end_byte,
        unit_type=unit_type,
        name=name,
        source=source,
        source_hash=source_hash,
        language=language,
        parser_key=parser_key,
    )


def unit_id(
    file_path: str, start_byte: int, end_byte: int, unit_type: str, source_hash: str
) -> str:
    """Return a stable ID for a code unit range and source hash."""

    raw = f"{file_path}:{start_byte}:{end_byte}:{unit_type}:{source_hash}".encode("utf-8")
    return sha256_hex(raw)


def line_offsets(lines: Iterable[str]) -> list[int]:
    """Return cumulative UTF-8 byte offsets for source lines."""

    offsets = [0]
    total = 0
    for line in lines:
        total += len(line.encode("utf-8"))
        offsets.append(total)
    return offsets
