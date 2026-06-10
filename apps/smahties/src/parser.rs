use std::path::Path;

use tree_sitter::{Language, Node, Parser};

use crate::{
    Result,
    model::{CodeUnit, SourceFile},
    scanner::sha256_hex,
};

#[derive(Clone, Debug, Default)]
pub struct ParserRegistry;

impl ParserRegistry {
    pub fn parse(&self, source_file: &SourceFile) -> Result<Vec<CodeUnit>> {
        let spec = spec_for_path(&source_file.absolute_path);
        let Some(spec) = spec else {
            return Ok(vec![fallback_unit(source_file, "text", "file")]);
        };

        let mut parser = Parser::new();
        parser.set_language(&((spec.language)()))?;
        let Some(tree) = parser.parse(&source_file.contents, None) else {
            return Ok(vec![fallback_unit(source_file, spec.language_name, "file")]);
        };

        let mut units = Vec::new();
        collect_units(
            tree.root_node(),
            source_file,
            spec,
            source_file.contents.as_bytes(),
            &mut units,
        );

        if units.is_empty() {
            units.push(fallback_unit(source_file, spec.language_name, "file"));
        }

        Ok(units)
    }

    pub fn cache_key_for_path(&self, path: &Path) -> String {
        spec_for_path(path)
            .map(|spec| spec.cache_key.to_string())
            .unwrap_or_else(|| "fallback-text:v1".into())
    }
}

struct LanguageSpec {
    language_name: &'static str,
    extensions: &'static [&'static str],
    language: fn() -> Language,
    unit_kinds: &'static [&'static str],
    cache_key: &'static str,
}

const TYPESCRIPT_KINDS: &[&str] = &[
    "function_declaration",
    "method_definition",
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "generator_function_declaration",
];
const RUST_KINDS: &[&str] = &[
    "function_item",
    "impl_item",
    "struct_item",
    "enum_item",
    "trait_item",
    "mod_item",
];
const C_FAMILY_KINDS: &[&str] = &[
    "function_definition",
    "class_specifier",
    "struct_specifier",
    "enum_specifier",
    "namespace_definition",
];
const GO_KINDS: &[&str] = &[
    "function_declaration",
    "method_declaration",
    "type_declaration",
];
const PYTHON_KINDS: &[&str] = &["function_definition", "class_definition"];
const BASH_KINDS: &[&str] = &["function_definition"];
const CSS_KINDS: &[&str] = &["rule_set", "media_statement"];
const JAVA_KINDS: &[&str] = &[
    "method_declaration",
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "constructor_declaration",
];
const RUBY_KINDS: &[&str] = &["method", "singleton_method", "class", "module"];
const CSHARP_KINDS: &[&str] = &[
    "method_declaration",
    "class_declaration",
    "interface_declaration",
    "struct_declaration",
    "enum_declaration",
];

fn specs() -> &'static [LanguageSpec] {
    &[
        LanguageSpec {
            language_name: "typescript",
            extensions: &["ts", "mts", "cts"],
            language: || tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            unit_kinds: TYPESCRIPT_KINDS,
            cache_key: "tree-sitter-typescript:v1",
        },
        LanguageSpec {
            language_name: "tsx",
            extensions: &["tsx"],
            language: || tree_sitter_typescript::LANGUAGE_TSX.into(),
            unit_kinds: TYPESCRIPT_KINDS,
            cache_key: "tree-sitter-tsx:v1",
        },
        LanguageSpec {
            language_name: "rust",
            extensions: &["rs"],
            language: || tree_sitter_rust::LANGUAGE.into(),
            unit_kinds: RUST_KINDS,
            cache_key: "tree-sitter-rust:v1",
        },
        LanguageSpec {
            language_name: "csharp",
            extensions: &["cs"],
            language: || tree_sitter_c_sharp::LANGUAGE.into(),
            unit_kinds: CSHARP_KINDS,
            cache_key: "tree-sitter-csharp:v1",
        },
        LanguageSpec {
            language_name: "cpp",
            extensions: &["cpp", "cxx", "cc", "hpp", "hxx", "hh"],
            language: || tree_sitter_cpp::LANGUAGE.into(),
            unit_kinds: C_FAMILY_KINDS,
            cache_key: "tree-sitter-cpp:v1",
        },
        LanguageSpec {
            language_name: "c",
            extensions: &["c", "h"],
            language: || tree_sitter_c::LANGUAGE.into(),
            unit_kinds: C_FAMILY_KINDS,
            cache_key: "tree-sitter-c:v1",
        },
        LanguageSpec {
            language_name: "go",
            extensions: &["go"],
            language: || tree_sitter_go::LANGUAGE.into(),
            unit_kinds: GO_KINDS,
            cache_key: "tree-sitter-go:v1",
        },
        LanguageSpec {
            language_name: "python",
            extensions: &["py"],
            language: || tree_sitter_python::LANGUAGE.into(),
            unit_kinds: PYTHON_KINDS,
            cache_key: "tree-sitter-python:v1",
        },
        LanguageSpec {
            language_name: "bash",
            extensions: &["sh", "bash", "zsh"],
            language: || tree_sitter_bash::LANGUAGE.into(),
            unit_kinds: BASH_KINDS,
            cache_key: "tree-sitter-bash:v1",
        },
        LanguageSpec {
            language_name: "css",
            extensions: &["css"],
            language: || tree_sitter_css::LANGUAGE.into(),
            unit_kinds: CSS_KINDS,
            cache_key: "tree-sitter-css:v1",
        },
        LanguageSpec {
            language_name: "java",
            extensions: &["java"],
            language: || tree_sitter_java::LANGUAGE.into(),
            unit_kinds: JAVA_KINDS,
            cache_key: "tree-sitter-java:v1",
        },
        LanguageSpec {
            language_name: "ruby",
            extensions: &["rb"],
            language: || tree_sitter_ruby::LANGUAGE.into(),
            unit_kinds: RUBY_KINDS,
            cache_key: "tree-sitter-ruby:v1",
        },
    ]
}

fn spec_for_path(path: &Path) -> Option<&'static LanguageSpec> {
    let extension = path.extension()?.to_string_lossy();
    specs()
        .iter()
        .find(|spec| spec.extensions.contains(&extension.as_ref()))
}

fn collect_units(
    node: Node<'_>,
    source_file: &SourceFile,
    spec: &LanguageSpec,
    source_bytes: &[u8],
    units: &mut Vec<CodeUnit>,
) {
    if spec.unit_kinds.contains(&node.kind()) {
        units.push(unit_from_node(node, source_file, spec, source_bytes));
        return;
    }

    for index in 0..node.named_child_count() {
        if let Some(child) = node.named_child(index) {
            collect_units(child, source_file, spec, source_bytes, units);
        }
    }
}

fn unit_from_node(
    node: Node<'_>,
    source_file: &SourceFile,
    spec: &LanguageSpec,
    source_bytes: &[u8],
) -> CodeUnit {
    let source = node.utf8_text(source_bytes).unwrap_or("").to_string();
    let source_hash = sha256_hex(source.as_bytes());
    let start = node.start_position();
    let end = node.end_position();
    let range = node.byte_range();
    let unit_type = node.kind().to_string();
    let name = node_name(node, source_bytes);

    CodeUnit {
        id: unit_id(
            &source_file.relative_path,
            range.start,
            range.end,
            &unit_type,
            &source_hash,
        ),
        file_path: source_file.relative_path.clone(),
        start_line: start.row as u32 + 1,
        end_line: end.row as u32 + 1,
        start_byte: range.start,
        end_byte: range.end,
        unit_type,
        name,
        source,
        source_hash,
        language: spec.language_name.into(),
        parser_key: spec.cache_key.into(),
    }
}

fn fallback_unit(source_file: &SourceFile, language: &str, unit_type: &str) -> CodeUnit {
    let source_hash = sha256_hex(source_file.contents.as_bytes());
    let end_line = source_file.contents.lines().count().max(1) as u32;
    CodeUnit {
        id: unit_id(
            &source_file.relative_path,
            0,
            source_file.contents.len(),
            unit_type,
            &source_hash,
        ),
        file_path: source_file.relative_path.clone(),
        start_line: 1,
        end_line,
        start_byte: 0,
        end_byte: source_file.contents.len(),
        unit_type: unit_type.into(),
        name: None,
        source: source_file.contents.clone(),
        source_hash,
        language: language.into(),
        parser_key: "fallback-text:v1".into(),
    }
}

fn node_name(node: Node<'_>, source_bytes: &[u8]) -> Option<String> {
    node.child_by_field_name("name")
        .and_then(|child| child.utf8_text(source_bytes).ok())
        .map(str::to_string)
        .or_else(|| first_named_descendant_text(node, source_bytes))
}

fn first_named_descendant_text(node: Node<'_>, source_bytes: &[u8]) -> Option<String> {
    const NAME_KINDS: &[&str] = &["identifier", "field_identifier", "type_identifier"];
    for index in 0..node.named_child_count() {
        let Some(child) = node.named_child(index) else {
            continue;
        };
        if NAME_KINDS.contains(&child.kind()) {
            return child.utf8_text(source_bytes).ok().map(str::to_string);
        }
        if let Some(name) = first_named_descendant_text(child, source_bytes) {
            return Some(name);
        }
    }
    None
}

fn unit_id(
    path: &str,
    start_byte: usize,
    end_byte: usize,
    unit_type: &str,
    source_hash: &str,
) -> String {
    sha256_hex(format!("{path}:{start_byte}:{end_byte}:{unit_type}:{source_hash}").as_bytes())
        .chars()
        .take(16)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn rust_parser_extracts_function_units_with_stable_ids() {
        let source = SourceFile {
            absolute_path: PathBuf::from("src/lib.rs"),
            relative_path: "src/lib.rs".into(),
            contents: "fn hello() -> &'static str { \"hi\" }\n".into(),
            hash: sha256_hex(b"fn hello() -> &'static str { \"hi\" }\n"),
        };
        let parser = ParserRegistry;

        let first = parser.parse(&source).unwrap();
        let second = parser.parse(&source).unwrap();

        assert_eq!(first.len(), 1);
        assert_eq!(first[0].id, second[0].id);
        assert_eq!(first[0].name.as_deref(), Some("hello"));
    }
}
