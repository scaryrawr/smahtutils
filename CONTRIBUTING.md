# Contributing

This repository contains consumer-facing local tools. Keep root documentation focused on using the apps; put contributor setup, validation, and design notes here or in `docs/`.

## Development setup

```bash
uv sync --locked --all-groups
```

Use the project scripts through `uv run`:

```bash
uv run smahtiepants --help
uv run smahties --help
uv run wickedpaste --help
```

## Validation

Run the checks that match your change:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

Documentation-only changes usually only need link/diff review unless they touch generated examples or tested behavior.

## Architecture and design references

- [`docs/architecture.md`](docs/architecture.md) - contributor architecture map and design constraints.
- [`src/smahtiepants/README.md`](src/smahtiepants/README.md) - user-facing DevDocs cache/search/server behavior.
- [`src/smahties/README.md`](src/smahties/README.md) - user-facing code-search and duplicate detection behavior.
- [`src/wickedpaste/README.md`](src/wickedpaste/README.md) - user-facing clipboard conversion behavior.
- [`src/wickedsmaht_config/README.md`](src/wickedsmaht_config/README.md) - shared config schema and precedence.

## Documentation style

Package READMEs should answer "How do I use this?" before "How is this built?". Keep implementation detail short and practical. Put deeper architecture, design rationale, and maintenance notes in `docs/` and link to them from here.

## Security and privacy

Do not log, commit, or paste secrets, clipboard contents, base64 images, embeddings, cached documentation content, indexed private source snippets, or model responses. Keep endpoint URLs and model names configurable.
