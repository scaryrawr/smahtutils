from __future__ import annotations

from pathlib import Path

from wickedsmaht_config import Config, resolve_setting

from .annoy_index import AnnoyIndexManager
from .context import RuntimeContext
from .embedding import OpenAiEmbedder
from .indexer import Indexer
from .models import Priority
from .parser import ParserRegistry
from .scanner import EXCLUDED_DIR_NAMES, EXCLUDED_FILE_NAMES, Scanner
from .service import AppState
from .store import Store
from .watcher import PollingWatcher, start


def build_state(
    root: Path,
    base_url: str | None,
    coding_embedding_model: str | None,
    api_required: bool,
) -> AppState:
    context = RuntimeContext.resolve(root)
    state_dir = context.state_dir()
    resolved_base_url, model = resolve_api_settings(base_url, coding_embedding_model, api_required)
    store = Store(state_dir / "smahties.sqlite")
    for excluded_dir in EXCLUDED_DIR_NAMES:
        store.delete_path_prefix(excluded_dir)
    for excluded_file in EXCLUDED_FILE_NAMES:
        store.delete_file_name(excluded_file)
    scanner = Scanner(context.storage_root)
    embedder = OpenAiEmbedder(resolved_base_url, model)
    indexer = Indexer(scanner, ParserRegistry(), store, embedder)
    annoy = AnnoyIndexManager(state_dir, store)
    return AppState(store, indexer, embedder, context, annoy)


def resolve_api_settings(
    base_url: str | None,
    coding_embedding_model: str | None,
    api_required: bool,
) -> tuple[str, str]:
    config = Config() if base_url and coding_embedding_model else Config.load()
    if not api_required:
        return (
            base_url or config.base_url or "",
            coding_embedding_model or config.coding_embedding_model or "not configured",
        )
    return (
        resolve_setting(base_url, config.base_url, "--base-url", "base_url"),
        resolve_setting(
            coding_embedding_model,
            config.coding_embedding_model,
            "--coding-embedding-model",
            "coding_embedding_model",
        ),
    )


async def start_mcp_state(state: AppState) -> PollingWatcher | None:
    state.indexer.spawn_worker()
    if auto_index_root := state.context.auto_index_root():
        await state.indexer.enqueue_path(auto_index_root, Priority.LOW)
        return start(auto_index_root, state.indexer)
    return None
