from __future__ import annotations

from pathlib import Path

from .config import DdserveConfig
from .search import search_docs
from .server_shared import list_docsets


def session_start_context(
    cache_root: str | Path,
    config: DdserveConfig,
    prompt: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    """Implement session start context."""
    docsets = list_docsets(cache_root)
    lines = ["ddserve documentation cache is available.", ""]
    if docsets:
        lines.append(
            "Installed docsets: "
            + ", ".join(f"{docset.name} ({docset.slug})" for docset in docsets[:25])
        )
    else:
        lines.append("No docsets are installed.")
    if prompt:
        try:
            results = search_docs(cache_root, prompt, config, limit=4, env=env)
            if results:
                lines.extend(["", "Prompt-relevant documentation snippets:"])
                for result in results:
                    lines.append(f"- {result.docset_slug}:{result.page_path} {result.page_title}")
                    snippet = " ".join(result.text.split())[:500]
                    lines.append(f"  {snippet}")
        except Exception as exc:
            lines.extend(["", f"Documentation search was unavailable: {exc}"])
    return {"additionalContext": "\n".join(lines)}
