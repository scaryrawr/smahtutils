from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def format_table(rows: Sequence[T], columns: Sequence[tuple[str, Callable[[T], object]]]) -> str:
    """Format table."""
    if not rows:
        return ""
    rendered = [[str(column[1](row) or "") for column in columns] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered))
        for index, (header, _value) in enumerate(columns)
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, (header, _value) in enumerate(columns)),
        "  ".join("-" * widths[index] for index in range(len(columns))),
    ]
    lines.extend(
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in rendered
    )
    return "\n".join(lines)


def format_bytes(bytes_: int | None) -> str:
    """Format bytes."""
    if bytes_ is None:
        return ""
    if bytes_ < 1024:
        return f"{bytes_} B"
    value = bytes_ / 1024
    for unit in ("KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if value >= 10 else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"
