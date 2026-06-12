from __future__ import annotations

from .errors import DdserveError


def assert_non_empty(value: str, label: str) -> None:
    """Assert non empty."""
    if not value.strip():
        raise DdserveError(f"Invalid {label}: value must not be empty")


def assert_positive_integer(value: int, label: str) -> None:
    """Assert positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DdserveError(f"Invalid {label}: expected a positive integer")


def assert_non_negative_integer(value: int, label: str) -> None:
    """Assert non negative integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DdserveError(f"Invalid {label}: expected a non-negative integer")
