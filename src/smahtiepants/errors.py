from __future__ import annotations


class SmahtiepantsError(Exception):
    """Base error for user-facing smahtiepants failures."""


DdserveError = SmahtiepantsError


class HttpError(SmahtiepantsError):
    """Represent HttpError."""

    def __init__(self, message: str, status: int, url: str) -> None:
        """Implement init."""
        super().__init__(message)
        self.status = status
        self.url = url


def get_error_message(error: BaseException) -> str:
    """Return error message."""
    return str(error)
