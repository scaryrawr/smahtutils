from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .errors import HttpError
from .models import DownloadedFile


class HttpClient:
    """Represent HttpClient."""

    def fetch_json(self, url: str) -> object:
        """Implement fetch json."""
        raise NotImplementedError

    def download_file(self, url: str, destination: str | Path) -> DownloadedFile:
        """Implement download file."""
        raise NotImplementedError


class FetchHttpClient(HttpClient):
    """Represent FetchHttpClient."""

    def __init__(self, retries: int = 2, timeout: float = 30.0) -> None:
        """Implement init."""
        self.retries = retries
        self.timeout = timeout

    def fetch_json(self, url: str) -> object:
        """Implement fetch json."""
        with self._open(url) as response:
            return json.loads(response.read().decode("utf-8"))

    def download_file(self, url: str, destination: str | Path) -> DownloadedFile:
        """Implement download file."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f"{target.name}.{int(time.time() * 1000)}.tmp")
        hasher = hashlib.sha256()
        bytes_written = 0
        try:
            with self._open(url) as response, temp.open("wb") as file:
                while True:
                    chunk = response.read(1024 * 128)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    hasher.update(chunk)
                    file.write(chunk)
            temp.replace(target)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        return DownloadedFile(path=str(target), bytes=bytes_written, sha256=hasher.hexdigest())

    def _open(self, url: str):
        """Implement open."""
        request = urllib.request.Request(
            url,
            headers={
                "user-agent": "smahtiepants/0.1.0",
                "accept": "application/json,text/html,*/*",
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if not is_retryable_status(exc.code) or attempt == self.retries:
                    raise HttpError(
                        f"Request failed with HTTP {exc.code}: {url}", exc.code, url
                    ) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt == self.retries:
                    raise HttpError(f"Request failed for {url}: {exc.reason}", 0, url) from exc
            time.sleep(0.25 * (attempt + 1))
        raise HttpError(f"Request failed for {url}: {last_error}", 0, url)


def is_retryable_status(status: int) -> bool:
    """Return whether retryable status."""
    return status in {408, 429} or status >= 500
