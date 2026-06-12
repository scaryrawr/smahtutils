from __future__ import annotations

import base64
import io
import platform
import subprocess
from typing import Any


def clipboard_image_data_url(image: Any) -> str:
    """Encode a clipboard image object as a PNG data URL for chat APIs."""

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("invalid clipboard image data")
    buffer = io.BytesIO()
    image.convert("RGBA").save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def get_clipboard_content() -> dict[str, object] | None:
    """Read clipboard content as an OpenAI chat content part, preferring images."""

    image = get_clipboard_image()
    if image is not None:
        return {"type": "image_url", "image_url": {"url": clipboard_image_data_url(image)}}
    text = get_clipboard_text()
    if text:
        return {"type": "text", "text": text}
    return None


def get_clipboard_image() -> Any | None:
    """Return an image from the system clipboard when Pillow can read one."""

    try:
        from PIL import ImageGrab

        image = ImageGrab.grabclipboard()
    except Exception:
        return None
    return image if hasattr(image, "save") and hasattr(image, "size") else None


def get_clipboard_text() -> str | None:
    """Return text from the system clipboard using platform clipboard commands."""

    system = platform.system()
    commands: list[list[str]]
    if system == "Darwin":
        commands = [["pbpaste"]]
    elif system == "Windows":
        commands = [["powershell", "-NoProfile", "-Command", "Get-Clipboard"]]
    else:
        commands = [
            ["wl-paste", "--no-newline"],
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "-b", "-o"],
        ]

    for command in commands:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=2)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.stdout:
            return result.stdout
    return None
