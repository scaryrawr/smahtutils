from __future__ import annotations

from PIL import Image

from wickedpaste.clipboard import clipboard_image_data_url


def test_clipboard_image_data_url_encodes_png() -> None:
    image = Image.new("RGBA", (1, 1), (255, 0, 0, 255))

    assert clipboard_image_data_url(image).startswith("data:image/png;base64,")
