# wickedpaste

`wickedpaste` converts the current clipboard contents into structured text. It reads an image first, falls back to plain text, sends the content to an OpenAI-compatible chat endpoint, and prints JSON containing minimal HTML plus GitHub Flavored Markdown.

This is a vibe-coded personal dev utility, not a polished product. Your mileage may vary.

## Usage

```bash
cargo run -p wickedpaste -- --base-url http://127.0.0.1:14892/v1 --model <model>
```

You can omit flags when `$HOME/.wickedsmaht/config.json` provides them:

```json
{
  "base_url": "http://127.0.0.1:14892/v1",
  "model": "my-chat-model"
}
```

The output is expected to match this shape:

```json
{
  "html": "<p>Converted content</p>",
  "markdown": "Converted content"
}
```

## Clipboard behavior

`wickedpaste` checks for image data before text. Images are encoded as PNG data URLs and sent as multimodal chat content. If neither image nor text can be read, the command exits without output.

## Arguments

| Flag | Config key | Description |
| --- | --- | --- |
| `--base-url` | `base_url` | OpenAI-compatible API base URL. |
| `--model` | `model` | Chat model used for conversion. |

## Development

```bash
cargo test -p wickedpaste
cargo run -p wickedpaste -- --base-url http://127.0.0.1:14892/v1 --model <model>
```

Avoid logging clipboard contents, base64 image data, or model responses.
