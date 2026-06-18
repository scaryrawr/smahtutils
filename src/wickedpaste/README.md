# wickedpaste

`wickedpaste` converts the current clipboard into both HTML and GitHub Flavored Markdown using an OpenAI-compatible chat model.

Use it when you want to turn copied text, screenshots, diagrams, or rich visual content into a compact machine-readable JSON result.

## Quick start

Put text or an image on your clipboard, then run:

```bash
uv run wickedpaste --base-url http://127.0.0.1:14892/v1 --model <chat-model>
```

With shared config in place:

```bash
uv run wickedpaste
```

## Configuration

`wickedpaste` reads these shared settings from `$HOME/.wickedsmaht/config.json`:

```json
{
  "base_url": "http://127.0.0.1:14892/v1",
  "model": "my-chat-model"
}
```

CLI flags override config values:

```bash
uv run wickedpaste --base-url http://127.0.0.1:14892/v1 --model <chat-model>
```

The OpenAI client uses `OPENAI_API_KEY` from the environment when present. For local endpoints that do not require a key, the app sends a placeholder key.

## Output shape

The command prints a JSON object:

```json
{
  "html": "<p>Converted content</p>",
  "markdown": "Converted content"
}
```

The HTML is intentionally minimal: no top-level document tags, no body tag, no styling, and no unnecessary whitespace.

## How it works

`wickedpaste` checks the clipboard for an image first. If an image is present, it is encoded as a PNG data URL and sent as multimodal chat content. If no image is available, clipboard text is sent instead. If neither image nor text can be read, the command exits without output.

Text clipboard access uses platform clipboard commands:

| Platform | Commands |
| --- | --- |
| macOS | `pbpaste` |
| Windows | `powershell -NoProfile -Command Get-Clipboard` |
| Linux | `wl-paste`, then `xclip`, then `xsel` |

## Safety notes

Clipboard data can contain secrets or private information. Review what is on your clipboard before running the command, and do not log or commit command output that contains sensitive content.
