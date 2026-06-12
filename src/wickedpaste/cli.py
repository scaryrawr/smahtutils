from __future__ import annotations

import argparse
import asyncio
import os

from openai import AsyncOpenAI

from wickedsmaht_config import Config, resolve_setting

from .clipboard import get_clipboard_content

SYSTEM_PROMPT = (
    "Create HTML and Markdown representation of the content it may be image contents to text "
    "formats or text transformations to other text formats. The HTML should be minimal, we don't "
    "need the <html> or other top level tags, no body, no styling, avoid adding unneeded "
    "whitespace to HTML, very plain minimal unfancy html elements."
)


def main() -> None:
    asyncio.run(async_main())


async def async_main() -> None:
    parser = argparse.ArgumentParser(prog="wickedpaste")
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--model")
    args = parser.parse_args()
    base_url, model = resolve_api_settings(args.base_url, args.model)

    content = get_clipboard_content()
    if content is None:
        return

    client = AsyncOpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))
    response = await client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Convert the following into HTML and GitHub Flavored Markdown",
                    },
                    content,
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "smaht_text",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "html": {"type": "string"},
                        "markdown": {"type": "string"},
                    },
                    "required": ["html", "markdown"],
                },
            },
        },
    )

    for choice in response.choices:
        if choice.message.content:
            print(choice.message.content, end="")


def resolve_api_settings(base_url: str | None, model: str | None) -> tuple[str, str]:
    config = Config() if base_url and model else Config.load()
    return (
        resolve_setting(base_url, config.base_url, "--base-url", "base_url"),
        resolve_setting(model, config.model, "--model", "model"),
    )


if __name__ == "__main__":
    main()
