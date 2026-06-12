from __future__ import annotations

import hashlib
import html
import posixpath
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from .models import EXTRACTED_CONTENT_FORMAT, PageManifestEntry

CPP_REFERENCE_ROOTS = {
    "algorithm",
    "atomic",
    "chrono",
    "concept",
    "container",
    "coroutine",
    "error",
    "experimental",
    "filesystem",
    "header",
    "io",
    "iterator",
    "keyword",
    "language",
    "locale",
    "memory",
    "meta",
    "named_req",
    "numeric",
    "preprocessor",
    "regex",
    "string",
    "symbol_index",
    "thread",
    "types",
    "utility",
}
SKIPPED_CONTAINER_TAGS = {"script", "style", "nav"}
IGNORED_VOID_TAGS = {"img"}


def extract_markdown_pages(index: dict[str, object], db: dict[str, str], pages_dir: str | Path):
    """Implement extract markdown pages."""
    pages: list[PageManifestEntry] = []
    skipped = 0
    target = Path(pages_dir)
    target.mkdir(parents=True, exist_ok=True)
    entries = index.get("entries")
    if not isinstance(entries, list):
        return {"pages": pages, "skippedEntries": 0}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            skipped += 1
            continue
        name = str(entry.get("name") or entry["path"])
        path = str(entry["path"])
        source_key = source_key_for_path(path)
        page_html = db.get(source_key) or db.get(path)
        if not page_html:
            skipped += 1
            continue
        id_ = page_id(path, name)
        file_name = f"{id_}.md"
        (target / file_name).write_text(render_markdown(name, path, page_html), encoding="utf-8")
        pages.append(
            PageManifestEntry(
                id=id_,
                name=name,
                path=path,
                type=entry.get("type") if isinstance(entry.get("type"), str) else None,
                file=f"pages/{file_name}",
                format=EXTRACTED_CONTENT_FORMAT,
                source_key=source_key,
            )
        )
    return {"pages": pages, "skippedEntries": skipped}


def render_markdown(title: str, path: str, html_text: str) -> str:
    """Implement render markdown."""
    section = extract_html_section(html_text, path)
    body = SimpleMarkdownParser(path).convert(section)
    body = remove_duplicate_leading_heading(remove_unpaired_surrogates(body), title).strip()
    return remove_unpaired_surrogates(f"# {title}\n\n> DevDocs path: {path}\n\n{body}\n")


class SimpleMarkdownParser(HTMLParser):
    """Represent SimpleMarkdownParser."""

    def __init__(self, current_path: str) -> None:
        """Implement init."""
        super().__init__(convert_charrefs=True)
        self.current_path = current_path
        self.parts: list[str] = []
        self.skip_depth = 0
        self.pre_depth = 0
        self.heading_level: int | None = None
        self.link_stack: list[tuple[str, list[str]]] = []

    def convert(self, value: str) -> str:
        """Implement convert."""
        self.feed(value)
        self.close()
        return clean_markdown("".join(self.parts))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle starttag."""
        attrs_dict = {key: value for key, value in attrs}
        if self.skip_depth:
            if tag in SKIPPED_CONTAINER_TAGS:
                self.skip_depth += 1
            return
        if tag in SKIPPED_CONTAINER_TAGS:
            self.skip_depth += 1
            return
        if tag in IGNORED_VOID_TAGS:
            return
        if tag == "br":
            self.emit("\n")
        elif tag in {"p", "div", "section", "article", "summary", "tr"}:
            self.break_block()
        elif tag in {"ul", "ol"}:
            self.break_block()
        elif tag == "li":
            self.break_block()
            self.emit("- ")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.heading_level = int(tag[1])
            self.break_block()
            self.emit("#" * self.heading_level + " ")
        elif tag == "pre":
            self.pre_depth += 1
            language = attrs_dict.get("data-language")
            self.break_block()
            self.emit(f"```{language or ''}\n")
        elif tag == "code" and not self.pre_depth:
            self.emit("`")
        elif tag == "a":
            self.link_stack.append(
                (normalize_link_href(attrs_dict.get("href") or "", self.current_path), [])
            )
        elif tag in {"th", "td"}:
            self.emit(" | ")

    def handle_endtag(self, tag: str) -> None:
        """Handle endtag."""
        if tag in SKIPPED_CONTAINER_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "a" and self.link_stack:
            href, chunks = self.link_stack.pop()
            label = clean_inline("".join(chunks))
            self.emit(f"[{label}]({href})" if href and label else label)
            return
        if tag == "code" and not self.pre_depth:
            self.emit("`")
        elif tag == "pre" and self.pre_depth:
            self.pre_depth -= 1
            self.emit("\n```\n\n")
        elif tag in {"p", "div", "section", "article", "summary", "ul", "ol", "li", "tr"}:
            self.break_block()
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.heading_level = None
            self.break_block()

    def handle_data(self, data: str) -> None:
        """Handle data."""
        if self.skip_depth:
            return
        text = data if self.pre_depth else re.sub(r"\s+", " ", data)
        if self.link_stack:
            self.link_stack[-1][1].append(text)
        else:
            self.emit(text)

    def emit(self, value: str) -> None:
        """Implement emit."""
        self.parts.append(value)

    def break_block(self) -> None:
        """Implement break block."""
        if not self.parts:
            return
        current = "".join(self.parts)
        if not current.endswith("\n\n"):
            self.parts.append("\n\n" if not current.endswith("\n") else "\n")


def clean_markdown(value: str) -> str:
    """Implement clean markdown."""
    lines = [line.rstrip() for line in html.unescape(value).replace("\r\n", "\n").split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return collapse_spaces_outside_code_fences(text).strip()


def collapse_spaces_outside_code_fences(value: str) -> str:
    """Collapse repeated spaces outside fenced code blocks."""
    output: list[str] = []
    in_code = False
    for line in value.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
            output.append(line)
        elif in_code:
            output.append(line)
        else:
            output.append(re.sub(r"[ \t]{2,}", " ", line))
    return "\n".join(output)


def clean_inline(value: str) -> str:
    """Implement clean inline."""
    return re.sub(r"\s+", " ", value).strip()


def remove_duplicate_leading_heading(markdown: str, title: str) -> str:
    """Implement remove duplicate leading heading."""
    match = re.match(r"^\s{0,3}#{1,6}[ \t]+(.+?)\s*(?:\n+|$)", markdown)
    if not match or normalized_heading_text(match.group(1)) != normalized_heading_text(title):
        return markdown
    return markdown[match.end() :]


def normalized_heading_text(value: str) -> str:
    """Implement normalized heading text."""
    return re.sub(r"\s+", " ", re.sub(r"[`*_~\[\]]", "", value)).strip().lower()


def extract_html_section(html_text: str, path: str) -> str:
    """Implement extract html section."""
    anchor = anchor_for_path(path)
    if not anchor:
        return html_text
    heading = find_heading_by_id(html_text, anchor)
    if heading:
        start, end, level = heading
        next_heading = find_next_heading_at_or_above_level(html_text, end, level)
        return html_text[start : next_heading[0] if next_heading else len(html_text)]
    element = find_element_by_id(html_text, anchor)
    if element:
        start, end, tag = element
        return html_text[start : find_closing_tag_end(html_text, tag, end) or len(html_text)]
    return html_text


def source_key_for_path(path: str) -> str:
    """Implement source key for path."""
    return path.split("#", 1)[0] or "index"


def anchor_for_path(path: str) -> str | None:
    """Implement anchor for path."""
    if "#" not in path or path.endswith("#"):
        return None
    return html.unescape(path.split("#", 1)[1])


def find_heading_by_id(html_text: str, anchor: str) -> tuple[int, int, int] | None:
    """Implement find heading by id."""
    pattern = re.compile(r"<h([1-6])\b[^>]*\bid=(['\"])(.*?)\2[^>]*>", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(html_text):
        if html.unescape(match.group(3)) == anchor:
            return (match.start(), match.end(), int(match.group(1)))
    return None


def find_element_by_id(html_text: str, anchor: str) -> tuple[int, int, str] | None:
    """Implement find element by id."""
    pattern = re.compile(
        r"<([a-z][a-z0-9:-]*)\b[^>]*\bid=(['\"])(.*?)\2[^>]*>", re.IGNORECASE | re.DOTALL
    )
    for match in pattern.finditer(html_text):
        if html.unescape(match.group(3)) == anchor:
            return (match.start(), match.end(), match.group(1).lower())
    return None


def find_next_heading_at_or_above_level(
    html_text: str, start: int, level: int
) -> tuple[int, int] | None:
    """Implement find next heading at or above level."""
    pattern = re.compile(r"<h([1-6])\b[^>]*>", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(html_text, pos=start):
        next_level = int(match.group(1))
        if next_level <= level:
            return (match.start(), next_level)
    return None


def find_closing_tag_end(html_text: str, tag: str, start: int) -> int | None:
    """Implement find closing tag end."""
    match = re.search(rf"</{re.escape(tag)}>", html_text[start:], re.IGNORECASE)
    return start + match.end() if match else None


def normalize_link_href(href: str, current_path: str) -> str:
    """Normalize link href."""
    if (
        not href
        or re.match(r"^(?:data|javascript|mailto):", href, re.IGNORECASE)
        or href.startswith("//")
    ):
        return href
    cpp_path = normalize_cpp_reference_url(href)
    if cpp_path:
        return cpp_path
    parsed = urlparse(href)
    if parsed.scheme:
        return href
    path, suffix = split_href(href)
    if not path:
        return f"{source_key_for_path(current_path)}{suffix}"
    return f"{normalize_devdocs_path(path, current_path)}{suffix}"


def normalize_cpp_reference_url(href: str) -> str | None:
    """Normalize cpp reference url."""
    parsed = urlparse(href)
    if not parsed.hostname or not parsed.hostname.endswith("cppreference.com"):
        return None
    match = re.match(r"^/w/cpp/?(.+)?$", parsed.path)
    if not match:
        return None
    return f"{match.group(1) or 'index'}{parsed.fragment and '#' + parsed.fragment}"


def split_href(href: str) -> tuple[str, str]:
    """Implement split href."""
    match = re.search(r"[?#]", href)
    if not match:
        return href, ""
    return href[: match.start()], href[match.start() :]


def normalize_devdocs_path(path: str, current_path: str) -> str:
    """Normalize devdocs path."""
    trimmed = re.sub(r"^/+", "", path)
    trimmed = re.sub(r"^cpp/", "", trimmed)
    if path.startswith("/") or trimmed.startswith("../") or trimmed.startswith("./"):
        return posixpath.normpath(
            posixpath.join(posixpath.dirname(source_key_for_path(current_path)), trimmed)
        )
    first_segment = trimmed.split("/", 1)[0]
    if "/" in trimmed and first_segment in CPP_REFERENCE_ROOTS:
        return posixpath.normpath(trimmed)
    return posixpath.normpath(
        posixpath.join(posixpath.dirname(source_key_for_path(current_path)), trimmed)
    )


def page_id(path: str, name: str) -> str:
    """Implement page id."""
    label = re.sub(r"[^a-z0-9]+", "-", f"{name}-{path}".lower()).strip("-")[:80]
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]
    return f"{label or 'page'}-{digest}"


def remove_unpaired_surrogates(text: str) -> str:
    """Implement remove unpaired surrogates."""
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
