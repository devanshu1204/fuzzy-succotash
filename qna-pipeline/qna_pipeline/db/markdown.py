"""Preprocessed-markdown reader: grep + page-text extraction.

The preprocessed file (`{doc_id}.md` under `PREPROCESSED_OUTPUT_DIR`) uses
two distinct page numbering systems:

- **Physical** — explicit markers `{N}---` separate every page. N is 0-indexed
  sequential and is an internal detail of the markdown file.
- **Printed** — the page number printed in the original document's footer
  (e.g. `02 | Annual Report 2023-24`). This is what Mongo stores in
  `section.pages` / `chapter.pages` and what the user sees when flipping
  through the PDF.

Public API accepts and returns **printed** page numbers; physical pages
never surface to callers. The mapping is built once per doc at load time
and cached in-process.
"""

import asyncio
import logging
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

from config.settings import (
    FOOTER_REGEX_LEFT,
    FOOTER_REGEX_RIGHT,
    GET_PAGE_TEXT_TOKEN_CAP,
    GREP_MATCH_LIMIT,
    GREP_SNIPPET_CHARS,
    PREPROCESSED_OUTPUT_DIR,
    TOKEN_COUNT_ENCODING,
)

log = logging.getLogger(__name__)

_PAGE_MARKER_RE = re.compile(r"^\{(\d+)\}-+$")
_footer_left_re = re.compile(FOOTER_REGEX_LEFT)
_footer_right_re = re.compile(FOOTER_REGEX_RIGHT)

_load_lock = threading.Lock()


class PageNotFound(Exception):
    """Raised when a requested printed page has no footer mapping in the
    markdown (typically front matter or unnumbered pages).
    """


class _LoadedDoc:
    __slots__ = (
        "text",
        "lines",
        "physical_to_line_range",
        "printed_to_physical",
        "physical_to_printed",
        "line_to_physical",
    )

    def __init__(
        self,
        text: str,
        lines: list[str],
        physical_to_line_range: dict[int, tuple[int, int]],
        printed_to_physical: dict[int, int],
        physical_to_printed: dict[int, int],
        line_to_physical: list[int],
    ) -> None:
        self.text = text
        self.lines = lines
        self.physical_to_line_range = physical_to_line_range
        self.printed_to_physical = printed_to_physical
        self.physical_to_printed = physical_to_printed
        self.line_to_physical = line_to_physical


def _resolve_path(doc_id: str) -> Path:
    return PREPROCESSED_OUTPUT_DIR / f"{doc_id}.md"


def _build_page_map(
    lines: list[str],
) -> tuple[
    dict[int, tuple[int, int]],
    dict[int, int],
    dict[int, int],
    list[int],
]:
    """Walk the file once and build:

    - physical_to_line_range[physical] = (start_line, end_line) inclusive,
      excluding the marker line itself
    - printed_to_physical[printed] = physical
    - physical_to_printed[physical] = printed
    - line_to_physical[line_idx] = physical (or -1 if pre-first-marker)
    """
    marker_line_for_page: dict[int, int] = {}
    for i, line in enumerate(lines):
        m = _PAGE_MARKER_RE.match(line)
        if m:
            marker_line_for_page[int(m.group(1))] = i

    sorted_pages = sorted(marker_line_for_page)
    physical_to_line_range: dict[int, tuple[int, int]] = {}
    for idx, p in enumerate(sorted_pages):
        start = marker_line_for_page[p] + 1
        if idx + 1 < len(sorted_pages):
            end = marker_line_for_page[sorted_pages[idx + 1]] - 1
        else:
            end = len(lines) - 1
        if start <= end:
            physical_to_line_range[p] = (start, end)

    line_to_physical = [-1] * len(lines)
    for p, (start, end) in physical_to_line_range.items():
        for i in range(start, end + 1):
            line_to_physical[i] = p

    printed_to_physical: dict[int, int] = {}
    physical_to_printed: dict[int, int] = {}
    for p, (start, end) in physical_to_line_range.items():
        for i in range(start, end + 1):
            line = lines[i]
            m = _footer_left_re.match(line) or _footer_right_re.match(line)
            if m:
                printed = int(m.group(1))
                if printed not in printed_to_physical:
                    printed_to_physical[printed] = p
                physical_to_printed[p] = printed
                break

    return (
        physical_to_line_range,
        printed_to_physical,
        physical_to_printed,
        line_to_physical,
    )


@lru_cache(maxsize=8)
def _load_doc(doc_id: str) -> _LoadedDoc:
    path = _resolve_path(doc_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed markdown not found at {path}. Run the "
            f"doc-processing-pipeline first or set PREPROCESSED_OUTPUT_DIR."
        )
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    (
        physical_to_line_range,
        printed_to_physical,
        physical_to_printed,
        line_to_physical,
    ) = _build_page_map(lines)
    log.info(
        f"[markdown:{doc_id}] loaded {len(lines)} lines, "
        f"{len(physical_to_line_range)} physical pages, "
        f"{len(printed_to_physical)} printed-page mappings"
    )
    return _LoadedDoc(
        text=text,
        lines=lines,
        physical_to_line_range=physical_to_line_range,
        printed_to_physical=printed_to_physical,
        physical_to_printed=physical_to_printed,
        line_to_physical=line_to_physical,
    )


def _load_doc_threadsafe(doc_id: str) -> _LoadedDoc:
    with _load_lock:
        return _load_doc(doc_id)


def _make_snippet(line: str, match_start: int, match_end: int) -> str:
    win = GREP_SNIPPET_CHARS
    lo = max(0, match_start - win)
    hi = min(len(line), match_end + win)
    snippet = line[lo:hi]
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(line) else ""
    return f"{prefix}{snippet}{suffix}"


def _grep_sync(
    doc_id: str,
    pattern: str,
    pages_filter: Optional[list[int]],
    regex: bool,
    limit: int,
) -> list[dict]:
    doc = _load_doc_threadsafe(doc_id)

    compiled = re.compile(pattern if regex else re.escape(pattern), re.IGNORECASE)

    physical_filter: Optional[set[int]] = None
    if pages_filter:
        physical_filter = set()
        for printed in pages_filter:
            phys = doc.printed_to_physical.get(printed)
            if phys is not None:
                physical_filter.add(phys)
        if not physical_filter:
            return []

    cap = max(1, min(limit, GREP_MATCH_LIMIT))
    matches: list[dict] = []
    for line_idx, line in enumerate(doc.lines):
        if not line:
            continue
        if _PAGE_MARKER_RE.match(line):
            continue
        phys = doc.line_to_physical[line_idx]
        if phys < 0:
            continue
        if physical_filter is not None and phys not in physical_filter:
            continue
        m = compiled.search(line)
        if m is None:
            continue
        printed = doc.physical_to_printed.get(phys)
        matches.append(
            {
                "printed_page": printed,
                "physical_page": phys,
                "line": line_idx + 1,
                "snippet": _make_snippet(line, m.start(), m.end()),
            }
        )
        if len(matches) >= cap:
            break
    return matches


def _get_page_text_sync(doc_id: str, printed_pages: list[int]) -> str:
    doc = _load_doc_threadsafe(doc_id)
    parts: list[str] = []
    missing: list[int] = []
    for printed in printed_pages:
        phys = doc.printed_to_physical.get(printed)
        if phys is None:
            missing.append(printed)
            continue
        line_range = doc.physical_to_line_range.get(phys)
        if line_range is None:
            missing.append(printed)
            continue
        start, end = line_range
        page_text = "\n".join(doc.lines[start : end + 1]).strip()
        parts.append(f"--- Printed page {printed} ---\n{page_text}")
    if missing and not parts:
        raise PageNotFound(
            f"None of the requested printed pages have a footer mapping in "
            f"{doc_id}.md: {missing}. Front matter and unnumbered pages have "
            f"no printed page number. Try a nearby printed page from a "
            f"section/chapter `pages` array."
        )
    joined = "\n\n".join(parts)
    truncated = _truncate_to_token_budget(joined, GET_PAGE_TEXT_TOKEN_CAP)
    if missing:
        truncated += (
            f"\n\n[note: printed pages with no footer mapping skipped: "
            f"{missing}]"
        )
    return truncated


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    try:
        import tiktoken

        enc = tiktoken.get_encoding(TOKEN_COUNT_ENCODING)
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        truncated = enc.decode(toks[:max_tokens])
        return truncated + f"\n\n[truncated at {max_tokens} tokens]"
    except Exception:
        cap_chars = max_tokens * 4
        if len(text) <= cap_chars:
            return text
        return text[:cap_chars] + f"\n\n[truncated at ~{max_tokens} tokens]"


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def grep(
    doc_id: str,
    pattern: str,
    pages_filter: Optional[list[int]] = None,
    regex: bool = False,
    limit: int = GREP_MATCH_LIMIT,
) -> list[dict]:
    """Case-insensitive substring (or regex) search over the preprocessed
    markdown.

    Args:
        doc_id: document identifier (resolves to `{doc_id}.md`).
        pattern: search string. Treated as a literal when `regex=False`
            (default); otherwise as a Python regex.
        pages_filter: optional list of PRINTED page numbers to restrict
            the search to. Pages with no footer mapping are silently
            excluded.
        regex: when True, treat `pattern` as a regex.
        limit: max matches to return (capped at GREP_MATCH_LIMIT).

    Returns:
        List of `{printed_page, physical_page, line, snippet}` dicts.
        `printed_page` is None if the matching line is on a physical
        page with no footer.
    """
    return await asyncio.to_thread(
        _grep_sync, doc_id, pattern, pages_filter, regex, limit
    )


async def get_page_text(doc_id: str, printed_pages: list[int]) -> str:
    """Return the raw text of the requested printed pages, joined and
    truncated to `GET_PAGE_TEXT_TOKEN_CAP` tokens.

    Raises:
        PageNotFound: if NONE of the requested printed pages have a footer
            mapping. (Partial misses are reported in a trailing note.)
    """
    return await asyncio.to_thread(_get_page_text_sync, doc_id, printed_pages)


def _list_pages_sync(doc_id: str) -> dict:
    doc = _load_doc_threadsafe(doc_id)
    printed = doc.printed_to_physical
    physical_count = len(doc.physical_to_line_range)
    if not printed:
        return {
            "printed_page_min": None,
            "printed_page_max": None,
            "printed_page_count": 0,
            "physical_page_count": physical_count,
            "has_unmapped_pages": physical_count > 0,
        }
    return {
        "printed_page_min": min(printed),
        "printed_page_max": max(printed),
        "printed_page_count": len(printed),
        "physical_page_count": physical_count,
        "has_unmapped_pages": physical_count > len(printed),
    }


async def list_pages(doc_id: str) -> dict:
    """Return a small descriptor of the document's page numbering.

    Cheap orientation tool for the lookup agent: lets it sanity-check a
    user-supplied printed page number and pick a sensible pages_filter
    window without grepping blindly.

    Keys:
        printed_page_min / printed_page_max: extrema of footer-derived
            printed page numbers (None if the doc has no footer mapping).
        printed_page_count: number of distinct printed pages mapped.
        physical_page_count: number of physical pages in the markdown.
        has_unmapped_pages: True when some physical pages have no printed
            footer (front matter, inserts, foldouts) — printed numbering
            may have gaps.
    """
    return await asyncio.to_thread(_list_pages_sync, doc_id)
