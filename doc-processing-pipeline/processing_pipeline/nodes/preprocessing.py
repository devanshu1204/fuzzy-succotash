import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from config.prompts.doc_processing.preprocessing_prompt import TOC_EXTRACTION_PROMPT
from config.settings import PREPROCESSED_OUTPUT_DIR
from processing_pipeline.state import DocumentProcessingState
from utils.llm import llm

log = logging.getLogger(__name__)

_PAGE_MARKER_RE = re.compile(r"\{(\d+)\}-{3,}")
_TOC_PAGE_LIMIT = 10

# Datalab emits each figure as:
#     ![<rich description>](<hash>.jpg)
#
#     <same rich description as a plain paragraph>
# Since the description is already present below, the image-tag line is pure
# noise (hash filename + markdown syntax). We drop the tag when we can confirm
# the duplication; otherwise we leave it alone to avoid silent data loss.
_IMAGE_LINE_RE = re.compile(
    r"^!\[(?P<alt>.+?)\]\([^)]+\.(?:jpg|jpeg|png|svg|webp|gif)\)\s*$",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^#{1,6}\s")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
# Safety cap on how far ahead we'll look for a duplicate alt — past any
# realistic block. Effective limit is "next heading or next image tag",
# whichever comes first.
_LOOKAHEAD_CAP = 200


class TOCEntry(BaseModel):
    section_name: str = Field(
        description="The leaf item title exactly as it appears in the Table of Contents (e.g., 'Bank at a Glance', 'Board's Report')."
    )
    chapter_name: str = Field(
        description="The top-level grouping/part heading the entry belongs to (e.g., 'INTEGRATED REPORT', 'STATUTORY REPORTS', 'FINANCIAL STATEMENTS'). Empty string if none."
    )
    page_number: int = Field(
        description="Page number printed next to the entry in the Table of Contents."
    )


class TableOfContents(BaseModel):
    entries: list[TOCEntry] = Field(
        description="All Table of Contents entries, in the order they appear."
    )


def _extract_first_n_pages(markdown: str, n: int) -> str:
    matches = list(_PAGE_MARKER_RE.finditer(markdown))
    if not matches:
        return markdown

    chunks: list[str] = []
    for idx, match in enumerate(matches):
        page_idx = int(match.group(1))
        if page_idx >= n:
            break
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        content = markdown[start:end].strip()
        chunks.append(f"--- Page {page_idx + 1} ---\n{content}")

    return "\n\n".join(chunks)


def _clean_markdown(md: str) -> str:
    """Strip Datalab image-tag noise while preserving every piece of
    semantic content. Three patterns are handled:

    - Pattern A — alt text reappears as the very next non-blank line
      (next paragraph IS the description). Drop the image-tag line only;
      that next line becomes the description.
    - Pattern B / C-with-duplicate — alt text reappears later in the same
      block (a richer description or a data table sits in between). Drop
      both the image-tag line and the trailing duplicate; the
      intermediate content survives as the real description.
    - Pattern C-no-duplicate — alt text never reappears in the block.
      Replace the image-tag line with `Figure: <alt>` so the alt
      (which is the only description) is preserved as plain text and the
      hash filename + markdown syntax are removed.

    A block ends at the next heading (`#`–`######`) or the next image
    tag, whichever comes first.
    """
    lines = md.splitlines()
    n = len(lines)
    drop_indices: set[int] = set()
    transform: dict[int, str] = {}
    pattern_a = pattern_b = pattern_c = 0

    for i in range(n):
        m = _IMAGE_LINE_RE.match(lines[i].strip())
        if not m:
            continue
        alt = m.group("alt").strip()

        first_non_blank_idx: int | None = None
        duplicate_idx: int | None = None
        end = min(i + 1 + _LOOKAHEAD_CAP, n)
        for j in range(i + 1, end):
            stripped = lines[j].strip()
            if not stripped:
                continue
            if _HEADING_RE.match(stripped) or _IMAGE_LINE_RE.match(stripped):
                break  # block boundary
            if first_non_blank_idx is None:
                first_non_blank_idx = j
            if stripped == alt:
                duplicate_idx = j
                break

        if duplicate_idx is None:
            transform[i] = f"Figure: {alt}"
            pattern_c += 1
        elif duplicate_idx == first_non_blank_idx:
            drop_indices.add(i)
            pattern_a += 1
        else:
            drop_indices.add(i)
            drop_indices.add(duplicate_idx)
            pattern_b += 1

    out: list[str] = []
    for i in range(n):
        if i in drop_indices:
            continue
        out.append(transform.get(i, lines[i]))

    text = "\n".join(out)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    log.info(
        f"Cleaned markdown: dropped {pattern_a} Pattern-A image tags, "
        f"{pattern_b} Pattern-B/C image tags + duplicates, "
        f"transformed {pattern_c} Pattern-C tags into 'Figure: ...' lines"
    )
    return text


def preprocessing(state: DocumentProcessingState) -> dict:
    extracted_data = state.get("extracted_data") or {}
    markdown = extracted_data.get("markdown") or ""
    document_name = state.get("document_name") or "document"

    stem = Path(document_name).stem
    cached_md_path = PREPROCESSED_OUTPUT_DIR / f"{stem}.md"
    cached_toc_path = PREPROCESSED_OUTPUT_DIR / f"{stem}.toc.json"

    if cached_md_path.exists() and cached_toc_path.exists():
        cleaned_markdown = cached_md_path.read_text(encoding="utf-8")
        toc_entries = json.loads(cached_toc_path.read_text(encoding="utf-8"))
        log.info(
            f"Using cached preprocessing at {cached_md_path} + {cached_toc_path} "
            f"({len(toc_entries)} TOC entries); skipping clean + TOC LLM call"
        )
        return {
            "toc": toc_entries,
            "preprocessed_data": {"markdown": cleaned_markdown},
        }

    if not markdown:
        log.warning("No markdown in extracted_data; skipping preprocessing")
        return {"toc": [], "preprocessed_data": {"markdown": ""}}

    if cached_md_path.exists():
        cleaned_markdown = cached_md_path.read_text(encoding="utf-8")
        log.info(
            f"Using cached cleaned markdown at {cached_md_path}; "
            f"re-running TOC extraction (no TOC cache found)"
        )
    else:
        cleaned_markdown = _clean_markdown(markdown)
        log.info(
            f"Markdown size: {len(markdown):,} -> {len(cleaned_markdown):,} chars "
            f"({(1 - len(cleaned_markdown) / len(markdown)) * 100:.1f}% reduction)"
        )
        cached_md_path.parent.mkdir(parents=True, exist_ok=True)
        cached_md_path.write_text(cleaned_markdown, encoding="utf-8")
        log.info(f"Saved preprocessed markdown to: {cached_md_path}")

    toc_pages = _extract_first_n_pages(cleaned_markdown, _TOC_PAGE_LIMIT)
    log.info(f"Extracting TOC from first {_TOC_PAGE_LIMIT} pages")

    structured_llm = llm.with_structured_output(TableOfContents)
    result: TableOfContents = structured_llm.invoke(
        [
            {"role": "system", "content": TOC_EXTRACTION_PROMPT},
            {"role": "user", "content": toc_pages},
        ]
    )

    toc_entries = [entry.model_dump() for entry in result.entries]
    log.info(f"Extracted {len(toc_entries)} TOC entries")

    cached_toc_path.parent.mkdir(parents=True, exist_ok=True)
    cached_toc_path.write_text(
        json.dumps(toc_entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"Saved TOC sidecar to: {cached_toc_path}")

    return {
        "toc": toc_entries,
        "preprocessed_data": {"markdown": cleaned_markdown},
    }
