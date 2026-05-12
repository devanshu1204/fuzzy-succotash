import json
import logging
import re
from pathlib import Path
from typing import Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    SEGMENT_CHUNK_OVERLAP,
    SEGMENT_CHUNK_SIZE,
    SEGMENTATION_OUTPUT_DIR,
    TOKEN_COUNT_ENCODING,
)
from processing_pipeline.state import DocumentProcessingState

log = logging.getLogger(__name__)

_PAGE_MARKER_RE = re.compile(r"\{(\d+)\}-{3,}")
# Footer formats seen in real OCR output: "02 | Annual Report 2023-24" and
# "Annual Report 2023-24 | 03". Bold markers (**) and surrounding whitespace
# are tolerated. The non-number side must contain at least one alpha char so we
# don't accidentally match table separators.
_FOOTER_LEFT_RE = re.compile(r"^\**\s*(\d{1,4})\s*\**\s*\|\s*.*[A-Za-z].*$")
_FOOTER_RIGHT_RE = re.compile(r"^.*[A-Za-z].*\s*\|\s*\**\s*(\d{1,4})\s*\**$")


def _split_into_pages(markdown: str) -> list[tuple[int, str]]:
    matches = list(_PAGE_MARKER_RE.finditer(markdown))
    if not matches:
        return [(0, markdown)]
    pages: list[tuple[int, str]] = []
    for idx, match in enumerate(matches):
        dl_idx = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        pages.append((dl_idx, markdown[start:end]))
    return pages


def _detect_footer_page(page_text: str) -> Optional[int]:
    lines = [ln.strip() for ln in page_text.strip().splitlines() if ln.strip()]
    for line in reversed(lines[-5:]):
        for pattern in (_FOOTER_LEFT_RE, _FOOTER_RIGHT_RE):
            m = pattern.match(line)
            if m:
                return int(m.group(1))
    return None


def _strip_footer(page_text: str, footer_page: int) -> str:
    lines = page_text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if (
            _FOOTER_LEFT_RE.match(stripped) or _FOOTER_RIGHT_RE.match(stripped)
        ) and str(footer_page) in stripped:
            del lines[i]
        break
    return "\n".join(lines)


def _build_footer_to_pages(
    pages: list[tuple[int, str]],
) -> dict[int, list[int]]:
    footer_to_dl: dict[int, list[int]] = {}
    for dl_idx, text in pages:
        fp = _detect_footer_page(text)
        if fp is None:
            continue
        footer_to_dl.setdefault(fp, []).append(dl_idx)
    return footer_to_dl


def _gather_section_text(
    pages_by_dl: dict[int, str],
    footer_to_dl: dict[int, list[int]],
    start_footer: int,
    end_footer: int,
) -> tuple[str, list[int]]:
    pieces: list[str] = []
    covered: list[int] = []
    for fp in range(start_footer, end_footer + 1):
        dl_indices = footer_to_dl.get(fp)
        if not dl_indices:
            continue
        for dl_idx in dl_indices:
            raw = pages_by_dl.get(dl_idx, "")
            cleaned = _strip_footer(raw, fp).strip()
            if cleaned:
                pieces.append(cleaned)
        covered.append(fp)
    return "\n\n".join(pieces), covered


def segmentation(state: DocumentProcessingState) -> dict:
    document_name = state.get("document_name") or "document"
    stem = Path(document_name).stem
    cached_path = SEGMENTATION_OUTPUT_DIR / f"{stem}.json"

    if cached_path.exists():
        cached = json.loads(cached_path.read_text(encoding="utf-8"))
        segments = cached.get("segments") or []
        log.info(
            f"Using cached segmentation at {cached_path} "
            f"({len(segments)} segments); skipping re-segmentation"
        )
        return {"segments": segments}

    # Prefer the cleaned markdown produced by preprocessing; fall back to raw
    # extraction output for safety.
    preprocessed_data = state.get("preprocessed_data") or {}
    markdown = preprocessed_data.get("markdown") or ""
    if not markdown:
        extracted_data = state.get("extracted_data") or {}
        markdown = extracted_data.get("markdown") or ""

    toc = state.get("toc") or []
    doc_id = state.get("document_id") or stem

    if not markdown:
        log.warning("No markdown in preprocessed_data or extracted_data; skipping segmentation")
        return {"segments": []}
    if not toc:
        log.warning("Empty TOC; skipping segmentation")
        return {"segments": []}

    pages = _split_into_pages(markdown)
    pages_by_dl = {dl_idx: text for dl_idx, text in pages}
    footer_to_dl = _build_footer_to_pages(pages)

    if not footer_to_dl:
        log.warning("No footer page numbers detected; cannot map TOC to pages")
        return {"segments": []}

    max_footer = max(footer_to_dl.keys())
    sorted_toc = sorted(
        [e for e in toc if isinstance(e.get("page_number"), int)],
        key=lambda e: e["page_number"],
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SEGMENT_CHUNK_SIZE,
        chunk_overlap=SEGMENT_CHUNK_OVERLAP,
    )
    encoder = tiktoken.get_encoding(TOKEN_COUNT_ENCODING)

    segments: list[dict] = []
    for i, entry in enumerate(sorted_toc):
        start_fp = entry["page_number"]
        end_fp = (
            sorted_toc[i + 1]["page_number"] - 1
            if i + 1 < len(sorted_toc)
            else max_footer
        )
        if end_fp < start_fp:
            end_fp = start_fp

        section_text, covered_pages = _gather_section_text(
            pages_by_dl, footer_to_dl, start_fp, end_fp
        )
        if not section_text:
            log.info(
                f"No content for section '{entry.get('section_name')}' "
                f"(pages {start_fp}-{end_fp}); skipping"
            )
            continue

        chunks = splitter.split_text(section_text)
        for chunk in chunks:
            segments.append(
                {
                    "seg_id": f"s_{len(segments):03d}",
                    "section_name": entry.get("section_name", ""),
                    "chapter_name": entry.get("chapter_name", ""),
                    "pages": covered_pages,
                    "token_count": len(encoder.encode(chunk, disallowed_special=())),
                    "text": chunk,
                }
            )

    output_doc = {"doc_id": doc_id, "segments": segments}
    output_path = SEGMENTATION_OUTPUT_DIR / f"{Path(document_name).stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"Wrote {len(segments)} segments to {output_path}")

    return {"segments": segments}
