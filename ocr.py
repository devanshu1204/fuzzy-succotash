"""
Annual Report PDF Parser using Docling + OpenAI VLM for image/chart annotation.

Pipeline:
  - Standard Docling PDF pipeline for text, tables (TableFormer), layout
  - OpenAI Vision API for annotating charts/figures extracted from the PDF
  - Output: Markdown file + separate images folder

Usage:
  1. Set INPUT_PDF_PATH to your PDF file path
  2. Set OPENAI_API_KEY to your OpenAI API key
  3. Run: python parse_annual_report.py
"""

import os
import time
import logging
from pathlib import Path

# ─── CONFIGURATION — edit these ───────────────────────────────────────────────

INPUT_PDF_PATH = "ICICI.pdf"        # Path to your PDF file
OUTPUT_DIR     = "output"                   # Folder where results will be saved
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")                   # Your OpenAI API key

# OpenAI model to use for image/chart description
# Options: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"
# gpt-4o-mini is cheapest and good enough for chart descriptions
OPENAI_MODEL = "gpt-4o-mini"

# Prompt sent to OpenAI for each extracted chart/figure
PICTURE_PROMPT = (
    "You are analyzing a figure or chart extracted from a corporate annual report. "
    "Describe what this image shows in detail. If it is a chart or graph, mention: "
    "the type of chart, the metrics shown, key values, trends, and any notable highlights. "
    "If it is a table screenshot or infographic, extract the key data points. "
    "Be concise but thorough — 3 to 6 sentences."
)

# Image resolution scale (1.0 = default, 2.0 = higher quality, slower)
IMAGE_RESOLUTION_SCALE = 1.0

# Whether to run OCR (set True if your PDF has scanned pages)
DO_OCR = False

# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_converter():
    """Build the Docling DocumentConverter with OpenAI picture annotation."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        PictureDescriptionApiOptions,
        TableStructureOptions,
    )

    # Configure picture annotation via OpenAI
    picture_options = PictureDescriptionApiOptions(
        url="https://api.openai.com/v1/chat/completions",
        params=dict(
            model=OPENAI_MODEL,
            max_completion_tokens=400,
            seed=42,
        ),
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        prompt=PICTURE_PROMPT,
        timeout=60,
        concurrency=5,
    )

    # Build the main PDF pipeline options
    pipeline_options = PdfPipelineOptions()

    # Table structure recognition (critical for financial tables)
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True   # precisely match text cells to table structure
    )

    # OCR (enable if you have scanned pages)
    pipeline_options.do_ocr = DO_OCR

    # Image extraction
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = False       # don't need full page images
    pipeline_options.generate_picture_images = True     # extract figures/charts

    # Required to allow calls to external APIs (e.g. OpenAI)
    pipeline_options.enable_remote_services = True

    # Attach the OpenAI annotator
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = picture_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter


def get_page_labels(pdf_path: Path) -> dict:
    """Returns {physical_page_no (1-indexed): display_label}.
    Falls back to str(physical_page_no) when the PDF has no page labels."""
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    labels = {}
    for i in range(len(doc)):
        label = doc[i].get_label().strip()
        labels[i + 1] = label if label else str(i + 1)
    doc.close()
    return labels


def extract_footer_page_numbers(conv_result) -> dict:
    """Scan Docling PAGE_FOOTER elements for printed page numbers via regex.
    Returns {physical_page_no: detected_page_label_str}."""
    import re
    from docling_core.types.doc.document import TextItem

    try:
        from docling_core.types.doc.labels import DocItemLabel
        footer_label = DocItemLabel.PAGE_FOOTER
    except (ImportError, AttributeError):
        footer_label = None

    # Patterns tried in priority order — stop at first match per page
    _PATTERNS = [
        r"^\s*(\d+)\s*$",               # lone number on the line
        r"[Pp]age\s+(\d+)",             # "Page N" / "page N"
        r"(?:^|\|)\s*(\d+)\s*(?:\||$)", # number bracketed by pipes
        r"(\d+)\s*$",                   # number at end of string
        r"^\s*(\d+)",                   # number at start of string
    ]

    page_numbers = {}
    for element, _level in conv_result.document.iterate_items():
        if not isinstance(element, TextItem):
            continue
        if footer_label is not None and element.label != footer_label:
            continue
        if not (hasattr(element, "prov") and element.prov):
            continue

        page_no = element.prov[0].page_no
        if page_no in page_numbers:
            continue

        text = (element.text or "").strip()
        if not text:
            continue

        for pattern in _PATTERNS:
            m = re.search(pattern, text)
            if m:
                num = int(m.group(1))
                if 1 <= num <= 9999:    # sanity-check: skip years, large numbers
                    page_numbers[page_no] = str(num)
                    log.debug(f"Footer page number: physical={page_no} → '{num}' (from: {text!r})")
                break

    log.info(f"Footer page-number detection: found labels for {len(page_numbers)} page(s)")
    return page_numbers


def save_outputs(conv_result, output_dir: Path, page_labels: dict):
    """Save page-wise markdown + images to output directory."""
    from docling_core.types.doc.document import (
        PictureItem, TableItem, SectionHeaderItem, TextItem, ListItem, CodeItem,
    )

    try:
        from docling_core.types.doc.labels import DocItemLabel
        _SKIP_LABELS = {DocItemLabel.PAGE_FOOTER, DocItemLabel.PAGE_HEADER}
    except (ImportError, AttributeError):
        _SKIP_LABELS = set()

    doc_stem = conv_result.input.file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / f"{doc_stem}_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    picture_count = 0
    table_count = 0
    current_page = None
    md_lines = []

    for element, _level in conv_result.document.iterate_items():
        # ── Detect page change ────────────────────────────────────────────────
        page_no = None
        if hasattr(element, "prov") and element.prov:
            page_no = element.prov[0].page_no

        if page_no is not None and page_no != current_page:
            current_page = page_no
            display = page_labels.get(page_no, str(page_no))
            md_lines.append(f"\n\n---\n\n## Page {display}\n\n")

        # ── Skip page headers / footers from body content ────────────────────
        if _SKIP_LABELS and hasattr(element, "label") and element.label in _SKIP_LABELS:
            continue

        # ── Render element ────────────────────────────────────────────────────
        if isinstance(element, PictureItem) and element.image:
            picture_count += 1
            img_path = images_dir / f"figure-{picture_count}.png"
            with img_path.open("wb") as f:
                element.image.pil_image.save(f, format="PNG")
            md_lines.append(f"![Figure {picture_count}]({images_dir.name}/figure-{picture_count}.png)\n\n")
            for ann in getattr(element, "annotations", None) or []:
                if hasattr(ann, "text") and ann.text:
                    md_lines.append(f"*{ann.text}*\n\n")

        elif isinstance(element, TableItem):
            if element.image:
                table_count += 1
                img_path = images_dir / f"table-{table_count}.png"
                with img_path.open("wb") as f:
                    element.image.pil_image.save(f, format="PNG")
            try:
                md_lines.append(element.export_to_markdown() + "\n\n")
            except Exception:
                if element.image:
                    md_lines.append(f"![Table {table_count}]({images_dir.name}/table-{table_count}.png)\n\n")

        elif isinstance(element, SectionHeaderItem):
            h = max(1, getattr(element, "level", 1))
            md_lines.append(f"{'#' * h} {element.text}\n\n")

        elif isinstance(element, ListItem):
            enumerated = getattr(element, "enumerated", False)
            idx = getattr(element, "index", 1)
            marker = f"{idx}." if enumerated else "-"
            md_lines.append(f"{marker} {element.text}\n")

        elif isinstance(element, CodeItem):
            lang = getattr(element, "code_language", "") or ""
            md_lines.append(f"```{lang}\n{element.text}\n```\n\n")

        elif isinstance(element, TextItem) and element.text:
            md_lines.append(f"{element.text}\n\n")

    log.info(f"Saved {picture_count} figure(s) and {table_count} table image(s) → {images_dir}")

    # ── Write page-wise markdown ──────────────────────────────────────────────
    md_path = output_dir / f"{doc_stem}.md"
    md_path.write_text("".join(md_lines), encoding="utf-8")
    log.info(f"Saved page-wise markdown → {md_path}")

    # ── Save JSON (useful for downstream RAG chunking) ────────────────────────
    json_path = output_dir / f"{doc_stem}.json"
    conv_result.document.save_as_json(json_path)
    log.info(f"Saved JSON → {json_path}")

    return md_path


def main():
    input_path = Path(INPUT_PDF_PATH)
    output_dir = Path(OUTPUT_DIR)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not input_path.exists():
        raise FileNotFoundError(f"PDF not found: {input_path.resolve()}")

    if OPENAI_API_KEY.startswith("sk-..."):
        raise ValueError("Please set your OPENAI_API_KEY in the script configuration.")

    log.info(f"Input  : {input_path.resolve()}")
    log.info(f"Output : {output_dir.resolve()}")
    log.info(f"Model  : {OPENAI_MODEL}")

    # ── Build converter ───────────────────────────────────────────────────────
    log.info("Initializing Docling converter...")
    converter = build_converter()

    # ── Convert ───────────────────────────────────────────────────────────────
    log.info("Converting PDF (this may take a while for large reports)...")
    start = time.time()
    result = converter.convert(str(input_path))
    elapsed = time.time() - start
    log.info(f"Conversion complete in {elapsed:.1f}s")

    # ── Check result ──────────────────────────────────────────────────────────
    if result is None or result.document is None:
        raise RuntimeError("Conversion failed — no document returned.")

    pages = len(result.document.pages) if result.document.pages else "?"
    log.info(f"Parsed {pages} page(s)")

    # ── Resolve page labels: pymupdf fallback → footer detection override ─────
    page_labels = get_page_labels(input_path)
    footer_numbers = extract_footer_page_numbers(result)
    page_labels.update(footer_numbers)   # footer-detected wins over pymupdf
    md_path = save_outputs(result, output_dir, page_labels)

    print("\n" + "═" * 60)
    print(f"  ✅  Done!")
    print(f"  📄  Markdown : {md_path}")
    print(f"  📁  Output   : {output_dir.resolve()}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()