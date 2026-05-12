import logging
import re
from pathlib import Path

from datalab_sdk import DatalabClient, ConvertOptions

from config.settings import DATALAB_API_KEY, EXTRACTION_OUTPUT_DIR, INPUT_DIR
from processing_pipeline.state import DocumentProcessingState

log = logging.getLogger(__name__)

_PAGE_MARKER_RE = re.compile(r"\{\d+\}-{3,}")


def _resolve_document_path(document_name: str) -> Path:
    document_path = INPUT_DIR / document_name
    if not document_path.exists():
        raise FileNotFoundError(
            f"Document not found at {document_path}. "
            f"Ensure '{document_name}' exists in {INPUT_DIR}"
        )
    return document_path


def extraction(state: DocumentProcessingState) -> dict:
    document_name = state["document_name"]
    output_stem = Path(document_name).stem
    cached_path = EXTRACTION_OUTPUT_DIR / f"{output_stem}.md"

    if cached_path.exists():
        markdown = cached_path.read_text(encoding="utf-8")
        page_count = len(_PAGE_MARKER_RE.findall(markdown))
        log.info(
            f"Using cached extraction at {cached_path} "
            f"(page_count={page_count}); skipping Datalab"
        )
        return {
            "extracted_data": {
                "markdown": markdown,
                "page_count": page_count,
                "metadata": {},
            }
        }

    document_path = _resolve_document_path(document_name)

    log.info(f"Starting Datalab OCR extraction for: {document_name}")

    client = DatalabClient(api_key=DATALAB_API_KEY)

    options = ConvertOptions(
        output_format="markdown",
        mode="accurate",
        paginate=True,
        additional_config={
            "keep_pageheader_in_output": True,
            "keep_pagefooter_in_output": True,
        },
    )

    result = client.convert(str(document_path), options=options)

    if not result.success:
        error_msg = result.error or "Unknown error during Datalab OCR conversion"
        log.error(f"Datalab OCR failed: {error_msg}")
        raise RuntimeError(f"Datalab OCR conversion failed: {error_msg}")

    extracted_data = {
        "markdown": result.markdown,
        "page_count": result.page_count,
        "metadata": result.metadata,
    }

    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_text(result.markdown or "", encoding="utf-8")

    log.info(f"Saved extraction output to: {cached_path}")

    log.info(
        f"Datalab OCR extraction complete: {result.page_count} pages processed for {document_name}"
    )

    return {"extracted_data": extracted_data}