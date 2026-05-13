import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from langgraph.types import Send

from config.prompts.doc_processing.segment_analyzer_prompt import SEGMENT_ANALYZER_PROMPT
from config.settings import (
    SEGMENT_ANALYZER_CONCURRENCY,
    SEGMENTATION_OUTPUT_DIR,
)
from processing_pipeline.schemas.segment_analysis import SegmentAnalysis
from processing_pipeline.state import DocumentProcessingState
from utils.llm import llm
from utils.mongo import ensure_indexes, get_segment, upsert_segment

log = logging.getLogger(__name__)

# Module-level semaphore so all parallel workers within a single graph run
# share one budget — caps in-flight LLM calls at SEGMENT_ANALYZER_CONCURRENCY
# regardless of how many Sends the dispatcher fans out.
_llm_semaphore = asyncio.Semaphore(SEGMENT_ANALYZER_CONCURRENCY)
_indexes_ready = False
_indexes_lock = asyncio.Lock()

# Shared progress counters across parallel workers within one graph run.
# Reset by the dispatcher at the start of every run.
_progress: dict[str, int] = {"total": 0, "done": 0, "skipped": 0}
_progress_lock = asyncio.Lock()


async def _bump_progress(skipped: bool) -> tuple[int, int, int]:
    async with _progress_lock:
        if skipped:
            _progress["skipped"] += 1
        else:
            _progress["done"] += 1
        return _progress["done"], _progress["skipped"], _progress["total"]


async def _ensure_indexes_once() -> None:
    global _indexes_ready
    if _indexes_ready:
        return
    async with _indexes_lock:
        if _indexes_ready:
            return
        await ensure_indexes()
        _indexes_ready = True


def _load_segments_from_disk(document_name: str) -> tuple[str, list[dict]]:
    stem = Path(document_name).stem
    path = SEGMENTATION_OUTPUT_DIR / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Segmentation output not found at {path}. "
            f"Run the segmentation node first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("doc_id") or stem, data.get("segments") or []


def segment_analyzer_agent(state: DocumentProcessingState) -> dict:
    """Dispatcher: loads the segmentation JSON from disk and stashes the
    `doc_id` + segment list on state. The actual fan-out happens via a
    conditional edge that emits one `Send` per segment to the worker node.
    """
    document_name = state.get("document_name") or "document"
    doc_id, segments = _load_segments_from_disk(document_name)
    _progress["total"] = len(segments)
    _progress["done"] = 0
    _progress["skipped"] = 0
    log.info(
        f"Dispatching {len(segments)} segments to analyzer "
        f"(concurrency={SEGMENT_ANALYZER_CONCURRENCY})"
    )
    return {"doc_id": doc_id, "segments": segments}


def dispatch_segments_to_workers(state: DocumentProcessingState) -> list[Send]:
    """Conditional edge: emit one Send per segment to the worker node.

    LangGraph will create N parallel branches; the worker's module-level
    Semaphore enforces the 10-at-a-time cap.
    """
    segments = state.get("segments") or []
    doc_id = state.get("doc_id") or ""
    return [
        Send(
            "segment_analyzer_worker",
            {"segment": segment, "doc_id": doc_id},
        )
        for segment in segments
    ]


async def segment_analyzer_worker(state: DocumentProcessingState) -> dict:
    """Per-segment LLM call with structured output + Mongo upsert.

    Runs in parallel across segments (one branch per Send). The shared
    semaphore caps concurrent LLM calls; the DB upsert keys on
    (doc_id, seg_id) so re-running the pipeline replaces prior analyses.

    Identity fields (doc_id, seg_id, chapter_name, section_name, pages,
    token_count) are NEVER asked of the LLM — they are copied verbatim
    from the segment dict produced by segmentation. The LLM only fills
    the analytical fields defined by `SegmentAnalysis`.
    """
    segment = state["segment"]
    doc_id = state.get("doc_id") or ""
    seg_id = segment.get("seg_id")

    await _ensure_indexes_once()

    existing = await get_segment(doc_id, seg_id)
    if existing is not None:
        done, skipped, total = await _bump_progress(skipped=True)
        log.info(
            f"[{doc_id}/{seg_id}] cached in MongoDB; skipping LLM "
            f"(progress: {done + skipped}/{total} — {done} analyzed, {skipped} skipped)"
        )
        return {"segment_analyses": [existing]}

    structured_llm = llm.with_structured_output(SegmentAnalysis)

    async with _llm_semaphore:
        log.info(f"[{doc_id}/{seg_id}] analyzing segment ({segment.get('token_count')} tok)")
        analysis: SegmentAnalysis = await structured_llm.ainvoke(
            [
                {"role": "system", "content": SEGMENT_ANALYZER_PROMPT},
                {"role": "user", "content": segment.get("text", "")},
            ]
        )

    # Build the record explicitly: LLM output first, then passthrough
    # identity fields which always win — so if a schema field ever
    # collides with a passthrough key in the future, the real
    # segmentation value still survives.
    record: dict[str, Any] = {
        **analysis.model_dump(),
        "doc_id": doc_id,
        "seg_id": seg_id,
        "chapter_name": segment.get("chapter_name", ""),
        "section_name": segment.get("section_name", ""),
        "pages": segment.get("pages", []),
        "token_count": segment.get("token_count"),
    }

    await upsert_segment(record)
    done, skipped, total = await _bump_progress(skipped=False)
    log.info(
        f"[{doc_id}/{seg_id}] upserted to MongoDB "
        f"(progress: {done + skipped}/{total} — {done} analyzed, {skipped} skipped)"
    )

    return {"segment_analyses": [record]}
