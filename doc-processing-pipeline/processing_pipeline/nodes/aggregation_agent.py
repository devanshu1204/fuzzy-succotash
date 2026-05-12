import asyncio
import json
import logging
from collections import defaultdict
from typing import Any

from langgraph.types import Send

from config.prompts.aggregation_prompt import AGGREGATION_PROMPT
from config.settings import (
    CHAPTER_AGGREGATOR_CONCURRENCY,
    SECTION_AGGREGATOR_CONCURRENCY,
)
from processing_pipeline.schemas.segment_analysis import AggregateAnalysis
from processing_pipeline.state import DocumentProcessingState
from utils.llm import llm
from utils.mongo import (
    ensure_chapter_indexes,
    ensure_section_indexes,
    get_chapter,
    get_section,
    upsert_chapter,
    upsert_section,
)

log = logging.getLogger(__name__)

# Per-level semaphores: each reduction stage gets its own budget so a slow
# section stage can't starve the chapter stage (and they never run at the
# same time anyway — they run sequentially).
_section_semaphore = asyncio.Semaphore(SECTION_AGGREGATOR_CONCURRENCY)
_chapter_semaphore = asyncio.Semaphore(CHAPTER_AGGREGATOR_CONCURRENCY)

_section_indexes_ready = False
_chapter_indexes_ready = False
_indexes_lock = asyncio.Lock()

# Shared progress counters per reduction level. Reset by each dispatcher
# at the start of its stage.
_section_progress: dict[str, int] = {"total": 0, "done": 0, "skipped": 0}
_chapter_progress: dict[str, int] = {"total": 0, "done": 0, "skipped": 0}
_progress_lock = asyncio.Lock()


async def _bump_progress(progress: dict[str, int], skipped: bool) -> tuple[int, int, int]:
    async with _progress_lock:
        if skipped:
            progress["skipped"] += 1
        else:
            progress["done"] += 1
        return progress["done"], progress["skipped"], progress["total"]


async def _ensure_section_indexes_once() -> None:
    global _section_indexes_ready
    if _section_indexes_ready:
        return
    async with _indexes_lock:
        if _section_indexes_ready:
            return
        await ensure_section_indexes()
        _section_indexes_ready = True


async def _ensure_chapter_indexes_once() -> None:
    global _chapter_indexes_ready
    if _chapter_indexes_ready:
        return
    async with _indexes_lock:
        if _chapter_indexes_ready:
            return
        await ensure_chapter_indexes()
        _chapter_indexes_ready = True


def _group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[r.get(key, "")].append(r)
    return dict(buckets)


# ---------------------------------------------------------------------------
# Stage 1: segments → sections
# ---------------------------------------------------------------------------

def aggregation_agent(state: DocumentProcessingState) -> dict:
    """Join node after all `segment_analyzer_worker` Sends complete.

    LangGraph runs this once on the merged state; the conditional edge
    below then fans out one Send per (section_name) group.
    """
    analyses = state.get("segment_analyses") or []
    groups = _group_by(analyses, "section_name")
    _section_progress["total"] = len(groups)
    _section_progress["done"] = 0
    _section_progress["skipped"] = 0
    log.info(
        f"section reduction: {len(analyses)} segment analyses → "
        f"{len(groups)} sections (concurrency={SECTION_AGGREGATOR_CONCURRENCY})"
    )
    return {}


def dispatch_sections_to_aggregators(state: DocumentProcessingState) -> list[Send]:
    analyses = state.get("segment_analyses") or []
    doc_id = state.get("doc_id") or ""
    groups = _group_by(analyses, "section_name")
    sends: list[Send] = []
    for section_name, members in groups.items():
        ordered = sorted(members, key=lambda m: m.get("seg_id", ""))
        sends.append(
            Send(
                "section_aggregator",
                {
                    "section_name": section_name,
                    "section_segments": ordered,
                    "doc_id": doc_id,
                },
            )
        )
    return sends


async def section_aggregator(state: DocumentProcessingState) -> dict:
    """Per-section LLM reduction + Mongo upsert."""
    section_name: str = state["section_name"]
    segments: list[dict] = state["section_segments"]
    doc_id: str = state.get("doc_id") or ""

    chapter_name = segments[0].get("chapter_name", "") if segments else ""
    pages = sorted({p for s in segments for p in (s.get("pages") or [])})
    segment_ids = [s.get("seg_id") for s in segments]

    await _ensure_section_indexes_once()

    existing = await get_section(doc_id, section_name)
    if existing is not None:
        done, skipped, total = await _bump_progress(_section_progress, skipped=True)
        log.info(
            f"[section/{section_name!r}] cached in MongoDB; skipping LLM "
            f"(progress: {done + skipped}/{total} — {done} aggregated, {skipped} skipped)"
        )
        return {"section_analyses": [existing]}

    payload = {
        "level": "section",
        "name": section_name,
        "chapter_name": chapter_name,
        "child_analyses": segments,
    }

    structured_llm = llm.with_structured_output(AggregateAnalysis)
    async with _section_semaphore:
        log.info(
            f"[section/{section_name!r}] reducing {len(segments)} segments "
            f"(pages={pages[:5]}{'...' if len(pages) > 5 else ''})"
        )
        analysis: AggregateAnalysis = await structured_llm.ainvoke(
            [
                {"role": "system", "content": AGGREGATION_PROMPT},
                {"role": "user", "content": json.dumps(payload, indent=2, default=str)},
            ]
        )

    record: dict[str, Any] = {
        **analysis.model_dump(),
        "doc_id": doc_id,
        "section_name": section_name,
        "chapter_name": chapter_name,
        "pages": pages,
        "segment_ids": segment_ids,
    }

    await upsert_section(record)
    done, skipped, total = await _bump_progress(_section_progress, skipped=False)
    log.info(
        f"[section/{section_name!r}] upserted to MongoDB "
        f"(progress: {done + skipped}/{total} — {done} aggregated, {skipped} skipped)"
    )

    return {"section_analyses": [record]}


# ---------------------------------------------------------------------------
# Stage 2: sections → chapters
# ---------------------------------------------------------------------------

def chapter_aggregator_dispatch(state: DocumentProcessingState) -> dict:
    """Join node after all `section_aggregator` Sends complete."""
    analyses = state.get("section_analyses") or []
    groups = _group_by(analyses, "chapter_name")
    _chapter_progress["total"] = len(groups)
    _chapter_progress["done"] = 0
    _chapter_progress["skipped"] = 0
    log.info(
        f"chapter reduction: {len(analyses)} section analyses → "
        f"{len(groups)} chapters (concurrency={CHAPTER_AGGREGATOR_CONCURRENCY})"
    )
    return {}


def dispatch_chapters_to_aggregators(state: DocumentProcessingState) -> list[Send]:
    analyses = state.get("section_analyses") or []
    doc_id = state.get("doc_id") or ""
    groups = _group_by(analyses, "chapter_name")
    sends: list[Send] = []
    for chapter_name, members in groups.items():
        ordered = sorted(members, key=lambda m: min(m.get("pages") or [10**9]))
        sends.append(
            Send(
                "chapter_aggregator",
                {
                    "chapter_name": chapter_name,
                    "chapter_sections": ordered,
                    "doc_id": doc_id,
                },
            )
        )
    return sends


async def chapter_aggregator(state: DocumentProcessingState) -> dict:
    """Per-chapter LLM reduction + Mongo upsert."""
    chapter_name: str = state["chapter_name"]
    sections: list[dict] = state["chapter_sections"]
    doc_id: str = state.get("doc_id") or ""

    section_names = [s.get("section_name") for s in sections]
    pages = sorted({p for s in sections for p in (s.get("pages") or [])})

    await _ensure_chapter_indexes_once()

    existing = await get_chapter(doc_id, chapter_name)
    if existing is not None:
        done, skipped, total = await _bump_progress(_chapter_progress, skipped=True)
        log.info(
            f"[chapter/{chapter_name!r}] cached in MongoDB; skipping LLM "
            f"(progress: {done + skipped}/{total} — {done} aggregated, {skipped} skipped)"
        )
        return {"chapter_analyses": [existing]}

    payload = {
        "level": "chapter",
        "name": chapter_name,
        "child_analyses": sections,
    }

    structured_llm = llm.with_structured_output(AggregateAnalysis)
    async with _chapter_semaphore:
        log.info(
            f"[chapter/{chapter_name!r}] reducing {len(sections)} sections "
            f"(pages={pages[:5]}{'...' if len(pages) > 5 else ''})"
        )
        analysis: AggregateAnalysis = await structured_llm.ainvoke(
            [
                {"role": "system", "content": AGGREGATION_PROMPT},
                {"role": "user", "content": json.dumps(payload, indent=2, default=str)},
            ]
        )

    record: dict[str, Any] = {
        **analysis.model_dump(),
        "doc_id": doc_id,
        "chapter_name": chapter_name,
        "section_names": section_names,
        "pages": pages,
    }

    await upsert_chapter(record)
    done, skipped, total = await _bump_progress(_chapter_progress, skipped=False)
    log.info(
        f"[chapter/{chapter_name!r}] upserted to MongoDB "
        f"(progress: {done + skipped}/{total} — {done} aggregated, {skipped} skipped)"
    )

    return {"chapter_analyses": [record]}
