"""Typed PyMongo helpers for the GRA sub-agents.

These give bounded, projection-limited reads against the sections, chapters,
and segments collections — predictable output sizes are essential for keeping
each sub-agent inside the 128K token budget.

Connection is module-level and lazy. All public helpers are async wrappers
over sync PyMongo calls via `asyncio.to_thread` so they cooperate with
LangGraph's async tool execution.
"""

import asyncio
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection

from config.settings import (
    MONGODB_CHAPTERS_COLLECTION,
    MONGODB_DB,
    MONGODB_SECTIONS_COLLECTION,
    MONGODB_SEGMENTS_COLLECTION,
    MONGODB_URI,
)

_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        if not MONGODB_URI:
            raise RuntimeError(
                "MONGODB_URI is not set. Add it to qna-pipeline/.env before "
                "running the GRA sub-agents."
            )
        _client = MongoClient(MONGODB_URI)
    return _client


def _sections() -> Collection:
    return _get_client()[MONGODB_DB][MONGODB_SECTIONS_COLLECTION]


def _chapters() -> Collection:
    return _get_client()[MONGODB_DB][MONGODB_CHAPTERS_COLLECTION]


def _segments() -> Collection:
    return _get_client()[MONGODB_DB][MONGODB_SEGMENTS_COLLECTION]


# ---------------------------------------------------------------------------
# Sync core
# ---------------------------------------------------------------------------

def _list_sections_sync(doc_id: str) -> list[dict]:
    cursor = _sections().find(
        {"doc_id": doc_id},
        projection={
            "_id": 0,
            "section_name": 1,
            "chapter_name": 1,
            "summary": 1,
            "pages": 1,
        },
    )
    return list(cursor)


def _list_chapters_sync(doc_id: str) -> list[dict]:
    cursor = _chapters().find({"doc_id": doc_id}, projection={"_id": 0})
    return list(cursor)


def _list_chapter_summaries_sync(doc_id: str) -> list[dict]:
    cursor = _chapters().find(
        {"doc_id": doc_id},
        projection={
            "_id": 0,
            "chapter_name": 1,
            "summary": 1,
            "pages": 1,
            "section_names": 1,
        },
    )
    return list(cursor)


def _get_section_full_sync(doc_id: str, section_name: str) -> Optional[dict]:
    record = _sections().find_one(
        {"doc_id": doc_id, "section_name": section_name},
        projection={"_id": 0},
    )
    return record


def _get_chapter_full_sync(doc_id: str, chapter_name: str) -> Optional[dict]:
    record = _chapters().find_one(
        {"doc_id": doc_id, "chapter_name": chapter_name},
        projection={"_id": 0},
    )
    return record


def _get_segments_meta_sync(doc_id: str, section_name: str) -> list[dict]:
    cursor = _segments().find(
        {"doc_id": doc_id, "section_name": section_name},
        projection={
            "_id": 0,
            "seg_id": 1,
            "pages": 1,
            "summary": 1,
            "salient_quotes": 1,
            "topics": 1,
        },
        sort=[("seg_id", 1)],
    )
    return list(cursor)


# ---------------------------------------------------------------------------
# Async wrappers (public)
# ---------------------------------------------------------------------------

async def list_sections(doc_id: str) -> list[dict]:
    """Section summaries used by the planner sub-agent.

    Returns: list of {section_name, chapter_name, summary, pages}.
    """
    return await asyncio.to_thread(_list_sections_sync, doc_id)


async def list_chapters(doc_id: str) -> list[dict]:
    """Full chapter aggregates used by the document agent.

    Returns: list of full AggregateAnalysis dicts (contradictions, decisions,
    risks, key_entities, summary, etc.).
    """
    return await asyncio.to_thread(_list_chapters_sync, doc_id)


async def list_chapter_summaries(doc_id: str) -> list[dict]:
    """Lightweight chapter index used as a fallback by the planner when the
    section list is too large.

    Returns: list of {chapter_name, summary, pages, section_names}.
    """
    return await asyncio.to_thread(_list_chapter_summaries_sync, doc_id)


async def get_section_full(doc_id: str, section_name: str) -> Optional[dict]:
    """Full section AggregateAnalysis dict. Worker entry point — start here
    before drilling into segments or raw page text.
    """
    return await asyncio.to_thread(_get_section_full_sync, doc_id, section_name)


async def get_chapter_full(doc_id: str, chapter_name: str) -> Optional[dict]:
    """Full chapter AggregateAnalysis dict for the document agent."""
    return await asyncio.to_thread(_get_chapter_full_sync, doc_id, chapter_name)


async def get_segments_meta(doc_id: str, section_name: str) -> list[dict]:
    """Per-segment metadata for a section (no raw text — that comes from the
    markdown helpers). Used by workers that need finer-grained navigation
    within a section's pages.
    """
    return await asyncio.to_thread(_get_segments_meta_sync, doc_id, section_name)
