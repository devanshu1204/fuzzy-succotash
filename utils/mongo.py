import asyncio
import logging
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

log = logging.getLogger(__name__)

_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        if not MONGODB_URI:
            raise RuntimeError(
                "MONGODB_URI is not set. Add it to .env before running the pipeline."
            )
        _client = MongoClient(MONGODB_URI)
    return _client


def get_segments_collection() -> Collection:
    return _get_client()[MONGODB_DB][MONGODB_SEGMENTS_COLLECTION]


def get_sections_collection() -> Collection:
    return _get_client()[MONGODB_DB][MONGODB_SECTIONS_COLLECTION]


def get_chapters_collection() -> Collection:
    return _get_client()[MONGODB_DB][MONGODB_CHAPTERS_COLLECTION]


def _ensure_segment_indexes_sync() -> None:
    get_segments_collection().create_index(
        [("doc_id", 1), ("seg_id", 1)], unique=True, name="doc_seg_unique"
    )


def _ensure_section_indexes_sync() -> None:
    get_sections_collection().create_index(
        [("doc_id", 1), ("section_name", 1)], unique=True, name="doc_section_unique"
    )


def _ensure_chapter_indexes_sync() -> None:
    get_chapters_collection().create_index(
        [("doc_id", 1), ("chapter_name", 1)], unique=True, name="doc_chapter_unique"
    )


def _get_segment_sync(doc_id: str, seg_id: str) -> Optional[dict]:
    record = get_segments_collection().find_one(
        {"doc_id": doc_id, "seg_id": seg_id}
    )
    if record is not None:
        record.pop("_id", None)
    return record


def _upsert_segment_sync(record: dict) -> None:
    get_segments_collection().update_one(
        {"doc_id": record["doc_id"], "seg_id": record["seg_id"]},
        {"$set": record},
        upsert=True,
    )


def _get_section_sync(doc_id: str, section_name: str) -> Optional[dict]:
    record = get_sections_collection().find_one(
        {"doc_id": doc_id, "section_name": section_name}
    )
    if record is not None:
        record.pop("_id", None)
    return record


def _get_chapter_sync(doc_id: str, chapter_name: str) -> Optional[dict]:
    record = get_chapters_collection().find_one(
        {"doc_id": doc_id, "chapter_name": chapter_name}
    )
    if record is not None:
        record.pop("_id", None)
    return record


def _upsert_section_sync(record: dict) -> None:
    get_sections_collection().update_one(
        {"doc_id": record["doc_id"], "section_name": record["section_name"]},
        {"$set": record},
        upsert=True,
    )


def _upsert_chapter_sync(record: dict) -> None:
    get_chapters_collection().update_one(
        {"doc_id": record["doc_id"], "chapter_name": record["chapter_name"]},
        {"$set": record},
        upsert=True,
    )


async def ensure_segment_indexes() -> None:
    await asyncio.to_thread(_ensure_segment_indexes_sync)


async def ensure_section_indexes() -> None:
    await asyncio.to_thread(_ensure_section_indexes_sync)


async def ensure_chapter_indexes() -> None:
    await asyncio.to_thread(_ensure_chapter_indexes_sync)


async def get_segment(doc_id: str, seg_id: str) -> Optional[dict]:
    return await asyncio.to_thread(_get_segment_sync, doc_id, seg_id)


async def upsert_segment(record: dict) -> None:
    await asyncio.to_thread(_upsert_segment_sync, record)


async def get_section(doc_id: str, section_name: str) -> Optional[dict]:
    return await asyncio.to_thread(_get_section_sync, doc_id, section_name)


async def get_chapter(doc_id: str, chapter_name: str) -> Optional[dict]:
    return await asyncio.to_thread(_get_chapter_sync, doc_id, chapter_name)


async def upsert_section(record: dict) -> None:
    await asyncio.to_thread(_upsert_section_sync, record)


async def upsert_chapter(record: dict) -> None:
    await asyncio.to_thread(_upsert_chapter_sync, record)


# Backwards-compat alias used by segment_analyzer_agent.
ensure_indexes = ensure_segment_indexes
