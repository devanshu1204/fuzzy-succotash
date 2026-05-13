"""Document-level agent tool — exposed to the GRA orchestrator.

`query_document(query)` spins a fresh ReAct sub-agent whose ONLY data
source is the chapters collection (~10-20 records for a 500-page doc).
Use this for doc-wide queries: full summaries, cross-chapter contradictions,
enumeration of all risks / decisions / entities.
"""

import json
import logging
from typing import Callable

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from config.prompts.qna.document_agent_prompt import DOCUMENT_AGENT_PROMPT
from config.settings import DOCUMENT_AGENT_RECURSION_LIMIT
from qna_pipeline.db.mongo import (
    get_chapter_full,
    list_chapter_summaries,
    list_chapters,
)
from utils.llm import llm

log = logging.getLogger(__name__)


def _json_default(obj):
    try:
        return str(obj)
    except Exception:
        return None


def _build_document_agent_tools(document_id: str) -> list[Callable]:
    @tool
    async def list_chapter_summaries_tool() -> str:
        """List every chapter with `chapter_name`, `summary`, `pages`,
        `section_names`. Cheap orientation call — start here for most
        document-level questions.
        """
        chapters = await list_chapter_summaries(document_id)
        return json.dumps(chapters, ensure_ascii=False, default=_json_default)

    @tool
    async def list_chapters_full_tool() -> str:
        """Return every chapter's FULL AggregateAnalysis (summary, key_entities,
        key_claims, decisions, risks, contradictions, metrics, salient_quotes,
        topics). Expensive — call AT MOST ONCE per question, only when you
        need to enumerate or compare typed arrays across chapters.
        """
        chapters = await list_chapters(document_id)
        return json.dumps(chapters, ensure_ascii=False, default=_json_default)

    @tool
    async def get_chapter_full_tool(chapter_name: str) -> str:
        """Return ONE chapter's full AggregateAnalysis. Use when you need
        to inspect a specific chapter in detail after orienting with
        `list_chapter_summaries`.

        Args:
            chapter_name: Exact chapter_name from `list_chapter_summaries`.
        """
        chapter = await get_chapter_full(document_id, chapter_name)
        if chapter is None:
            return f"[error: chapter_name={chapter_name!r} not found]"
        return json.dumps(chapter, ensure_ascii=False, default=_json_default)

    return [
        list_chapter_summaries_tool,
        list_chapters_full_tool,
        get_chapter_full_tool,
    ]


def make_query_document_tool(document_id: str) -> Callable:
    """Build a `query_document` tool scoped to `document_id`."""

    @tool
    async def query_document(query: str) -> str:
        """Dispatch the document-level agent for a doc-wide question.

        Use when the question spans the WHOLE document:
            - full / executive summary,
            - "list every risk / decision / entity",
            - "any contradictions across the document",
            - cross-chapter consistency checks.

        Call this directly when `plan_sections` returns an empty `tasks` list,
        or when the routing heuristics in your prompt say to.

        Args:
            query: Self-contained question for the document agent. It has no
                memory of the parent conversation.

        Returns:
            The document agent's final answer as a plain string.
        """
        log.info(f"[query_document:{document_id}] query={query!r}")
        agent_tools = _build_document_agent_tools(document_id)
        prompt = DOCUMENT_AGENT_PROMPT.format(document_id=document_id)
        sub_agent = create_react_agent(llm._llm, agent_tools, prompt=prompt)
        result = await sub_agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            {"recursion_limit": DOCUMENT_AGENT_RECURSION_LIMIT},
        )
        answer = result["messages"][-1].content
        if not isinstance(answer, str):
            answer = str(answer)
        return answer

    return query_document
