"""Section worker tool — exposed to the GRA orchestrator.

`run_section_worker(section_name, sub_query)` spins a fresh ReAct sub-agent
scoped to ONE section with four typed tools. The GRA emits N parallel calls
in one assistant turn to fan out across sections; LangGraph's ToolNode
parallelizes them.
"""

import json
import logging
from typing import Callable

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from config.prompts.section_worker_prompt import SECTION_WORKER_PROMPT
from config.settings import GREP_MATCH_LIMIT, WORKER_INNER_RECURSION_LIMIT
from qna_pipeline.db.markdown import PageNotFound, get_page_text, grep
from qna_pipeline.db.mongo import (
    get_section_full,
    get_segments_meta,
)
from utils.llm import llm

log = logging.getLogger(__name__)


def _json_default(obj):
    try:
        return str(obj)
    except Exception:
        return None


async def _build_worker_tools(
    document_id: str, section_name: str
) -> tuple[list[Callable], dict, list[int]]:
    """Build the four tools for a section worker. Returns (tools, section, pages)
    so the caller can also report whether the section even exists."""
    section = await get_section_full(document_id, section_name)
    section_pages: list[int] = (
        list(section.get("pages") or []) if section else []
    )

    section_json = (
        json.dumps(section, ensure_ascii=False, default=_json_default)
        if section
        else "null"
    )

    @tool
    async def get_section_full_tool() -> str:
        """Return the full AggregateAnalysis for the assigned section: summary,
        key_entities, key_claims, decisions, risks, contradictions, metrics,
        salient_quotes, topics, pages. Call this FIRST — in most cases it
        answers the sub-question directly.
        """
        return section_json

    @tool
    async def get_segments_meta_tool() -> str:
        """Return per-segment metadata for the assigned section (seg_id,
        pages, summary, salient_quotes, topics). Use only when the section
        aggregate is too coarse and you need to localize where in the
        section something appears.
        """
        meta = await get_segments_meta(document_id, section_name)
        return json.dumps(meta, ensure_ascii=False, default=_json_default)

    @tool
    async def grep_in_section_tool(pattern: str, regex: bool = False) -> str:
        """Case-insensitive search restricted to this section's pages.
        Returns up to GREP_MATCH_LIMIT matches as JSON with
        {printed_page, line, snippet}. Use for needle-in-haystack lookups
        inside this section.

        Args:
            pattern: The text to search for. Treated literally when
                regex=False (default).
            regex: When True, treat `pattern` as a Python regex.
        """
        if not section_pages:
            return "[error: no pages known for this section — section may not exist in MongoDB]"
        matches = await grep(
            document_id,
            pattern,
            pages_filter=section_pages,
            regex=regex,
            limit=GREP_MATCH_LIMIT,
        )
        return json.dumps(matches, ensure_ascii=False, default=_json_default)

    @tool
    async def get_page_text_tool(printed_pages: list[int]) -> str:
        """Return the raw markdown text for the requested PRINTED page
        numbers (the ones in the document's footer / `pages` arrays).
        Truncated to fit a per-call token budget. Use sparingly — at most
        TWO calls per task.
        """
        try:
            return await get_page_text(document_id, printed_pages)
        except PageNotFound as e:
            return f"[error: {e}]"

    return (
        [
            get_section_full_tool,
            get_segments_meta_tool,
            grep_in_section_tool,
            get_page_text_tool,
        ],
        section or {},
        section_pages,
    )


def make_run_section_worker_tool(document_id: str) -> Callable:
    """Build a `run_section_worker` tool scoped to `document_id`."""

    @tool
    async def run_section_worker(section_name: str, sub_query: str) -> str:
        """Dispatch ONE section worker for ONE section.

        Call this multiple times IN PARALLEL (multiple tool calls in the
        same assistant turn) to fan out across the sections returned by
        `plan_sections`. Each worker is a fresh ReAct sub-agent with no
        memory of the parent conversation.

        Args:
            section_name: Exact section_name (must come from the plan).
            sub_query: Self-contained question for this worker. State what
                to extract from THIS section.

        Returns:
            The worker's final answer as a plain string.
        """
        log.info(
            f"[run_section_worker:{document_id}] section={section_name!r} "
            f"sub_query={sub_query!r}"
        )
        worker_tools, section, pages = await _build_worker_tools(
            document_id, section_name
        )
        if not section:
            return (
                f"[error: section_name={section_name!r} not found in MongoDB "
                f"for document_id={document_id!r}; verify the plan came from "
                f"plan_sections()]"
            )
        prompt = SECTION_WORKER_PROMPT.format(
            document_id=document_id, section_name=section_name
        )
        sub_agent = create_react_agent(llm._llm, worker_tools, prompt=prompt)
        result = await sub_agent.ainvoke(
            {"messages": [HumanMessage(content=sub_query)]},
            {"recursion_limit": WORKER_INNER_RECURSION_LIMIT},
        )
        answer = result["messages"][-1].content
        if not isinstance(answer, str):
            answer = str(answer)
        return answer

    return run_section_worker
