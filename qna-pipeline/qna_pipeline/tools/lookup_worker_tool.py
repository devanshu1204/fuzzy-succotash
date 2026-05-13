"""Lookup worker tool — dispatched by the LookupAgent orchestrator.

`run_lookup_worker(sub_query, pages_filter=None)` spins a fresh ReAct
sub-agent with three retrieval tools (`grep`, `get_page_text`,
`list_pages`) scoped to one document. The orchestrator emits N parallel
calls in one assistant turn to fan out across independent lookup
sub-questions; LangGraph's ToolNode parallelizes them.

When the orchestrator supplies a `pages_filter` at dispatch time, it
becomes the default scope for the worker's grep — the worker can still
override per call by passing an explicit `pages_filter`.
"""

import json
import logging
from typing import Callable, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from config.prompts.lookup_worker_prompt import LOOKUP_WORKER_PROMPT
from config.settings import GREP_MATCH_LIMIT, WORKER_INNER_RECURSION_LIMIT
from qna_pipeline.db.markdown import PageNotFound
from qna_pipeline.db.markdown import get_page_text as get_page_text_impl
from qna_pipeline.db.markdown import grep as grep_impl
from qna_pipeline.tools.grep_tools import make_list_pages_tool
from utils.llm import llm

log = logging.getLogger(__name__)


def _json_default(obj):
    try:
        return str(obj)
    except Exception:
        return None


def _build_worker_tools(
    document_id: str, dispatch_pages_filter: Optional[list[int]]
) -> list[Callable]:
    """Build the three tools for one lookup worker.

    `dispatch_pages_filter` is the page-scope hint from the orchestrator;
    it becomes the default for the worker's grep when the worker does not
    pass its own `pages_filter`.
    """

    @tool
    async def grep(
        pattern: str,
        pages_filter: Optional[list[int]] = None,
        regex: bool = False,
        limit: int = GREP_MATCH_LIMIT,
    ) -> str:
        """Case-insensitive search over the preprocessed markdown for this
        document. Returns up to `limit` matches.

        Args:
            pattern: Text to find. Treated literally (regex-escaped) when
                regex=False (default).
            pages_filter: Optional list of PRINTED page numbers to restrict
                the search to. If omitted, falls back to the orchestrator's
                dispatch-time hint (if any); pass an explicit list to
                override, or pass an empty list to force doc-wide.
            regex: When True, treat `pattern` as a Python regex.
            limit: Max matches to return (capped at GREP_MATCH_LIMIT).

        Returns:
            JSON array of {printed_page, physical_page, line, snippet}.
            Empty array if no matches.
        """
        effective_filter: Optional[list[int]]
        if pages_filter is None:
            effective_filter = dispatch_pages_filter
        elif len(pages_filter) == 0:
            effective_filter = None
        else:
            effective_filter = pages_filter
        log.info(
            f"[lookup_worker_grep:{document_id}] pattern={pattern!r} "
            f"pages_filter={effective_filter} regex={regex} limit={limit}"
        )
        matches = await grep_impl(
            document_id,
            pattern,
            pages_filter=effective_filter,
            regex=regex,
            limit=limit,
        )
        return json.dumps(matches, ensure_ascii=False, default=_json_default)

    @tool
    async def get_page_text(printed_pages: list[int]) -> str:
        """Return the raw markdown text for the requested PRINTED page
        numbers. Output is truncated to fit a per-call token budget.

        Args:
            printed_pages: List of printed page numbers to retrieve.

        Returns:
            The joined page text with `--- Printed page N ---` separators,
            or an error message if none of the requested pages have a
            footer mapping.
        """
        log.info(
            f"[lookup_worker_get_page_text:{document_id}] "
            f"printed_pages={printed_pages}"
        )
        try:
            return await get_page_text_impl(document_id, printed_pages)
        except PageNotFound as e:
            return f"[error: {e}]"

    list_pages = make_list_pages_tool(document_id)

    return [list_pages, grep, get_page_text]


def make_run_lookup_worker_tool(document_id: str) -> Callable:
    """Build a `run_lookup_worker` tool scoped to `document_id`."""

    @tool
    async def run_lookup_worker(
        sub_query: str, pages_filter: Optional[list[int]] = None
    ) -> str:
        """Dispatch ONE lookup worker for ONE focused sub-question.

        Call this multiple times IN PARALLEL (multiple tool calls in the
        same assistant turn) to fan out across independent sub-queries.
        Each worker is a fresh ReAct sub-agent with no memory of the
        parent conversation, equipped with `grep`, `get_page_text`, and
        `list_pages`.

        Args:
            sub_query: A self-contained question for this worker. Include
                the literal phrase, page hint, or quote — the worker has
                no other context.
            pages_filter: Optional list of PRINTED pages to restrict the
                worker's default grep scope. Pass when the user named a
                specific page or you want the worker to start narrow.

        Returns:
            The worker's final answer as a plain string.
        """
        log.info(
            f"[run_lookup_worker:{document_id}] sub_query={sub_query!r} "
            f"pages_filter={pages_filter}"
        )
        worker_tools = _build_worker_tools(document_id, pages_filter)
        prompt = LOOKUP_WORKER_PROMPT.format(
            document_id=document_id, grep_match_limit=GREP_MATCH_LIMIT
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

    return run_lookup_worker
