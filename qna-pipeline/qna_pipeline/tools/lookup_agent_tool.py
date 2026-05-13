"""Lookup Agent (orchestrator) — exposed as a tool to the supervisor.

The lookup agent is a specialist for EXACT retrieval: literal phrases,
page-named questions, quote / line verification. It mirrors the GRA's
fan-out pattern but with a much smaller tool surface:

- `list_pages` — cheap orientation at the orchestrator level.
- `run_lookup_worker(sub_query, pages_filter=None)` — dispatch parallel
  workers. Each worker is a fresh ReAct sub-agent with `grep`,
  `get_page_text`, and `list_pages`, and returns a short final answer.

The orchestrator deliberately has NO direct `grep` / `get_page_text`.
Heavy reading happens in worker context windows so the orchestrator
stays small even on multi-target adversarial questions.

Tools are CLOSURES over `document_id` (the value from outer QnAState).
They never receive InjectedState because LangGraph's inner ReAct loop
does not propagate the outer state through nested tools.
"""

import logging
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent

from config.prompts.qna.lookup_orchestrator_prompt import LOOKUP_ORCHESTRATOR_PROMPT
from config.settings import (
    LOOKUP_INNER_RECURSION_LIMIT,
    LOOKUP_WORKER_PARALLEL_CAP,
)
from qna_pipeline.tools.grep_tools import make_list_pages_tool
from qna_pipeline.tools.lookup_worker_tool import make_run_lookup_worker_tool
from utils.llm import llm

log = logging.getLogger(__name__)


@tool
async def lookup(
    question: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Ask the Lookup Agent a self-contained retrieval question about the
    target document.

    Use this for exact-phrase search, "on which page is X mentioned",
    "what does page N say", or verifying a quoted line against the
    document. The lookup agent decomposes the question into sub-queries,
    fans out parallel workers (each with grep / get_page_text /
    list_pages over the full doc), and synthesises a grounded answer.

    Args:
        question: A clear, self-contained question. Include the exact
            phrase, page number, or quote to verify; the lookup agent
            has no memory of the supervisor conversation.

    Returns:
        The lookup agent's final synthesized answer as a plain string.
    """
    document_id = state.get("document_id")
    if not document_id:
        return (
            "ERROR: document_id is missing from pipeline input; the lookup "
            "agent cannot run without a target document. Tell the user "
            "that a document_id is required."
        )

    log.info(f"[lookup:{document_id}] question={question!r}")

    list_pages_tool = make_list_pages_tool(document_id)
    run_lookup_worker = make_run_lookup_worker_tool(document_id)

    orchestrator_tools = [list_pages_tool, run_lookup_worker]

    prompt = LOOKUP_ORCHESTRATOR_PROMPT.format(
        document_id=document_id,
        lookup_worker_parallel_cap=LOOKUP_WORKER_PARALLEL_CAP,
    )

    orchestrator = create_react_agent(llm._llm, orchestrator_tools, prompt=prompt)
    result = await orchestrator.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        {"recursion_limit": LOOKUP_INNER_RECURSION_LIMIT},
    )
    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        answer = str(answer)
    return answer
