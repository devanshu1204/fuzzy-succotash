"""Global Reasoning Agent (GRA) — exposed as a tool to the supervisor.

The GRA is a Claude-Code-style orchestrator with FIVE tools:
- `plan_sections(query)` — plans which sections to drill into.
- `run_section_worker(section_name, sub_query)` — fan-out parallel workers.
- `query_document(query)` — doc-wide chapter-aggregate agent.
- `grep(pattern, ...)` — preprocessed-markdown search.
- `get_page_text(printed_pages)` — raw markdown for printed pages.

All five tools are CLOSURES over `document_id` (the value from outer
QnAState). They never receive InjectedState because LangGraph's inner
ReAct loop does not propagate the outer state through nested tools.

The GRA never sees raw section / segment / page text in bulk — only short
sub-agent answers and bounded grep results. That keeps the orchestrator's
running context well below 128K tokens regardless of document size.
"""

import logging
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent

from config.prompts.qna.global_reasoning_prompt import GLOBAL_REASONING_PROMPT
from config.settings import (
    GRA_INNER_RECURSION_LIMIT,
    GREP_MATCH_LIMIT,
    PLAN_MAX_TASKS,
)
from qna_pipeline.tools.document_agent_tool import make_query_document_tool
from qna_pipeline.tools.grep_tools import make_grep_tools
from qna_pipeline.tools.section_planner_tool import make_plan_sections_tool
from qna_pipeline.tools.section_worker_tool import make_run_section_worker_tool
from utils.llm import llm

log = logging.getLogger(__name__)


@tool
async def global_reasoning(
    question: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Ask the Global Reasoning Agent a self-contained question about the
    target document.

    The GRA orchestrates pre-computed structured aggregates (sections,
    chapters), parallel section workers, a document-level chapter agent,
    and grep over the preprocessed markdown — picking the right route per
    question. All work is scoped to `document_id` in pipeline state.

    Args:
        question: A clear, self-contained question. Include all context
            needed to answer; the GRA has no memory of the supervisor
            conversation.

    Returns:
        The GRA's final synthesized answer as a plain string.
    """
    document_id = state.get("document_id")
    if not document_id:
        return (
            "ERROR: document_id is missing from pipeline input; the global "
            "reasoning agent cannot run without a target document. Tell the "
            "user that a document_id is required."
        )

    log.info(f"[global_reasoning:{document_id}] question={question!r}")

    plan_sections = make_plan_sections_tool(document_id)
    run_section_worker = make_run_section_worker_tool(document_id)
    query_document = make_query_document_tool(document_id)
    grep_tool, get_page_text_tool = make_grep_tools(document_id)

    gra_tools = [
        plan_sections,
        run_section_worker,
        query_document,
        grep_tool,
        get_page_text_tool,
    ]

    prompt = GLOBAL_REASONING_PROMPT.format(
        document_id=document_id,
        plan_max_tasks=PLAN_MAX_TASKS,
        grep_match_limit=GREP_MATCH_LIMIT,
    )

    gra_agent = create_react_agent(llm._llm, gra_tools, prompt=prompt)
    result = await gra_agent.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        {"recursion_limit": GRA_INNER_RECURSION_LIMIT},
    )
    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        answer = str(answer)
    return answer
