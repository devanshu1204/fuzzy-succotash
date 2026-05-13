"""Section planner tool — exposed to the GRA orchestrator.

Builds a structured plan that names which sections of the document to drill
into and what each worker should extract. Pure single-LLM-call (no ReAct
loop); deterministic input → output.
"""

import json
import logging
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from config.prompts.qna.section_planner_prompt import SECTION_PLANNER_PROMPT
from config.settings import PLAN_MAX_TASKS
from qna_pipeline.db.mongo import list_sections
from qna_pipeline.schemas.plan import Plan
from utils.llm import llm

log = logging.getLogger(__name__)


def _format_sections_for_planner(sections: list[dict]) -> str:
    """Render the section list as a compact JSON-ish block the planner LLM
    can scan. We include the projected fields only.
    """
    rendered = []
    for s in sections:
        rendered.append(
            {
                "section_name": s.get("section_name", ""),
                "chapter_name": s.get("chapter_name", ""),
                "pages": s.get("pages", []),
                "summary": s.get("summary", ""),
            }
        )
    return json.dumps(rendered, ensure_ascii=False, indent=1)


def make_plan_sections_tool(document_id: str) -> Callable:
    """Build a `plan_sections` tool scoped to `document_id`.

    The closure pattern is required: this tool is built inside
    `global_reasoning_tool` and exposed to the GRA's inner ReAct loop, which
    does not propagate the outer InjectedState.
    """

    @tool
    async def plan_sections(query: str) -> str:
        """Plan which sections the worker agents should drill into for a
        section-targeted query.

        Use this when the user's question targets specific sections of the
        document (lookup, comparison across named sections, "what does
        chapter X say about Y"). The planner returns a structured `Plan`
        with 0 to {plan_max_tasks} `(section_name, sub_query)` tasks.

        If the planner returns an empty `tasks` list, the question is
        doc-wide — call `query_document` instead of `run_section_worker`.

        Args:
            query: The user's question or a refined sub-question. Be
                specific; the planner does not see the broader conversation.

        Returns:
            JSON string with shape:
                {{"tasks": [{{"section_name": str, "sub_query": str}}, ...],
                  "rationale": str}}
        """
        sections = await list_sections(document_id)
        if not sections:
            return json.dumps(
                {
                    "tasks": [],
                    "rationale": (
                        f"No sections found in MongoDB for document_id="
                        f"{document_id!r}. Check that the doc-processing-"
                        f"pipeline has been run on this document."
                    ),
                }
            )

        log.info(
            f"[plan_sections:{document_id}] planning over {len(sections)} sections"
        )

        sections_blob = _format_sections_for_planner(sections)
        system = SystemMessage(
            content=SECTION_PLANNER_PROMPT.format(plan_max_tasks=PLAN_MAX_TASKS)
        )
        user = HumanMessage(
            content=(
                f"USER QUESTION:\n{query}\n\n"
                f"AVAILABLE SECTIONS ({len(sections)} total):\n{sections_blob}\n\n"
                "Return a Plan object."
            )
        )

        structured = llm._llm.with_structured_output(Plan)
        result: Plan = await structured.ainvoke([system, user])

        if len(result.tasks) > PLAN_MAX_TASKS:
            log.warning(
                f"[plan_sections:{document_id}] planner returned "
                f"{len(result.tasks)} tasks; truncating to {PLAN_MAX_TASKS}"
            )
            result.tasks = result.tasks[:PLAN_MAX_TASKS]

        valid_section_names = {s["section_name"] for s in sections}
        kept = []
        dropped = []
        for t in result.tasks:
            if t.section_name in valid_section_names:
                kept.append(t)
            else:
                dropped.append(t.section_name)
        if dropped:
            log.warning(
                f"[plan_sections:{document_id}] dropped invented sections "
                f"from plan: {dropped}"
            )
        result.tasks = kept

        return result.model_dump_json()

    return plan_sections
