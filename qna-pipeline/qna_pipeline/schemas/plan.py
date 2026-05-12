from pydantic import BaseModel, Field


class Task(BaseModel):
    section_name: str = Field(
        description=(
            "Exact section_name of a section in the document. Must match a "
            "section_name returned by list_sections — do not invent or alter."
        )
    )
    sub_query: str = Field(
        description=(
            "Self-contained question for the worker assigned to this section. "
            "State what to extract from THIS section specifically; the worker "
            "has no memory of the original user question."
        )
    )


class Plan(BaseModel):
    """Planner output for the section route.

    `tasks` is the list of (section, sub_query) pairs to dispatch in parallel.
    Empty list signals "route via query_document instead" — the query is doc-
    wide and should not pass through section selection.
    """

    tasks: list[Task] = Field(
        description=(
            "0 to PLAN_MAX_TASKS pairs. Return [] when the query is doc-wide "
            "(spans all sections / asks about contradictions across the doc / "
            "wants a global summary) — the orchestrator will route to the "
            "document agent instead. Otherwise return the minimum number of "
            "sections needed to answer; bias toward fewer."
        )
    )
    rationale: str = Field(
        description=(
            "One short sentence explaining why these sections (or why an "
            "empty list). Surfaces planner reasoning for debugging."
        )
    )
