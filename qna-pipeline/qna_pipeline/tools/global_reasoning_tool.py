from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent

from config.prompts.global_reasoning_prompt import GLOBAL_REASONING_PROMPT
from qna_pipeline.tools.mcp_clients import get_mongodb_tools
from utils.llm import llm


@tool
async def global_reasoning(
    question: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Ask the MongoDB reasoning agent a self-contained question, scoped to
    the current document.

    Use this when the user's question can be answered by reasoning over
    structured data stored in MongoDB: records, counts, aggregations,
    lookups, filters, relationships between entities, or any structured
    fact in the database. All queries are scoped to the `document_id`
    field in pipeline state.

    Args:
        question: A clear, self-contained question for the reasoning agent.
            Include all context needed to answer; the sub-agent has no
            memory of the parent conversation.

    Returns:
        The sub-agent's final answer as a plain string. Returns an error
        message if no `document_id` is present in pipeline state.
    """
    document_id = state.get("document_id")
    if not document_id:
        return (
            "ERROR: document_id is missing from pipeline input; the global "
            "reasoning agent cannot run without a target document. Tell "
            "the user that a document_id is required to scope MongoDB "
            "queries."
        )
    mcp_tools = await get_mongodb_tools()
    sub_agent = create_react_agent(
        llm._llm,
        mcp_tools,
        prompt=GLOBAL_REASONING_PROMPT.format(document_id=document_id),
    )
    result = await sub_agent.ainvoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content
