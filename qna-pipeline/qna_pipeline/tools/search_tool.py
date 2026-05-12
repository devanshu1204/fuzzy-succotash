from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent

from config.prompts.search_prompt import SEARCH_PROMPT
from qna_pipeline.tools.mcp_clients import get_pageindex_tools
from utils.llm import llm


@tool
async def search(question: str, state: Annotated[dict, InjectedState]) -> str:
    """Ask the PageIndex search agent a self-contained question about the
    indexed PDF document.

    Use this when the answer requires content from the indexed document —
    quoting passages, summarising sections, reasoning over specific pages,
    or comparing parts of the document. The target document is identified
    by the `pageindex_doc_id` field in pipeline state.

    Args:
        question: A clear, self-contained question for the search agent.
            Include all context needed to answer; the sub-agent has no
            memory of the parent conversation.

    Returns:
        The sub-agent's final answer as a plain string. Returns an error
        message if no `pageindex_doc_id` is present in pipeline state.
    """
    pageindex_doc_id = state.get("pageindex_doc_id")
    if not pageindex_doc_id:
        return (
            "ERROR: pageindex_doc_id is missing from pipeline input; the "
            "search agent cannot run without a target document. Tell the "
            "user that a pageindex_doc_id is required to query indexed "
            "documents."
        )
    mcp_tools = await get_pageindex_tools()
    sub_agent = create_react_agent(
        llm._llm,
        mcp_tools,
        prompt=SEARCH_PROMPT.format(pageindex_doc_id=pageindex_doc_id),
    )
    result = await sub_agent.ainvoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content
