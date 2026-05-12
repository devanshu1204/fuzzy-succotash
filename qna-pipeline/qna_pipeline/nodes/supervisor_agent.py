from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts.supervisor_prompt import SUPERVISOR_PROMPT
from qna_pipeline.state import QnAState
from qna_pipeline.tools.global_reasoning_tool import global_reasoning
from qna_pipeline.tools.search_tool import search
from utils.llm import llm

_llm_with_tools = llm.bind_tools(
    [global_reasoning, search],
    parallel_tool_calls=False,
)


def supervisor_agent(state: QnAState) -> dict:
    existing = state.get("supervisor_messages", [])

    if not existing:
        system_msg = SystemMessage(
            content=SUPERVISOR_PROMPT.format(
                document_id=state.get("document_id") or "<not provided>",
                pageindex_doc_id=state.get("pageindex_doc_id") or "<not provided>",
            )
        )
        user_msg = HumanMessage(content=state["question"])
        invoke_messages = [system_msg, user_msg]
        to_append = [system_msg, user_msg]
    else:
        invoke_messages = existing
        to_append = []

    response = _llm_with_tools.invoke(invoke_messages)
    to_append.append(response)

    return {
        "current_agent": "supervisor",
        "supervisor_messages": to_append,
    }
