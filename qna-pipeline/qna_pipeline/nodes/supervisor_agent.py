import json
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts.supervisor_prompt import SUPERVISOR_PROMPT, SUPERVISOR_PROMPT_V2
from config.settings import SUPERVISOR_PROMPT_VERSION
from qna_pipeline.state import QnAState
from qna_pipeline.tools.global_reasoning_tool import global_reasoning
from qna_pipeline.tools.lookup_agent_tool import lookup
from utils.llm import llm

log = logging.getLogger(__name__)

_llm_with_tools = llm.bind_tools(
    [global_reasoning, lookup],
    parallel_tool_calls=False,
)

if SUPERVISOR_PROMPT_VERSION == "v2":
    _ACTIVE_SUPERVISOR_PROMPT = SUPERVISOR_PROMPT_V2
    _ACTIVE_SUPERVISOR_PROMPT_VERSION = "v2"
else:
    if SUPERVISOR_PROMPT_VERSION != "v1":
        log.warning(
            f"[supervisor] SUPERVISOR_PROMPT_VERSION={SUPERVISOR_PROMPT_VERSION!r} "
            f"is unknown; falling back to v1."
        )
    _ACTIVE_SUPERVISOR_PROMPT = SUPERVISOR_PROMPT
    _ACTIVE_SUPERVISOR_PROMPT_VERSION = "v1"
log.info(f"[supervisor] active prompt version: {_ACTIVE_SUPERVISOR_PROMPT_VERSION}")

_PAYLOAD_KEYS = {"question", "document_id", "pageindex_doc_id", "run_id", "user_id"}


def _extract_text(content: Any) -> Optional[str]:
    """Return the textual body of a message regardless of whether content is a
    plain string or a list of content blocks (Anthropic/Bedrock format).
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else None
    return None


def _try_parse_json_payload(msg: Any) -> Optional[dict]:
    """If `msg` is a HumanMessage whose body is a JSON object containing any
    of the known QnAState payload keys, return the decoded dict. Otherwise
    return None. Lets the chat interface ship the full invocation payload as
    a single chat message.
    """
    if not isinstance(msg, HumanMessage):
        return None
    text = _extract_text(msg.content)
    if not text:
        return None
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        data = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if not (set(data.keys()) & _PAYLOAD_KEYS):
        return None
    return data


def supervisor_agent(state: QnAState) -> dict:
    existing = state.get("supervisor_messages", [])

    document_id = state.get("document_id")
    pageindex_doc_id = state.get("pageindex_doc_id")
    question = state.get("question")
    extracted_state: dict = {}

    if (not question or not (document_id or pageindex_doc_id)) and existing:
        payload = _try_parse_json_payload(existing[0])
        if payload is not None:
            log.info(f"[supervisor] extracted state payload from chat message: keys={sorted(payload)}")
            question = question or payload.get("question")
            document_id = document_id or payload.get("document_id")
            pageindex_doc_id = pageindex_doc_id or payload.get("pageindex_doc_id")
            if question:
                extracted_state["question"] = question
            if document_id:
                extracted_state["document_id"] = document_id
            if pageindex_doc_id:
                extracted_state["pageindex_doc_id"] = pageindex_doc_id
            if payload.get("run_id"):
                extracted_state["run_id"] = payload["run_id"]
            if payload.get("user_id"):
                extracted_state["user_id"] = payload["user_id"]

    if not existing:
        system_msg = SystemMessage(
            content=_ACTIVE_SUPERVISOR_PROMPT.format(
                document_id=document_id or "<not provided>",
            )
        )
        user_msg = HumanMessage(content=question or "")
        invoke_messages = [system_msg, user_msg]
        to_append = [system_msg, user_msg]
    elif extracted_state and question:
        system_msg = SystemMessage(
            content=_ACTIVE_SUPERVISOR_PROMPT.format(
                document_id=document_id or "<not provided>",
            )
        )
        user_msg = HumanMessage(content=question)
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
        **extracted_state,
    }
