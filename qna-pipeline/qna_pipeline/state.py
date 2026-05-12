import operator
from typing import Annotated, Literal, Optional

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class QnAState(TypedDict):
    # Identity
    run_id: str
    user_id: Optional[str]

    # Input (user-provided per invocation)
    question: str
    document_id: Optional[str]          # scopes MongoDB queries in `global_reasoning`
    pageindex_doc_id: Optional[str]     # scopes PageIndex queries in `search`

    # Routing hint (mirrors sibling pipeline's `current_agent` convention)
    current_agent: Optional[Literal["supervisor"]]

    # Synthesized output written by `finalize`
    final_answer: Optional[str]

    # Supervisor message channel (per-agent channel pattern)
    supervisor_messages: Annotated[list[BaseMessage], operator.add]
