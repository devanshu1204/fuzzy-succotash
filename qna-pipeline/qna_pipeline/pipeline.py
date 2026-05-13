from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from config.settings import GRAPH_RECURSION_LIMIT
from qna_pipeline.nodes.finalize import finalize
from qna_pipeline.nodes.supervisor_agent import supervisor_agent
from qna_pipeline.state import QnAState
from qna_pipeline.tools.global_reasoning_tool import global_reasoning
from qna_pipeline.tools.lookup_agent_tool import lookup

# ---------------------------------------------------------------------------
# Tool list
# ---------------------------------------------------------------------------

supervisor_tools = [global_reasoning, lookup]

supervisor_tool_node = ToolNode(
    tools=supervisor_tools,
    messages_key="supervisor_messages",
)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def should_continue_supervisor(state: QnAState) -> str:
    msgs = state.get("supervisor_messages", [])
    if msgs and hasattr(msgs[-1], "tool_calls") and msgs[-1].tool_calls:
        return "supervisor_tools"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    workflow = StateGraph(QnAState)

    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("supervisor_tools", supervisor_tool_node)
    workflow.add_node("finalize", finalize)

    workflow.set_entry_point("supervisor_agent")

    workflow.add_conditional_edges(
        "supervisor_agent",
        should_continue_supervisor,
        {
            "supervisor_tools": "supervisor_tools",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("supervisor_tools", "supervisor_agent")
    workflow.add_edge("finalize", END)

    return workflow.compile()


app = build_graph().with_config({"recursion_limit": GRAPH_RECURSION_LIMIT})
