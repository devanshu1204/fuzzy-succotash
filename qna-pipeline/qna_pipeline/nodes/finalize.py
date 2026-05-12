from qna_pipeline.state import QnAState


def finalize(state: QnAState) -> dict:
    msgs = state.get("supervisor_messages", [])
    if not msgs:
        return {"final_answer": ""}
    last = msgs[-1]
    content = last.content if hasattr(last, "content") else str(last)
    return {"final_answer": content}
