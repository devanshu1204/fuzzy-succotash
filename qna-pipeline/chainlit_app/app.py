"""Chainlit UI for the Modus QnA supervisor graph.

Run with:
    chainlit run chainlit_app/app.py -w   # from inside qna-pipeline/

Each user message:
  - generates a fresh run_id (or uses one from the message if present);
  - invokes `qna_pipeline.pipeline.app` via `astream_events(version="v2")`;
  - renders every `on_tool_start` / `on_tool_end` event live as a nested
    Chainlit Step — including all nested grep / plan_sections /
    run_section_worker / get_section_full_tool / etc. calls inside GRA;
  - persists a complete hierarchical trace with per-agent token totals to
    `Agent-Traces/{run_id}.json`;
  - posts a token-usage table under the final answer.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

# Make sibling packages importable when chainlit invokes this file directly.
# - _PIPELINE_DIR (qna-pipeline/) exposes the `qna_pipeline` package.
# - _REPO_ROOT (the Modus root) exposes the shared `config` and `utils` packages.
_PIPELINE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PIPELINE_DIR.parent
for _p in (_REPO_ROOT, _PIPELINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import chainlit as cl  # noqa: E402
from chainlit.input_widget import TextInput  # noqa: E402

from qna_pipeline.pipeline import app as qna_app  # noqa: E402

from chainlit_app.event_renderer import EventRenderer  # noqa: E402
from chainlit_app.trace_recorder import TraceRecorder  # noqa: E402
# Import-side-effect: registers @cl.data_layer and applies the SQLite schema
# under qna-pipeline/chainlit_data/ so conversations persist across restarts.
from chainlit_app import data_layer  # noqa: E402, F401

log = logging.getLogger(__name__)


# Repo-root Agent-Traces/ — already exists in the workspace.
_TRACES_DIR = _PIPELINE_DIR.parent / "Agent-Traces"

# Mirrors `_PAYLOAD_KEYS` in qna_pipeline/nodes/supervisor_agent.py.
_PAYLOAD_KEYS = {"question", "document_id", "pageindex_doc_id", "run_id", "user_id"}


# ---------------------------------------------------------------------------
# Auth — permissive local-dev mode
# ---------------------------------------------------------------------------
# Chainlit's data layer requires a user identifier to scope threads in the
# sidebar; without auth, list_threads() raises and the sidebar stays empty.
# We accept any non-empty username so multiple identifiers can coexist on the
# same local DB (log in as "dev", "alice", etc.); password is ignored.

@cl.password_auth_callback
def auth_callback(username: str, _password: str) -> Optional[cl.User]:
    if not (username or "").strip():
        return None
    return cl.User(identifier=username.strip())


# ---------------------------------------------------------------------------
# Chainlit lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.ChatSettings(
        [
            TextInput(
                id="document_id",
                label="document_id (MongoDB-scoped GRA target)",
                initial="",
            ),
            TextInput(
                id="pageindex_doc_id",
                label="pageindex_doc_id (PageIndex-scoped search target)",
                initial="",
            ),
        ]
    ).send()

    await cl.Message(
        author="Modus",
        content=(
            "**Modus QnA — Chainlit observer**\n\n"
            "Open the settings panel (cog icon) to set `document_id` and / or "
            "`pageindex_doc_id`, then ask a question. You can also paste a full "
            "JSON payload as your message, e.g.\n"
            "```json\n"
            '{"question":"What are the major risks?","document_id":"…"}\n'
            "```\n"
            "Each run streams the supervisor's and every nested sub-agent's "
            "tool calls below the answer, and writes a full trace to "
            "`Agent-Traces/{run_id}.json` with per-agent token totals."
        ),
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    cl.user_session.set("document_id", (settings.get("document_id") or "").strip() or None)
    cl.user_session.set(
        "pageindex_doc_id", (settings.get("pageindex_doc_id") or "").strip() or None
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    payload = _build_payload(message.content)

    if not payload.get("question"):
        await cl.Message(
            author="Modus",
            content="Provide a question — either as plain text or in a JSON payload.",
        ).send()
        return
    if not (payload.get("document_id") or payload.get("pageindex_doc_id")):
        await cl.Message(
            author="Modus",
            content=(
                "I need at least one of `document_id` or `pageindex_doc_id` to "
                "route the query. Set it in the settings panel (cog icon) or "
                "include it in a JSON payload."
            ),
        ).send()
        return

    run_id = payload.get("run_id") or str(uuid.uuid4())
    payload["run_id"] = run_id

    recorder = TraceRecorder(
        run_id=run_id,
        question=payload["question"],
        document_id=payload.get("document_id"),
        pageindex_doc_id=payload.get("pageindex_doc_id"),
        traces_dir=_TRACES_DIR,
    )
    renderer = EventRenderer()

    input_state = {
        "run_id": run_id,
        "user_id": payload.get("user_id"),
        "question": payload["question"],
        "document_id": payload.get("document_id"),
        "pageindex_doc_id": payload.get("pageindex_doc_id"),
        "supervisor_messages": [],
    }

    final_answer: Optional[str] = None

    try:
        async for ev in qna_app.astream_events(
            input_state,
            version="v2",
            config={
                "configurable": {"thread_id": run_id},
                "metadata": {"chainlit_run_id": run_id},
            },
        ):
            recorder.record(ev)
            await renderer.handle(ev)

            # Track the supervisor's last finalized answer.
            if ev.get("event") == "on_chain_end" and ev.get("name") == "finalize":
                out = ev.get("data", {}).get("output")
                if isinstance(out, dict) and isinstance(out.get("final_answer"), str):
                    final_answer = out["final_answer"]
    except Exception as e:
        log.exception("run failed")
        await cl.Message(
            author="Modus",
            content=f"**Run failed:** `{type(e).__name__}: {e}`",
        ).send()

    trace = recorder.finalize(fallback_answer=final_answer)
    final_answer = trace.get("final_answer") or final_answer or "_(no final answer produced)_"

    # Send the final answer AFTER the stream finishes so it lands below the
    # nested tool/LLM steps that streamed in during the run, then the token
    # table below that.
    await cl.Message(author="Modus", content=final_answer).send()
    await cl.Message(
        author="Modus",
        content=recorder.format_summary_markdown(trace),
    ).send()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_payload(message_text: str) -> dict[str, Any]:
    """Resolve the active payload by combining session settings + the message
    (which may itself be a JSON payload that overrides settings).
    """
    payload: dict[str, Any] = {
        "document_id": cl.user_session.get("document_id"),
        "pageindex_doc_id": cl.user_session.get("pageindex_doc_id"),
        "question": None,
    }

    text = (message_text or "").strip()
    parsed = _try_parse_json_payload(text)
    if parsed is not None:
        for k in _PAYLOAD_KEYS:
            if parsed.get(k):
                payload[k] = parsed[k]
    elif text:
        payload["question"] = text

    return payload


def _try_parse_json_payload(text: str) -> Optional[dict[str, Any]]:
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
