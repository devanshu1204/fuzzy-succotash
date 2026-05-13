"""Render LangChain `astream_events(version="v2")` events as Chainlit Steps.

Two streams of events surface in the UI:

1. **Tool calls** — `on_tool_start` / `on_tool_end` → one `cl.Step(type="tool")`
   per invocation, nested under whichever ancestor step opened first (using the
   event's `parent_ids` chain). This is the Claude-Code outline.

2. **LLM reasoning** — `on_chat_model_*` → one `cl.Step(type="llm")` per LLM
   call (supervisor + every nested agent), with tokens streamed live via
   `step.stream_token`. The user sees each agent's reasoning "typing in" the
   moment it generates, sibling to that agent's tool steps.

The final assistant answer is posted as a separate `cl.Message` after the
stream finishes — this renderer doesn't write directly into any message
bubble; it only emits Steps.

Verbosity = tool calls + LLM token stream; chain events, embeddings, and
retrievers are ignored.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import chainlit as cl

log = logging.getLogger(__name__)


_INPUT_PREVIEW_CHARS = 1500
_OUTPUT_PREVIEW_CHARS = 1500


# Tool name -> short display name for the agent that owns nested LLM calls.
_AGENT_DISPLAY_NAMES: dict[str, str] = {
    "global_reasoning":   "GRA",
    "search":             "search_agent",
    "run_section_worker": "section_worker",
    "query_document":     "document_agent",
    "plan_sections":      "section_planner",
}


def _fmt_input(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value if len(value) <= _INPUT_PREVIEW_CHARS else (
            value[:_INPUT_PREVIEW_CHARS] + "\n… (truncated)"
        )
    try:
        s = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        s = str(value)
    if len(s) > _INPUT_PREVIEW_CHARS:
        s = s[:_INPUT_PREVIEW_CHARS] + "\n… (truncated)"
    return s


def _fmt_output(value: Any) -> str:
    content = getattr(value, "content", None)
    if content is not None:
        value = content
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except Exception:
            value = str(value)
    if len(value) > _OUTPUT_PREVIEW_CHARS:
        value = value[:_OUTPUT_PREVIEW_CHARS] + "\n… (truncated)"
    return value


def _chunk_to_text(chunk: Any) -> str:
    """Extract the text portion of an AIMessageChunk (or similar)."""
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Anthropic-style block content: [{"type": "text", "text": "..."}, ...]
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


class EventRenderer:
    """Walks events and emits Chainlit Steps for tool calls + LLM reasoning."""

    def __init__(self, root_step_id: Optional[str] = None):
        # langchain run_id -> cl.Step (covers both tool and LLM events)
        self._steps: dict[str, Any] = {}
        # Optional id of a parent message/step that top-level activity should
        # render under. None → activity is top-level in the chat transcript.
        self._root_step_id = root_step_id

    async def handle(self, ev: dict[str, Any]) -> None:
        try:
            kind = ev.get("event")
            if kind == "on_tool_start":
                await self._on_tool_start(ev)
            elif kind == "on_tool_end":
                await self._on_tool_end(ev)
            elif kind == "on_chat_model_start":
                await self._on_chat_model_start(ev)
            elif kind == "on_chat_model_stream":
                await self._on_chat_model_stream(ev)
            elif kind == "on_chat_model_end":
                await self._on_chat_model_end(ev)
        except Exception:
            log.exception("event_renderer: handler failure on event %s", ev.get("event"))

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------

    async def _on_tool_start(self, ev: dict[str, Any]) -> None:
        run_id = ev.get("run_id")
        if not run_id:
            return
        name = ev.get("name") or "tool"
        parent_ids = ev.get("parent_ids") or []
        raw_input = ev.get("data", {}).get("input")

        parent_step_id = self._find_parent_step_id(parent_ids)

        step = cl.Step(
            name=name,
            type="tool",
            parent_id=parent_step_id,
        )
        step.input = _fmt_input(raw_input)
        try:
            step.language = "json"
        except Exception:
            pass
        await step.send()
        self._steps[run_id] = step

    async def _on_tool_end(self, ev: dict[str, Any]) -> None:
        run_id = ev.get("run_id")
        if not run_id:
            return
        step = self._steps.get(run_id)
        if step is None:
            return
        output = ev.get("data", {}).get("output")
        step.output = _fmt_output(output)
        try:
            await step.update()
        except Exception:
            log.exception("event_renderer: step.update() failed for %s", step.name)

    # ------------------------------------------------------------------
    # LLM events
    # ------------------------------------------------------------------

    async def _on_chat_model_start(self, ev: dict[str, Any]) -> None:
        run_id = ev.get("run_id")
        if not run_id:
            return
        parent_ids = ev.get("parent_ids") or []
        parent_step_id = self._find_parent_step_id(parent_ids)
        owner = self._describe_owner(parent_ids)

        step = cl.Step(
            name=f"{owner} · thinking",
            type="llm",
            parent_id=parent_step_id,
        )
        try:
            step.language = "markdown"
        except Exception:
            pass
        await step.send()
        self._steps[run_id] = step

    async def _on_chat_model_stream(self, ev: dict[str, Any]) -> None:
        run_id = ev.get("run_id")
        if not run_id:
            return
        chunk = ev.get("data", {}).get("chunk")
        token = _chunk_to_text(chunk)
        if not token:
            return
        step = self._steps.get(run_id)
        if step is None:
            return
        try:
            await step.stream_token(token)
        except Exception:
            log.exception("event_renderer: step.stream_token failed")

    async def _on_chat_model_end(self, ev: dict[str, Any]) -> None:
        run_id = ev.get("run_id")
        if not run_id:
            return
        step = self._steps.get(run_id)
        if step is None:
            return

        output = ev.get("data", {}).get("output")
        tool_calls = getattr(output, "tool_calls", None) or []
        had_text = bool((step.output or "").strip())

        if tool_calls and not had_text:
            try:
                await step.stream_token(
                    f"_(→ dispatching {len(tool_calls)} tool call(s))_"
                )
            except Exception:
                pass
        elif tool_calls:
            try:
                await step.stream_token(
                    f"\n\n_(→ dispatching {len(tool_calls)} tool call(s))_"
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_parent_step_id(self, parent_ids: list[str]) -> Optional[str]:
        for pid in reversed(parent_ids):
            step = self._steps.get(pid)
            if step is not None:
                return getattr(step, "id", None)
        return self._root_step_id

    def _describe_owner(self, parent_ids: list[str]) -> str:
        for pid in reversed(parent_ids):
            step = self._steps.get(pid)
            if step is None:
                continue
            tool_name = step.name or ""
            if tool_name in _AGENT_DISPLAY_NAMES:
                return _AGENT_DISPLAY_NAMES[tool_name]
            return tool_name
        return "supervisor"
