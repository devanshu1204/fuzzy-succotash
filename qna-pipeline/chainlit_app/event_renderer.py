"""Render LangChain `astream_events(version="v2")` events as Chainlit Steps.

The renderer maps each tool invocation to one `cl.Step` and uses the event's
`parent_ids` chain to set the Step's `parent_id`, so nested tool calls inside
GRA / section workers / document agent appear as collapsible children in the
Chainlit UI — same shape as Claude Code's tool-use display.

Per the agreed verbosity ("tool calls + results only"), this renderer ignores
LLM and chain events; only `on_tool_start` / `on_tool_end` produce visible
steps.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import chainlit as cl

log = logging.getLogger(__name__)


_INPUT_PREVIEW_CHARS = 1500
_OUTPUT_PREVIEW_CHARS = 1500


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
    # ToolMessage / structured output -> pull .content if present
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


class EventRenderer:
    """Walks events and emits Chainlit Steps for tool calls."""

    def __init__(self, root_step_id: Optional[str] = None):
        # langchain run_id -> cl.Step
        self._steps: dict[str, Any] = {}
        # When the user message itself should be the visual root, pass its id
        # so top-level tool steps render directly under the assistant turn.
        self._root_step_id = root_step_id

    async def handle(self, ev: dict[str, Any]) -> None:
        try:
            kind = ev.get("event")
            if kind == "on_tool_start":
                await self._on_tool_start(ev)
            elif kind == "on_tool_end":
                await self._on_tool_end(ev)
        except Exception:
            log.exception("event_renderer: handler failure on event %s", ev.get("event"))

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
            step.language = "json"  # Chainlit highlights the input panel
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

    def _find_parent_step_id(self, parent_ids: list[str]) -> Optional[str]:
        # parent_ids is oldest-first per LangChain; walk newest-first to find
        # the nearest ancestor that we've already turned into a Step.
        for pid in reversed(parent_ids):
            step = self._steps.get(pid)
            if step is not None:
                # cl.Step ids live on the instance; some chainlit versions name it `id`.
                return getattr(step, "id", None)
        return self._root_step_id
