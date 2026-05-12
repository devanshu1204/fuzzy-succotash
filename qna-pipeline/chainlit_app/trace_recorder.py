"""Trace recorder for Modus QnA runs.

Consumes `astream_events(version="v2")` events from the compiled supervisor
graph and produces:

1. A hierarchical JSON trace written to `Agent-Traces/{run_id}.json` with
   per-scope token totals (supervisor, each GRA call, each sub-agent call).
2. A markdown summary string for the UI's end-of-turn token table.

Scope rules
-----------
A "scope" is opened each time the run enters an agent that runs its own LLM
calls. The supervisor is the always-on root scope. Calling an agent tool
opens a child scope; tokens spent inside that tool's body accrue only to the
child scope. Closing the tool closes the scope.

  AGENT_TOOLS = {
      "global_reasoning"   -> "gra_agent_call_{N}",
      "search"             -> "search_agent_call_{N}",
      "plan_sections"      -> "plan_sections_call_{N}",
      "run_section_worker" -> "section_worker_call_{N}",
      "query_document"     -> "document_agent_call_{N}",
  }

Leaf tools (grep, get_page_text, get_section_full_tool, ...) do not open a
scope but are recorded as `tool_calls` entries inside whatever scope is
currently innermost.

Token attribution walks the LangChain event's `parent_ids` from innermost
outwards; the first id whose tool opened a scope wins. If none matches the
LLM call is charged to "supervisor".
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


AGENT_TOOLS: dict[str, tuple[str, str]] = {
    # tool_name -> (scope_kind, scope_id_template)
    "global_reasoning":   ("gra",             "gra_agent_call_{n}"),
    "search":             ("search_agent",    "search_agent_call_{n}"),
    "plan_sections":      ("section_planner", "plan_sections_call_{n}"),
    "run_section_worker": ("section_worker",  "section_worker_call_{n}"),
    "query_document":     ("document_agent",  "document_agent_call_{n}"),
}

_PREVIEW_CHARS = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preview(value: Any, limit: int = _PREVIEW_CHARS) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, default=str, ensure_ascii=False)
        except Exception:
            value = str(value)
    if len(value) <= limit:
        return value
    return value[:limit] + f"… (truncated, {len(value) - limit} more chars)"


def _safe_dict(value: Any) -> Any:
    """Coerce tool args/output into something JSON-serializable."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _safe_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_dict(v) for v in value]
    return str(value)


def _extract_usage(output: Any) -> dict[str, Optional[int]]:
    """Pull token counts out of an AIMessage or ChatResult-like object."""
    if output is None:
        return {"input": 0, "output": 0, "total": 0}

    usage = getattr(output, "usage_metadata", None)
    if usage:
        return {
            "input":  int(usage.get("input_tokens")  or 0),
            "output": int(usage.get("output_tokens") or 0),
            "total":  int(usage.get("total_tokens")  or 0),
        }

    rm = getattr(output, "response_metadata", None) or {}
    tu = (rm.get("token_usage") if isinstance(rm, dict) else None) or {}
    if tu:
        prompt = int(tu.get("prompt_tokens") or 0)
        completion = int(tu.get("completion_tokens") or 0)
        return {
            "input":  prompt,
            "output": completion,
            "total":  int(tu.get("total_tokens") or (prompt + completion)),
        }

    return {"input": 0, "output": 0, "total": 0}


@dataclass
class Scope:
    scope_id: str
    kind: str  # "supervisor" | "gra" | "section_worker" | "document_agent" | "section_planner" | "search_agent"
    parent_scope_id: Optional[str]
    started_at: str
    ended_at: Optional[str] = None
    tokens: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0, "total": 0})
    llm_calls: int = 0
    section_name: Optional[str] = None  # filled for section_worker scopes
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scope_id": self.scope_id,
            "kind": self.kind,
            "parent_scope_id": self.parent_scope_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "tokens": self.tokens,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
        }
        if self.section_name is not None:
            d["section_name"] = self.section_name
        return d


class TraceRecorder:
    def __init__(
        self,
        run_id: str,
        question: str,
        document_id: Optional[str],
        pageindex_doc_id: Optional[str],
        traces_dir: Path,
    ):
        self.run_id = run_id
        self.question = question
        self.document_id = document_id
        self.pageindex_doc_id = pageindex_doc_id
        self.traces_dir = traces_dir
        self.started_at = _now_iso()
        self._started_perf = time.perf_counter()
        self.final_answer: Optional[str] = None

        root = Scope(
            scope_id="supervisor",
            kind="supervisor",
            parent_scope_id=None,
            started_at=self.started_at,
        )
        self.scopes: dict[str, Scope] = {"supervisor": root}
        self.scope_order: list[str] = ["supervisor"]

        # langchain run_id -> scope_id, only while that tool-scope is open
        self._open_scope_by_runid: dict[str, str] = {}
        # langchain run_id -> (scope_id, index_in_tool_calls), for tool_call entries
        self._tool_entry_by_runid: dict[str, tuple[str, int]] = {}
        # langchain run_id -> perf_counter at start, for duration
        self._tool_started_perf: dict[str, float] = {}
        # kind -> N (for "_call_{N}" naming)
        self._counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, ev: dict[str, Any]) -> None:
        kind = ev.get("event")
        try:
            if kind == "on_tool_start":
                self._on_tool_start(ev)
            elif kind == "on_tool_end":
                self._on_tool_end(ev)
            elif kind == "on_chat_model_end":
                self._on_chat_model_end(ev)
            elif kind == "on_chain_end":
                # Capture the supervisor's finalized answer.
                if ev.get("name") == "finalize":
                    out = ev.get("data", {}).get("output")
                    if isinstance(out, dict):
                        fa = out.get("final_answer")
                        if isinstance(fa, str):
                            self.final_answer = fa
        except Exception:  # never break the run for trace bookkeeping
            log.exception("trace_recorder: failed to record event %s", kind)

    def finalize(self, fallback_answer: Optional[str] = None) -> dict[str, Any]:
        """Close any still-open scopes, write the JSON file, return the trace dict."""
        now = _now_iso()
        for scope_id in list(self._open_scope_by_runid.values()):
            scope = self.scopes.get(scope_id)
            if scope and scope.ended_at is None:
                scope.ended_at = now
        self._open_scope_by_runid.clear()
        # Close the always-on root scope as well.
        if self.scopes["supervisor"].ended_at is None:
            self.scopes["supervisor"].ended_at = now

        if self.final_answer is None and fallback_answer is not None:
            self.final_answer = fallback_answer

        # Aggregate totals.
        totals_tokens = {"input": 0, "output": 0, "total": 0}
        totals_llm = 0
        tool_call_counts: dict[str, int] = {}
        scope_kind_counts: dict[str, int] = {}
        for scope_id in self.scope_order:
            scope = self.scopes[scope_id]
            for k in ("input", "output", "total"):
                totals_tokens[k] += scope.tokens.get(k, 0)
            totals_llm += scope.llm_calls
            scope_kind_counts[scope.kind] = scope_kind_counts.get(scope.kind, 0) + 1
            for tc in scope.tool_calls:
                t = tc.get("tool", "?")
                tool_call_counts[t] = tool_call_counts.get(t, 0) + 1

        duration = round(time.perf_counter() - self._started_perf, 3)

        trace: dict[str, Any] = {
            "run_id": self.run_id,
            "question": self.question,
            "document_id": self.document_id,
            "pageindex_doc_id": self.pageindex_doc_id,
            "started_at": self.started_at,
            "ended_at": now,
            "duration_seconds": duration,
            "final_answer": self.final_answer,
            "scopes": [self.scopes[sid].to_dict() for sid in self.scope_order],
            "totals": {
                "tokens": totals_tokens,
                "llm_calls": totals_llm,
                "tool_calls_by_name": tool_call_counts,
                "scope_counts": scope_kind_counts,
            },
        }

        self._write_to_disk(trace)
        return trace

    def format_summary_markdown(self, trace: dict[str, Any]) -> str:
        """Render the per-scope token table the UI shows after the answer."""
        scopes: list[dict[str, Any]] = trace["scopes"]
        totals = trace["totals"]["tokens"]
        total_llm = trace["totals"]["llm_calls"]

        # Build a child index so we can render parent rows before their children.
        children: dict[Optional[str], list[dict[str, Any]]] = {}
        for s in scopes:
            children.setdefault(s.get("parent_scope_id"), []).append(s)

        def _fmt(n: Optional[int]) -> str:
            return "—" if n is None else f"{n:,}"

        def _label(s: dict[str, Any], depth: int) -> str:
            if depth > 0:
                prefix = ("&nbsp;&nbsp;" * (depth - 1)) + "└ "
            else:
                prefix = ""
            name = s["scope_id"]
            section = s.get("section_name")
            if section:
                name = f"{name} (section: {section})"
            return f"{prefix}{name}"

        rows: list[str] = []

        def walk(parent_id: Optional[str], depth: int) -> None:
            for s in children.get(parent_id, []):
                rows.append(
                    "| {label} | {inp} | {out} | **{tot}** | {calls} |".format(
                        label=_label(s, depth),
                        inp=_fmt(s["tokens"]["input"]),
                        out=_fmt(s["tokens"]["output"]),
                        tot=_fmt(s["tokens"]["total"]),
                        calls=s["llm_calls"],
                    )
                )
                walk(s["scope_id"], depth + 1)

        walk(None, 0)

        rid_short = self.run_id[:8] if self.run_id else ""
        header = (
            f"**Token usage for this run** (`run_id: {rid_short}…`)\n\n"
            "| Scope | Input | Output | Total | LLM calls |\n"
            "|---|---:|---:|---:|---:|\n"
        )
        footer = (
            f"| **Run total** | **{_fmt(totals['input'])}** | **{_fmt(totals['output'])}** "
            f"| **{_fmt(totals['total'])}** | **{total_llm}** |\n\n"
            f"Full trace: `Agent-Traces/{self.run_id}.json`"
        )
        return header + "\n".join(rows) + "\n" + footer

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_tool_start(self, ev: dict[str, Any]) -> None:
        run_id = ev.get("run_id")
        name = ev.get("name") or "?"
        parent_ids = ev.get("parent_ids") or []
        raw_input = ev.get("data", {}).get("input")

        parent_scope_id = self._innermost_open_scope(parent_ids)
        parent_scope = self.scopes[parent_scope_id]

        entry: dict[str, Any] = {
            "tool": name,
            "args": _safe_dict(raw_input),
            "result_preview": None,
            "started_at": _now_iso(),
            "duration_seconds": None,
        }

        # If this tool is itself an agent, open a child scope and link it.
        if name in AGENT_TOOLS:
            kind, template = AGENT_TOOLS[name]
            self._counters[kind] = self._counters.get(kind, 0) + 1
            scope_id = template.format(n=self._counters[kind])

            section_name: Optional[str] = None
            if name == "run_section_worker" and isinstance(raw_input, dict):
                section_name = raw_input.get("section_name")

            scope = Scope(
                scope_id=scope_id,
                kind=kind,
                parent_scope_id=parent_scope_id,
                started_at=_now_iso(),
                section_name=section_name,
            )
            self.scopes[scope_id] = scope
            self.scope_order.append(scope_id)
            if run_id:
                self._open_scope_by_runid[run_id] = scope_id
            entry["sub_agent_scope_id"] = scope_id

        parent_scope.tool_calls.append(entry)
        if run_id:
            self._tool_entry_by_runid[run_id] = (parent_scope_id, len(parent_scope.tool_calls) - 1)
            self._tool_started_perf[run_id] = time.perf_counter()

    def _on_tool_end(self, ev: dict[str, Any]) -> None:
        run_id = ev.get("run_id")
        if not run_id:
            return

        # Update the tool_call entry with output + duration.
        loc = self._tool_entry_by_runid.pop(run_id, None)
        if loc is not None:
            scope_id, idx = loc
            entry = self.scopes[scope_id].tool_calls[idx]
            output = ev.get("data", {}).get("output")
            # output is often a ToolMessage; pull .content if present.
            content = getattr(output, "content", None)
            preview_src = content if content is not None else output
            entry["result_preview"] = _preview(preview_src)
            t0 = self._tool_started_perf.pop(run_id, None)
            if t0 is not None:
                entry["duration_seconds"] = round(time.perf_counter() - t0, 3)

        # If this tool opened a child scope, close it.
        scope_id = self._open_scope_by_runid.pop(run_id, None)
        if scope_id is not None:
            scope = self.scopes.get(scope_id)
            if scope is not None and scope.ended_at is None:
                scope.ended_at = _now_iso()

    def _on_chat_model_end(self, ev: dict[str, Any]) -> None:
        parent_ids = ev.get("parent_ids") or []
        output = ev.get("data", {}).get("output")
        usage = _extract_usage(output)
        scope_id = self._innermost_open_scope(parent_ids)
        scope = self.scopes[scope_id]
        for k in ("input", "output", "total"):
            scope.tokens[k] += int(usage.get(k) or 0)
        scope.llm_calls += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _innermost_open_scope(self, parent_ids: list[str]) -> str:
        # parent_ids is documented oldest-first; walk newest-first.
        for pid in reversed(parent_ids):
            scope_id = self._open_scope_by_runid.get(pid)
            if scope_id is not None:
                return scope_id
        return "supervisor"

    def _write_to_disk(self, trace: dict[str, Any]) -> None:
        try:
            self.traces_dir.mkdir(parents=True, exist_ok=True)
            out = self.traces_dir / f"{self.run_id}.json"
            with out.open("w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False, default=str)
            log.info("trace_recorder: wrote %s", out)
        except Exception:
            log.exception("trace_recorder: failed to write trace JSON")
