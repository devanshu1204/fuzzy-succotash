# Plan: Chainlit UI for Claude-Code-style tool-use visibility into GRA + sub-agents

## Context

When the supervisor calls the Global Reasoning Agent today, you see *nothing* between "tool call: `global_reasoning(...)`" and the final string answer. The GRA, document_agent, section_worker, and section_planner all run as `create_react_agent` instances inside opaque `@tool` wrappers (synchronous `ainvoke`, return the last message), so every internal `grep`, `plan_sections`, `run_section_worker`, `query_document`, `grep_in_section_tool`, `get_section_full_tool`, etc. call is invisible. LangGraph Studio shows the supervisor's outer steps but stops at each tool boundary.

Goal: a custom UI that renders nested tool calls live, the way Claude Code or the Anthropic console does — agent steps with collapsible children showing tool name, arguments, and output. **Plus**: after every run, persist a complete trace JSON (supervisor → GRA → sub-agents → tools, in order, with timings and per-agent token usage) under `qna-pipeline/Agent-Traces/{run_id}.json`, and show a token-load summary in the UI at the end of the answer.

## Approach

Build a **Chainlit** app at `qna-pipeline/chainlit_app/` that imports the compiled graph directly from [pipeline.py:61](qna-pipeline/qna_pipeline/pipeline.py#L61) and consumes `app.astream_events(version="v2", subgraphs=True)`.

**Why this works with zero backend changes:** LangChain's `astream_events` installs a callback handler on the outer Runnable. That handler is inherited by every nested Runnable invoked within the same async context (via `contextvars`), including the `create_react_agent` instances called inside the `@tool` bodies at:
- [global_reasoning_tool.py:91-95](qna-pipeline/qna_pipeline/tools/global_reasoning_tool.py#L91-L95)
- [document_agent_tool.py](qna-pipeline/qna_pipeline/tools/document_agent_tool.py)
- [section_worker_tool.py](qna-pipeline/qna_pipeline/tools/section_worker_tool.py)
- [section_planner_tool.py](qna-pipeline/qna_pipeline/tools/section_planner_tool.py)

So every `on_tool_start` / `on_tool_end` event from inside those nested ReAct loops bubbles up to the outer event stream. The UI walks the events and renders them as a tree using each event's `run_id` and `parent_ids`.

Chainlit's [`cl.Step`](https://docs.chainlit.io) is built for exactly this: nested, collapsible "steps" with input/output rendered as code blocks. One `Step` per LangChain run_id, `parent_id=parent_run_id`, and Chainlit handles the hierarchical display.

## File changes

### New files

**`qna-pipeline/chainlit_app/app.py`** — Chainlit entrypoint.
- `@cl.on_chat_start`: greet, ask for or accept a JSON payload (`question`, `document_id`, `pageindex_doc_id`). Mirrors the payload shape parsed by [supervisor_agent.py:44-66](qna-pipeline/qna_pipeline/nodes/supervisor_agent.py#L44-L66).
- `@cl.on_message`: parse payload from message (or use stored form values), call `await render_run(payload)`.
- Imports `app` from [pipeline.py:61](qna-pipeline/qna_pipeline/pipeline.py#L61) and invokes `app.astream_events({"supervisor_messages": [HumanMessage(content=json.dumps(payload))]}, version="v2", config={"recursion_limit": SUPERVISOR_RECURSION_LIMIT})`.
- Streams events into `event_renderer.handle_event(...)`.
- On completion, posts the final answer pulled from the supervisor's last AIMessage (or `finalize` node output) as the assistant message in the main chat.

**`qna-pipeline/chainlit_app/event_renderer.py`** — converts LangChain `astream_events` v2 dicts into Chainlit Steps.
- Maintains an in-memory `dict[run_id -> cl.Step]` for the active run.
- For each event:
  - `on_tool_start` → create a `cl.Step(name=ev["name"], type="tool", parent_id=last(ev["parent_ids"]))`, set `step.input = ev["data"]["input"]`, `await step.send()`.
  - `on_tool_end` → look up step by `run_id`, set `step.output = preview(ev["data"]["output"], 500)`, `await step.update()`.
  - `on_chain_start` where `name in {"LangGraph", "agent", "tools"}` and `parent_ids` is non-empty → create a `cl.Step(type="run")` for the sub-agent boundary so GRA / section_worker / etc. appear as parent rows over their tool calls. (For top-level graph events with empty parent_ids, skip — the supervisor's user-visible work is already the chat message.)
  - `on_chain_end` → close the matching step.
  - All `on_chat_model_*` events: skip (verbosity = tool calls + results only, per your choice).
- Tag depth by counting `len(parent_ids)` and apply a simple `language="json"` to `step.input` for tool args, `language="markdown"` to `step.output`.
- Cap previews at 500 chars; show "… (truncated, full content in LangSmith)" if longer. (LangSmith integration is optional Phase 2.)

**`qna-pipeline/chainlit_app/trace_recorder.py`** — collects every relevant event of a run into a hierarchical trace tree and writes it to disk at run end.
- Class `TraceRecorder(run_id, question, document_id, ...)` with `record(event)` and `finalize(final_answer) -> dict` methods.
- Maintains `dict[run_id -> AgentScope]` keyed by LangChain `run_id`. An "agent scope" is opened on:
  - The supervisor node: detected via `on_chain_start` where `name == "supervisor_agent"` (or fallback: the first `on_chat_model_start` whose `parent_ids` is empty, i.e., the outermost LLM call).
  - Each GRA invocation: `on_tool_start` where `name == "global_reasoning"` → opens a scope tagged `gra_agent_call_{N}` where N is a per-run counter.
  - Each sub-agent invocation called from inside GRA: `on_tool_start` where `name in {"run_section_worker", "query_document", "plan_sections"}` → opens a scope tagged `{tool}_call_{N}`.
  - Note: `grep`, `get_page_text`, `grep_in_section_tool`, `get_section_full_tool`, `get_chapter_summaries_tool` are *leaf tools without their own LLM calls*. They don't open a scope; they're recorded as `tool_calls` inside whatever scope is currently active.
- For every `on_chat_model_end` event, extract `usage_metadata` from `ev["data"]["output"]` (the AIMessage). Walk `parent_ids` from nearest → farthest, find the *innermost* open agent scope, and add the call's `input_tokens` / `output_tokens` / `total_tokens` to that scope's running totals. Also bump that scope's `llm_calls += 1`. This is what gives each GRA call and each sub-agent call its own independent token count even when called multiple times in the same run.
- For every `on_tool_start` / `on_tool_end`, append a `{tool, args, result_preview, started_at, duration_seconds}` entry into the innermost open scope's `tool_calls` list. If the tool is itself an agent, the entry's `sub_agent_scope_id` points to the scope opened for it (so the tree is reconstructable).
- `finalize(final_answer)` closes all open scopes, computes `totals` (sum across all scopes), writes `Agent-Traces/{run_id}.json` (creates the directory if missing), and returns a dict the UI uses for the token-summary message.

**`qna-pipeline/chainlit_app/__init__.py`** — empty, makes the directory a package.

**`qna-pipeline/chainlit.md`** — Chainlit's welcome page (optional). One-line description of the app.

**`qna-pipeline/Agent-Traces/.gitkeep`** — create the directory; add `Agent-Traces/*.json` to `.gitignore` so trace dumps don't pollute git.

### Existing files to touch

**`qna-pipeline/requirements.txt`** — add `chainlit>=1.3` (or current stable). No other deps needed; Chainlit ships with the necessary async server.

**`qna-pipeline/.env`** — no required changes. Optionally add `CHAINLIT_AUTH_SECRET=...` if you ever want auth, but not for Phase 1 local dev.

**`.gitignore`** (root or `qna-pipeline/`) — append `qna-pipeline/Agent-Traces/*.json` so trace dumps stay local.

**No edits to** `pipeline.py`, `supervisor_agent.py`, any tool file, `langgraph.json`, or `settings.py`. The graph and tools stay byte-identical — the UI + recorder just observe events.

## Event handling spec (what renders)

| LangChain event | UI action | Notes |
|---|---|---|
| `on_tool_start` | Open `cl.Step(type="tool", name=ev.name)`, attach `input` | Names will include: `global_reasoning`, `search`, `grep`, `plan_sections`, `run_section_worker`, `query_document`, `get_page_text`, `grep_in_section_tool`, `get_section_full_tool` |
| `on_tool_end` | Close matching step by `run_id`, attach output preview | Truncate to 500 chars |
| `on_chain_start` for named sub-agents | Open parent `cl.Step(type="run")` so tool calls render under their owning agent | Filter by `name == "LangGraph"` with non-empty `parent_ids` to catch the nested `create_react_agent` boundaries |
| `on_chain_end` matching above | Close parent step | |
| `on_chat_model_*`, `on_llm_*`, `on_chain_*` for outer graph | Skip | Outer node execution is conveyed by the assistant message + nested tool rows |

Nesting works automatically because every event carries `parent_ids` and Chainlit's `cl.Step(parent_id=...)` renders children indented under their parent. A `grep` event inside the GRA's ReAct loop will list the GRA's outer tool-call run as an ancestor in `parent_ids`, so the `grep` step appears nested under the `global_reasoning` step — same shape as Claude Code's tool-use display.

`app.py` wires the renderer and the trace recorder side-by-side: every event is dispatched to both `event_renderer.handle_event(ev)` and `trace_recorder.record(ev)` so Chainlit gets its live UI updates and the recorder accumulates the full hierarchy in parallel — no duplicate event walking.

## Trace persistence + per-agent token accounting

### Where + filename
Each run writes one JSON file: `qna-pipeline/Agent-Traces/{run_id}.json`. `run_id` is taken from the user's payload if present (already a `_PAYLOAD_KEYS` field per [supervisor_agent.py:20](qna-pipeline/qna_pipeline/nodes/supervisor_agent.py#L20)); if missing, the Chainlit `app.py` generates a `uuid4` and threads it into the payload before invoking the graph (so the recorded `run_id` always matches what's stored in `QnAState`).

### Schema (top-level keys of the JSON file)
```jsonc
{
  "run_id": "...",
  "question": "...",
  "document_id": "...",
  "pageindex_doc_id": "...",
  "started_at": "2026-05-13T10:23:11.456Z",
  "ended_at":   "2026-05-13T10:23:42.901Z",
  "duration_seconds": 31.4,
  "final_answer": "...",

  "scopes": [
    {
      "scope_id": "supervisor",
      "kind": "supervisor",
      "started_at": "...", "ended_at": "...",
      "tokens": {"input": 1200, "output": 800, "total": 2000},
      "llm_calls": 3,
      "tool_calls": [
        {"tool": "global_reasoning", "args": {...}, "result_preview": "...",
         "started_at": "...", "duration_seconds": 18.2,
         "sub_agent_scope_id": "gra_agent_call_1"}
      ]
    },
    {
      "scope_id": "gra_agent_call_1",
      "kind": "gra",
      "parent_scope_id": "supervisor",
      "started_at": "...", "ended_at": "...",
      "tokens": {"input": 5400, "output": 2100, "total": 7500},
      "llm_calls": 5,
      "tool_calls": [
        {"tool": "plan_sections", "args": {...}, "result_preview": "...",
         "sub_agent_scope_id": "plan_sections_call_1"},
        {"tool": "run_section_worker", "args": {"section_name": "Risks"},
         "sub_agent_scope_id": "run_section_worker_call_1"},
        {"tool": "run_section_worker", "args": {"section_name": "Mitigations"},
         "sub_agent_scope_id": "run_section_worker_call_2"},
        {"tool": "grep", "args": {"pattern": "credit"}, "result_preview": "..."}
        // leaf tools (no sub_agent_scope_id) — grep / get_page_text / grep_in_section_tool / get_section_full_tool / get_chapter_summaries_tool
      ]
    },
    {
      "scope_id": "plan_sections_call_1",
      "kind": "section_planner",
      "parent_scope_id": "gra_agent_call_1",
      "tokens": {"input": 1200, "output": 400, "total": 1600},
      "llm_calls": 1,
      "tool_calls": []
    },
    {
      "scope_id": "run_section_worker_call_1",
      "kind": "section_worker",
      "parent_scope_id": "gra_agent_call_1",
      "tokens": {"input": 3100, "output": 900, "total": 4000},
      "llm_calls": 3,
      "tool_calls": [
        {"tool": "grep_in_section_tool", "args": {...}, "result_preview": "..."},
        {"tool": "get_section_full_tool", "args": {...}, "result_preview": "..."}
      ]
    }
    // run_section_worker_call_2, query_document_call_1, etc.
  ],

  "totals": {
    "tokens": {"input": 14900, "output": 4500, "total": 19400},
    "llm_calls": 14,
    "tool_calls_by_name": {
      "global_reasoning": 1, "plan_sections": 1, "run_section_worker": 2,
      "grep": 3, "grep_in_section_tool": 4, "get_section_full_tool": 2
    },
    "scope_counts": {"supervisor": 1, "gra": 1, "section_planner": 1, "section_worker": 2}
  }
}
```

Each *call* (not each agent *type*) is its own scope. If GRA is invoked twice in one supervisor turn, you get `gra_agent_call_1` and `gra_agent_call_2` with separate token totals. Same for `run_section_worker_call_1..N`, `query_document_call_1..N`, `plan_sections_call_1..N`.

### Token-source semantics
Tokens come from the AIMessage's `usage_metadata` (LangChain populates this for OpenAI / LiteLLM-proxied models). The recorder reads `ev["data"]["output"].usage_metadata` on `on_chat_model_end`. If `usage_metadata` is absent (some providers under some configs), fall back to `response_metadata.token_usage`. If both missing, log a warning and record `{"input": null, "output": null, "total": null}` for that call — never crash the run.

Attribution: the LLM call is charged to the *innermost open agent scope* in `parent_ids`. This is deterministic and gives correct accounting even when:
- The supervisor calls GRA, which calls section_worker × N in parallel → each section_worker's LLM calls accrue only to that worker's scope, not to GRA or supervisor.
- GRA is re-invoked by the supervisor in the same run after digesting partial answers → `gra_agent_call_2` starts at 0 tokens.

### UI token-summary message
After the final assistant message in Chainlit, post a second message (or a non-collapsed `cl.Step` named "Token usage") with markdown like:

```
**Token usage for this run** (`run_id: 7c3a…`)
| Scope | Input | Output | Total | LLM calls |
|---|---:|---:|---:|---:|
| Supervisor | 1,200 | 800 | **2,000** | 3 |
| gra_agent_call_1 | 5,400 | 2,100 | **7,500** | 5 |
| └ plan_sections_call_1 | 1,200 | 400 | **1,600** | 1 |
| └ run_section_worker_call_1 (section: Risks) | 3,100 | 900 | **4,000** | 3 |
| └ run_section_worker_call_2 (section: Mitigations) | 2,800 | 700 | **3,500** | 3 |
| **Run total** | **13,700** | **4,900** | **18,600** | 15 |

Full trace: `Agent-Traces/7c3a….json`
```

`trace_recorder.format_summary_markdown()` produces this string from the same data it writes to disk.

## How to run

```bash
cd qna-pipeline
pip install -r requirements.txt        # picks up chainlit
chainlit run chainlit_app/app.py -w    # -w = auto-reload on file save
```

Chainlit serves on `http://localhost:8000` by default.

The existing `langgraph dev` workflow continues to work in parallel — the two are independent observers of the same graph. You can keep Studio open for graph-topology inspection and use Chainlit for nested tool-use visibility.

## Verification

1. Start the app: `chainlit run chainlit_app/app.py -w` from `qna-pipeline/`.
2. In the browser, submit a payload that exercises GRA fan-out:
   ```json
   {"question":"What are the major risks discussed across this document?","document_id":"<a real doc_id you have indexed>"}
   ```
3. Expect to see in the Chainlit Steps panel, in order:
   - "supervisor_agent" assistant message reasoning (text)
   - **Tool**: `global_reasoning` (collapsed parent)
     - **Tool**: `plan_sections` → returns task list
     - **Tool**: `run_section_worker` (×N parallel) each containing:
       - **Tool**: `grep_in_section_tool` and/or `get_section_full_tool`
     - Optional **Tool**: `query_document` or `grep` / `get_page_text`
   - Final assistant answer in the main chat thread.
   - Below the final answer: the token-usage table described in *Trace persistence + per-agent token accounting*.
4. Verify the trace file: `cat qna-pipeline/Agent-Traces/{run_id}.json | jq '.totals'` shows the same totals as the UI table; `jq '.scopes[].scope_id'` lists `supervisor`, `gra_agent_call_1`, `plan_sections_call_1`, `run_section_worker_call_1..N`, etc.
5. Negative test: omit `document_id` — confirm the early-return string from [global_reasoning_tool.py:62-68](qna-pipeline/qna_pipeline/tools/global_reasoning_tool.py#L62-L68) shows up as the tool's output, no inner tool events emit, the supervisor finalizes, and a trace JSON is *still* written (with `gra_agent_call_1` having zero `llm_calls` and zero tokens).
6. Multi-GRA-call test: craft a follow-up turn (continued conversation) that causes the supervisor to invoke `global_reasoning` twice — confirm the trace contains both `gra_agent_call_1` *and* `gra_agent_call_2` with independent token totals.
7. Volume test: ask a question that triggers >5 section workers — Chainlit should render all of them as siblings under the `run_section_worker` parent, no UI lag with 500-char previews; trace JSON contains `run_section_worker_call_1` through `run_section_worker_call_N` each with their own token total.
8. Cost sanity: sum `scopes[].tokens.total` and confirm it equals `totals.tokens.total` (within ±1 for rounding edge cases) — guards against attribution bugs.

## Phase 2 (only if Phase 1 needs polish)

- **Replace LangChain event labels with custom human-friendly payloads**: instrument the 4 tool files with `langgraph.config.get_stream_writer()` to emit events like `{"label": "Searching for 'risk' across sections...", "agent": "GRA"}`. Render those preferentially in the renderer, fall back to raw LangChain events. ~30 LOC per tool, shared helper in `qna_pipeline/tools/_stream_events.py`.
- **Enable LangSmith** by setting `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=...`, `LANGSMITH_PROJECT=modus-qna` in `.env` — gives a replayable hierarchical trace tree on smith.langchain.com, no code changes. `langsmith==0.8.3` is already in `requirements.txt`.
- **Embed token streaming** for the supervisor's final-answer turn only (not nested LLM calls) by filtering `on_chat_model_stream` events where `parent_ids[-1]` matches the supervisor node — gives a typing effect on the final answer without 100x'ing event volume.
- **Move transport to LangGraph SDK over HTTP** if you ever want to host Chainlit and the graph as separate services.

## Critical files

To create:
- [qna-pipeline/chainlit_app/app.py](qna-pipeline/chainlit_app/app.py)
- [qna-pipeline/chainlit_app/event_renderer.py](qna-pipeline/chainlit_app/event_renderer.py)
- [qna-pipeline/chainlit_app/trace_recorder.py](qna-pipeline/chainlit_app/trace_recorder.py)
- [qna-pipeline/chainlit_app/__init__.py](qna-pipeline/chainlit_app/__init__.py)
- [qna-pipeline/Agent-Traces/.gitkeep](qna-pipeline/Agent-Traces/.gitkeep)
- [qna-pipeline/chainlit.md](qna-pipeline/chainlit.md) (optional welcome page)

To edit:
- [qna-pipeline/requirements.txt](qna-pipeline/requirements.txt) — add `chainlit>=1.3`
- [.gitignore](.gitignore) — add `qna-pipeline/Agent-Traces/*.json`

To reuse without editing (referenced for context):
- [qna-pipeline/qna_pipeline/pipeline.py](qna-pipeline/qna_pipeline/pipeline.py) — `app` is imported as-is
- [qna-pipeline/qna_pipeline/nodes/supervisor_agent.py](qna-pipeline/qna_pipeline/nodes/supervisor_agent.py) — payload shape mirrors `_PAYLOAD_KEYS`
- All four sub-agent tool files — observed via callback propagation, no edits
