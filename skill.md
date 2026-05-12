---
name: build-langgraph-agent
description: Use when scaffolding a new LangGraph + LangChain agent in this monorepo (e.g. `projects/inv-v2-*-agent`). Reproduces the production-grade architecture used by `inv-v2-checklist-agent`: a multi-agent StateGraph with explicit ToolNodes, typed state with custom reducers, RabbitMQ + Redis driven sub-run orchestration, idempotent execution, and LiteLLM-backed LLM with retry. Apply this skill when the user asks to "create a new agent", "scaffold an agent service", "add a new agent under projects/", or asks for the conventions used by the checklist agent.
---
# Build a LangGraph Agent (Onfinance Investigation v2)

This skill encodes the architecture and production decisions that `projects/inv-v2-checklist-agent` is built on. Use it as the blueprint when scaffolding a new agent service in this monorepo. The reference implementation lives at `projects/inv-v2-checklist-agent/`; mirror its structure unless the user explicitly asks to deviate.

The goal is **not** to copy files verbatim — copy *structure, contracts, and patterns*. Rename modules and state keys to match the new agent's domain.

---

## 1. Project layout (mandatory)

Every agent service in this monorepo follows this layout. Create these directories/files even if some start empty:

```
projects/inv-v2-<name>-agent/
├── main.py                       # Thin entrypoint -> services.run_service.main
├── langgraph.json                # LangGraph CLI config: {"graphs": {"agent": "./agent/agent.py:app"}, "env": ".env"}
├── requirements.txt              # Pin versions; copy from checklist-agent and adjust
├── Dockerfile                    # python:3.11.14-slim, CMD ["python", "main.py"]
├── build-and-push.sh             # ECR build/push + kubectl rollout (mirror checklist-agent)
├── .dockerignore
├── .gitignore
├── .env                          # Never commit secrets; loaded via python-dotenv
├── README.md
├── config/
│   ├── settings.py               # ALL os.getenv() calls live here, nowhere else
│   └── prompts/                  # System prompts as Python module constants
│       └── <agent>_prompt.py
├── agent/
│   ├── agent.py                  # build_graph() + compiled `app`
│   ├── state.py                  # TypedDict state + reducer functions
│   ├── nodes/                    # One file per graph node
│   │   ├── fetch_configurations.py
│   │   ├── load_*.py
│   │   ├── <agent_name>_agent.py
│   │   └── upload_*.py
│   └── tools/                    # One file per @tool
│       └── <verb>_<noun>.py
├── services/
│   ├── run_service.py            # asyncio entrypoint, signal handlers, logging setup
│   ├── rabbitmq_consumer.py      # Dual-priority queue consumer w/ semaphore
│   ├── redis_client.py           # Workflow-run state + Lua atomic updates + interrupt flag
│   └── workflow_state.py         # Pydantic models: SubRunInfo, WorkflowRun, QueueMessage
├── utils/
│   ├── llm.py                    # LLMWithRetry wrapper around ChatOpenAI
│   ├── crud_client.py            # AsyncCrudClient/CrudClient that auto-injects user_id
│   └── update_*.py               # Side-effect helpers that PATCH the CRUD service
└── k8s/
    ├── configmap.yaml            # All non-secret config
    └── secret.yaml               # Secrets (never commit real values)
```

**Why this layout matters:** `services/` is the production wrapper, `agent/` is the graph, `config/` centralizes env access, `utils/` holds cross-cutting helpers. Keeping these separate is what lets the graph run identically under `langgraph dev` (local) and the RabbitMQ consumer (prod).

---

## 2. The graph (`agent/agent.py`)

### Pattern

A `StateGraph(<AgentState>)` with **explicit ToolNodes per agent** (not a single shared one). Each LLM-driven agent node has its own tool list and its own `messages_key`, so tool messages flow into the right channel of state.

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from agent.state import <AgentState>
from config.settings import GRAPH_RECURSION_LIMIT

# Tool lists per agent
agent_a_tools = [tool_x, tool_y, ...]
agent_b_tools = [tool_x, tool_y, tool_z, ...]

# ToolNodes — note `messages_key` routes tool outputs into per-agent message channels
agent_a_tool_node = ToolNode(tools=agent_a_tools, messages_key="agent_a_messages")
agent_b_tool_node = ToolNode(tools=agent_b_tools, messages_key="agent_b_messages")

def should_continue_a(state) -> str:
    msgs = state.get("agent_a_messages", [])
    if msgs and hasattr(msgs[-1], "tool_calls") and msgs[-1].tool_calls:
        return "agent_a_tools"
    # ... business logic to decide next node or END
    return "next_node"

def build_graph() -> StateGraph:
    workflow = StateGraph(<AgentState>)
    workflow.add_node("fetch_configurations", fetch_configurations)
    workflow.add_node("agent_a", agent_a)
    workflow.add_node("agent_a_tools", agent_a_tool_node)
    workflow.add_node("agent_b", agent_b)
    workflow.add_node("agent_b_tools", agent_b_tool_node)
    workflow.add_node("upload_findings", upload_findings)

    workflow.set_entry_point("fetch_configurations")
    workflow.add_conditional_edges("agent_a", should_continue_a, {
        "agent_a_tools": "agent_a_tools",
        "agent_b": "agent_b",
        END: END,
    })
    workflow.add_edge("agent_a_tools", "agent_a")  # loop back
    workflow.add_conditional_edges("agent_b", should_continue_b, {
        "agent_b_tools": "agent_b_tools",
        "upload_findings": "upload_findings",
    })
    workflow.add_edge("agent_b_tools", "agent_b")
    workflow.add_edge("upload_findings", END)
    return workflow.compile()

app = build_graph().with_config({"recursion_limit": GRAPH_RECURSION_LIMIT})
```

### Conventions

- **Compile to a module-level `app`** — `langgraph.json` references `./agent/agent.py:app`.
- **Apply `recursion_limit`** at compile time via `.with_config({...})`. Default 100; production uses 220.
- **Routing functions live in `agent.py`**, not in nodes. They return literal strings matching the dict keys in `add_conditional_edges`.
- **Tool nodes always loop back** to their owning agent node — never to a different agent.
- **Each agent declares its own tool list** at the top of `agent.py`. Don't inline tool lists inside nodes.

---

## 3. State (`agent/state.py`)

### Pattern

A single `TypedDict` for the whole graph (it's the *sub-run* state, not the queue-level workflow state). Use `Annotated[..., reducer]` for any field that multiple nodes/tools mutate concurrently.

### Critical rules

1. **Per-agent message channels.** Use `applicability_messages` and `compliance_messages` (or analogues). Never a single `messages` field shared across agents — that breaks per-agent context window slicing.
2. **Reducers, not raw `operator.add`, for nested dicts.** When a field like `is_applicable` has nested fields (`final_answer`, `rationale`, `agent.todo_list`, `findings`), write a custom reducer (`merge_is_applicable`, `merge_is_compliant`, `merge_todo_list`) so partial updates don't clobber sibling fields.
3. **`operator.add` is fine for plain lists** (messages, findings, file_actions) — but only when the order/duplication semantics match.
4. **Todo merge by `task_id`.** Tools update todos by passing `[updated_todo]`; the reducer merges by `task_id`, preserving fields the tool didn't set.
5. **Sliding-window reducers for ephemeral tracking.** See `append_last_5_file_calls` — accepts `[]` as an explicit "clear" signal.
6. **Context-window indexes** (`<channel>_todo_context_start: Optional[int]`) let agent nodes do O(1) message-window slicing per active todo. Set when picking up a todo, clear when marking it done.

### Required state fields (rename for the new agent's domain)

```python
class <AgentState>(TypedDict):
    # IDs / identity
    run_id: str
    sub_run_id: str
    user_id: Optional[str]
    pipeline_id: str
    # ... domain IDs (entity_id, checklist_item_id, etc.)

    # Loaded configs
    entity_data: Optional[dict[str, Any]]
    entity_config: Optional[dict[str, Any]]
    pipeline_config: Optional[dict[str, Any]]
    # ... whatever fetch_configurations produces

    # Routing hint for tools
    current_agent: Optional[Literal["agent_a", "agent_b"]]

    # Domain results — wrap with custom reducers
    is_applicable: Annotated[IsApplicable, merge_is_applicable]
    is_compliant: Annotated[IsCompliant, merge_is_compliant]

    # Resume support — populated by fetch_configurations when a prior run is found
    prev_<agent_a>_run: Optional[dict[str, Any]]
    prev_<agent_b>_run: Optional[dict[str, Any]]

    # Per-agent channels
    agent_a_messages: Annotated[list[BaseMessage], operator.add]
    agent_b_messages: Annotated[list[BaseMessage], operator.add]
    agent_a_todo_context_start: Optional[int]
    agent_b_todo_context_start: Optional[int]
    agent_a_last_5_file_calls: Annotated[Optional[list[FileCallEntry]], append_last_5_file_calls]
    agent_b_last_5_file_calls: Annotated[Optional[list[FileCallEntry]], append_last_5_file_calls]
```

### Reducer template

```python
def merge_<field>(current: Optional[dict], update: Optional[dict]) -> dict:
    if current is None: current = {}
    if update is None: update = {}
    if not isinstance(current, dict): current = {}
    if not isinstance(update, dict): update = {}
    result = dict(current)
    # Shallow-merge scalars
    for key in ("final_answer", "rationale"):
        if key in update:
            result[key] = update[key]
    # Append-merge lists
    cur_findings = current.get("findings") or []
    upd_findings = update.get("findings") or []
    if upd_findings:
        result["findings"] = cur_findings + upd_findings
    # Deep-merge nested agent dict, calling sub-reducers
    # ...
    return result
```

---

## 4. Agent nodes (`agent/nodes/<agent>_agent.py`)

### Pattern

Each LLM-driven node is a plain function `(state) -> dict`. It builds messages and returns `{"<channel>_messages": [response], "current_agent": "<name>"}`. The graph decides what to do with `tool_calls`.

### Skeleton

```python
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from utils.llm import llm
from config.prompts.<agent>_prompt import <AGENT>_PROMPT
from config.settings import <AGENT>_CONTEXT_WINDOW

# Tool list at module scope so it can be shared with agent.py if needed
<agent>_tools = [tool_x, tool_y, ...]
llm_with_tools = llm.bind_tools(<agent>_tools)

def <agent>(state) -> dict:
    # 1. Short-circuit on cached previous run
    prev = state.get("prev_<agent>_run", {})
    if prev.get("final_answer") is not None and prev.get("rationale"):
        return {"<result_field>": {"final_answer": prev["final_answer"],
                                   "rationale": prev["rationale"]}}

    # 2. Slice messages to current todo's window (or last N if none in progress)
    existing_messages = _context_messages_for_current_todo(state)

    # 3. Build prompt from state — system prompt is .format()-ed with pipeline config
    pipeline_config = state.get("pipeline_config", {}) or {}
    system_prompt = <AGENT>_PROMPT.format(
        pipeline_title=pipeline_config.get("pipeline_name"),
        pipeline_description=pipeline_config.get("pipeline_description"),
        # ... any other dynamic fields
    )
    user_prompt = _build_user_prompt(state)

    # 4. Prepend system+user, then existing messages, invoke
    messages = [SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)] + existing_messages
    response = llm_with_tools.invoke(messages)

    # 5. Persist ONLY the AIMessage; ToolNode appends its own ToolMessages later
    return {"current_agent": "<agent>",
            "<channel>_messages": [response]}

def _context_messages_for_current_todo(state) -> list:
    msgs = state.get("<channel>_messages", []) or []
    start = state.get("<channel>_todo_context_start")
    if start is not None:
        return msgs[start:]                          # active todo: full slice
    limit = int(<AGENT>_CONTEXT_WINDOW or 10)
    msgs = msgs[-limit:]
    while msgs and isinstance(msgs[0], ToolMessage):  # leading ToolMessage is invalid
        msgs = msgs[1:]
    return msgs
```

### Conventions

- **Do not append `SystemMessage`/`HumanMessage` to state** — they're rebuilt on every iteration. Only persist the assistant's `AIMessage`.
- **Strip leading `ToolMessage`s** from sliced windows — APIs reject a turn that starts with a tool result.
- **`current_agent` is the routing key for shared tools.** Tools branch on it to decide which `*_messages` channel and which `*_findings`/`*_todo_list` to write to.
- **Always check `prev_*_run` first.** Resume semantics depend on this short-circuit.

---

## 5. Tools (`agent/tools/<verb>_<noun>.py`)

### Pattern

One `@tool` per file. Tools that need to read/write state use `InjectedState` + return a `Command(update={...})`. Tools that don't (pure functions) can return a plain string.

### Skeleton

```python
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing import Annotated

@tool
def <verb>_<noun>(
    arg1: str,
    arg2: list[str],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """One-line summary the LLM sees.

    Args:
        arg1: ...
        arg2: ...
    """
    current_agent = state.get("current_agent")
    # ... do work ...
    return _build_command(state, tool_call_id, "result text", extra_updates={...})

def _build_command(state, tool_call_id, content, extra_updates=None):
    current_agent = state.get("current_agent")
    tool_msg = ToolMessage(content=content, tool_call_id=tool_call_id)
    if current_agent == "agent_a":
        update = {"agent_a_messages": [tool_msg]}
    elif current_agent == "agent_b":
        update = {"agent_b_messages": [tool_msg]}
    else:
        update = {"agent_a_messages": [tool_msg]}  # safe default
    if extra_updates:
        update.update(extra_updates)
    return Command(update=update)
```

### Conventions

- **Branch on `state["current_agent"]`** to pick the right message channel and the right result-field path. This is what lets one tool serve multiple agents.
- **`Command(update=...)` is the only correct way** for a tool to mutate non-message state (todos, findings, applicability). Returning a plain dict from a tool only updates messages — nested fields will not merge correctly without the reducer pipeline triggered by `Command`.
- **Picking up a todo sets `<channel>_todo_context_start = max(0, len(messages) - 1)`** — this includes the AI message that called the tool, which is required so the next LLM turn sees the tool call/response pair.
- **Marking a todo done clears `<channel>_todo_context_start = None`** and `<channel>_last_5_file_calls = []`. Only an explicit `[]` clears the sliding window (the reducer treats `None` as "no update").
- **External-API tools** (file search, web search, CRUD lookups) live here. They read `state["current_agent"]` to format their query (e.g. checklist runner appends a citations directive).

---

## 6. LLM wrapper (`utils/llm.py`)

### Pattern

`ChatOpenAI` against a LiteLLM proxy, wrapped in `LLMWithRetry` that does exponential backoff with jitter on 429/5xx.

### Non-negotiables

- **Always go through the proxy.** `ChatOpenAI(base_url=LITELLM_PROXY_URL, api_key=LITELLM_API_KEY, model=MODEL_NAME)`. No direct provider SDKs.
- **Wrap with `LLMWithRetry`** before exporting. Defaults: 5 retries, 1s initial delay, 60s max, jittered.
- **`bind_tools` and `with_structured_output` must return another `LLMWithRetry`** (not the raw LLM) so retry semantics survive composition.
- **`__getattr__` delegation** lets the wrapper transparently expose anything else `ChatOpenAI` provides.
- **Retryable errors:** `RateLimitError` always; `APIError` only when `status_code in [429, 500, 502, 503, 504]`. Everything else raises immediately.

Copy `utils/llm.py` from `inv-v2-checklist-agent` verbatim. Do not edit the retry logic without explicit reason.

---

## 7. Configuration (`config/settings.py`)

### Pattern

A single module that reads every env var via `os.getenv()`, with `load_dotenv()` at the top. **No other module calls `os.getenv` directly.**

```python
import os
from dotenv import load_dotenv
load_dotenv()

# Group by subsystem with comments
## CRUD Service
CRUD_SERVICE_BASE_URL = os.getenv("CRUD_SERVICE_BASE_URL")
SYSTEM_USER_ID = os.getenv("SYSTEM_USER_ID", "000000000000000000000000")

## RabbitMQ
RABBITMQ_URL = os.getenv("RABBITMQ_URL")
QUEUE_NAME = os.getenv("QUEUE_NAME")
PRIORITY_QUEUE_NAME = os.getenv("PRIORITY_QUEUE_NAME")
RABBITMQ_PREFETCH_COUNT = int(os.getenv("RABBITMQ_PREFETCH_COUNT", "1"))
PRIORITY_QUEUE_PREFETCH_COUNT = int(os.getenv("PRIORITY_QUEUE_PREFETCH_COUNT", "5"))
RABBITMQ_HEARTBEAT = int(os.getenv("RABBITMQ_HEARTBEAT", "60"))
RABBITMQ_CONNECTION_TIMEOUT = int(os.getenv("RABBITMQ_CONNECTION_TIMEOUT", "30"))
RABBITMQ_QUEUE_DURABLE = os.getenv("RABBITMQ_QUEUE_DURABLE", "true").lower() == "true"

## Redis
REDIS_URL = os.getenv("REDIS_URL")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
REDIS_SOCKET_TIMEOUT = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
REDIS_KEY_TTL = int(os.getenv("REDIS_KEY_TTL", "86400"))

## Concurrency
MAX_CONCURRENT_SUBRUNS = int(os.getenv("MAX_CONCURRENT_SUBRUNS", "5"))
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "100"))

## LiteLLM
MODEL_NAME = os.getenv("MODEL_NAME")
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
```

Mirror in `k8s/configmap.yaml` (non-secret) and `k8s/secret.yaml` (secret). Production base URLs and queue names are configmap-driven, never hardcoded.

---

## 8. Production wrapper (`services/`)

This is the part most agent tutorials skip. The graph is invoked by a RabbitMQ consumer that does idempotency, resumability, and priority interrupts. Copy these modules with minimal edits.

### `workflow_state.py` — Pydantic models

- `SubRunInfo`: per-(entity × checklist) execution unit. Status: `pending | running | completed | failed | interrupted`.
- `WorkflowRun`: container of sub-runs for a `run_id`. Has `get_pending_subruns`, `get_resumable_subruns`, `is_complete`.
- `QueueMessage`: schema validated from RabbitMQ message body. Carries `run_id`, `pipeline_id`, list of entity IDs and checklist IDs, and `run_priority`.

### `redis_client.py` — atomic state + interrupt flag

- **All sub-run status updates go through a Lua script.** The script atomically reads the workflow JSON, mutates one sub-run, recalculates rollup status, writes back. This eliminates the race where two sub-runs finish concurrently and one stomps on the other.
- **`create_run_if_not_exists` uses SETNX** for idempotent run creation.
- **TTL on every key** (`REDIS_KEY_TTL`, default 24h) so crashed workflows don't leak.
- **Priority interrupt:** a global `checklist:priority_interrupt` key with no TTL — set when the high-priority consumer starts processing, cleared in a `finally` block. Stale flags from crashes are cleared once on startup.
- **`mark_subruns_interrupted`** flips pending sub-runs to `interrupted` so the next message for the same `run_id` can resume them.

### `rabbitmq_consumer.py` — orchestration

The pattern is:

1. **Two queues, two channels, two prefetches:** normal queue (prefetch low) + priority queue (prefetch higher). Each in its own channel because prefetch is per-channel.
2. **Two parallel consumer tasks** via `asyncio.gather(_consume_high_priority, _consume_normal_priority)`.
3. **Priority consumer** sets the Redis interrupt flag before processing and clears it in `finally`. Normal consumer polls the flag and pauses (sleeps 0.5s) while it's set.
4. **`_process_message` is idempotent:**
   - If run exists and is `completed`/`failed` → ack and clean up Redis.
   - If run exists and has resumable sub-runs → flip `interrupted` → `pending`, resume.
   - If run exists and only has running sub-runs → reject+requeue (another consumer has it).
   - If new run → cartesian-product entity_ids × checklist_ids into sub-runs, `create_run_if_not_exists`.
5. **Sub-runs run in batches of `MAX_CONCURRENT_SUBRUNS`** with `asyncio.gather(..., return_exceptions=True)`. Explicit `del tasks; del results; gc.collect()` between batches — this matters when running many large LangGraph states.
6. **`asyncio.Semaphore(MAX_CONCURRENT_SUBRUNS)`** wraps each sub-run invocation as a second guard.
7. **Status is mirrored to the CRUD service** at three points: `processing` on first dequeue, `completed`/`failed` per sub-run, `completed` on workflow finish. These are best-effort — log and continue on failure.
8. **Initial state for the graph** is a dict with all required fields explicitly initialized (empty dicts/lists for the `Annotated`-reducer fields). Don't rely on LangGraph defaults.

### `run_service.py` — entrypoint

- `logging.basicConfig` with `LOG_LEVEL` env var, `aio_pika`/`aiormq` bumped to WARNING.
- Signal handlers for SIGTERM/SIGINT trigger `consumer.shutdown()` (sets event, disconnects, clears interrupt).
- `main.py` is just `asyncio.run(main())` with KeyboardInterrupt handling.

---

## 9. CRUD client (`utils/crud_client.py`)

Subclass `httpx.Client` / `httpx.AsyncClient` to auto-inject `user_id` into every query string. This is monorepo-wide convention — every CRUD-service call must carry `user_id` (defaults to `SYSTEM_USER_ID`).

```python
class AsyncCrudClient(httpx.AsyncClient):
    async def request(self, method, url, *, params=None, **kwargs):
        return await super().request(method, url, params=_inject_user_id(params), **kwargs)
```

All node/tool HTTP calls to the CRUD service go through this client (or the sync `CrudClient`). Never use raw `httpx.AsyncClient` for CRUD calls.

---

## 10. Prompts (`config/prompts/`)

- One `.py` file per agent prompt. Export a single `<AGENT>_PROMPT: str` constant.
- Use `.format(...)` placeholders: `{pipeline_title}`, `{pipeline_description}`, `{completed_todos_rationale}`, `{critical_instructions}` etc. The agent node fills these from `pipeline_config`.
- Prompts encode the **todo workflow** explicitly: read_todo_list → pickup_todo_list_item → investigate → mark_todo_as_done → repeat → final tool call (`toggle_applicability` / `write_compliance_status`). Mirror this contract for any new agent that uses todos.

---

## 11. Things that look optional but aren't

- **Per-agent message channels.** Don't use a single `messages` field.
- **Custom reducers for nested-dict state.** Don't rely on `operator.or_` / shallow merge — sibling fields will get clobbered.
- **`Command(update=...)` from stateful tools.** Returning a plain dict won't trigger reducers correctly for nested fields.
- **Lua atomic update for sub-run status.** Without it, concurrent sub-runs corrupt the workflow rollup.
- **TTLs on every Redis key** + interrupt-flag cleanup on startup. Otherwise crashed workflows leak forever.
- **`current_agent` set on every agent-node return.** Tools branch on it; without it, tools default to the wrong channel and updates land in the wrong place.
- **Strip leading `ToolMessage`s** when slicing message windows.
- **Idempotency check** on every dequeue. RabbitMQ redelivers; re-running an already-completed sub-run is silently destructive.
- **Batch + `gc.collect()`** between sub-run batches. LangGraph state is heavy; without explicit cleanup the consumer OOMs after a few hundred sub-runs.

---

## 12. Scaffolding checklist

When asked to create a new agent, follow this order:

1. **Confirm domain shape.** What are the per-sub-run identifiers (analogue of `entity_id`/`checklist_item_id`)? What is the final result (analogue of `is_compliant.findings`)? How many LLM agents are in the graph?
2. **Copy `inv-v2-checklist-agent/`** as the starting skeleton — duplicate the directory, then rename module-level identifiers.
3. **Edit `agent/state.py` first.** Define the `TypedDict`, the result types, and reducers. Resist the urge to skip reducers — they're load-bearing.
4. **Stub each node and tool** with the correct signature and `current_agent` wiring before filling in business logic.
5. **Wire `agent/agent.py`** and run `langgraph dev` against a hand-crafted initial state to verify routing.
6. **Update `services/workflow_state.py::QueueMessage`** to match the new domain's queue payload. Update `_run_single_subrun` to construct the new initial state.
7. **Mirror env vars in `config/settings.py`, `.env`, and `k8s/configmap.yaml`.** Do not let env access leak outside `settings.py`.
8. **Adapt `Dockerfile` and `build-and-push.sh`** — change `ECR_REPOSITORY`, `DEPLOYMENT_NAME`, and the k8s deployment image name only.
9. **Smoke-test:** put one message on the queue, verify Redis state lifecycle (create → running → completed → deleted), verify CRUD-service status patches landed.

---

## 13. Reference

The canonical implementation is `projects/inv-v2-checklist-agent/`. When in doubt, read that source. Any deviation from the patterns above should be justified and called out in the new agent's `README.md`.
