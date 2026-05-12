# Multi-Agent QnA — Global Reasoning Agent Upgrade

## Context

The technical assignment requires a multi-agent QnA system over 500+ page scanned PDFs with:
- **128K-token context cap per pipeline step** (hard limit)
- **No RAG**: no vector DB, no similarity search, no embedding-based retrieval
- **Three required tiers**: Local Understanding, Aggregation, Global Reasoning
- Demo queries must span: full/section summarization, comparison across distant sections, entity/risk/decision extraction, contradiction detection, plus a needle-in-haystack ("a random line from the doc")

**Existing state.** [doc-processing-pipeline/](doc-processing-pipeline/) already produces three Mongo collections (`segments`, `sections`, `chapters`) with rich structured outputs — these satisfy Local Understanding + Aggregation tiers without any further work. [qna-pipeline/](qna-pipeline/) has a LangGraph supervisor that routes between `global_reasoning` (MongoDB MCP) and `search` (PageIndex MCP).

**The change.** Upgrade only the `global_reasoning` tool — supervisor, PageIndex search, finalize, and pipeline graph stay untouched. The new GRA is a Claude-Code-style orchestrator that holds **no raw section/segment text in its own context** — it dispatches bounded work to sub-agents and synthesizes their short answers. PageIndex is retained as a peer to the supervisor for now, but the new GRA gets its own grep + page-text tools so PageIndex could be dropped in the future without losing capability.

## Architecture

The new GRA is a ReAct agent with 5 tools. It chooses one of 3 routes per query:

**Route 1 — Section route** (lookup, comparison across specific sections)
1. GRA calls `plan_sections(query)` → spawns **Section Planner** sub-agent.
2. Planner fetches `(section_name, summary)` for all sections of `document_id` via typed Mongo helper, runs a single structured-output LLM call against a `Plan = {tasks: [(section_name, sub_query)], rationale}` schema (1–10 tasks, bias to fewer). Returns JSON.
3. GRA emits K parallel `run_section_worker(section_name, sub_query)` tool calls in one assistant turn — LangGraph's `ToolNode` parallelizes via `asyncio.gather` when `parallel_tool_calls=True`.
4. Each **Section Worker** is a fresh ReAct sub-agent with typed tools: `get_section_full` (Mongo), `get_segments_meta` (Mongo, per-segment summary + salient quotes + pages — no raw text), `grep_in_section` (markdown, page-filtered to section's pages), `get_page_text` (markdown). Tiered fetch — starts with the dense section aggregate, drills to per-segment summaries, only resorts to raw page text when needed.
5. Workers return short answers (not raw text); GRA synthesizes.

**Route 2 — Document route** (doc-wide aggregation, contradictions, full summary)
1. GRA calls `query_document(query)` → spawns **Document Agent** sub-agent.
2. Sub-agent has access *only* to the `chapters` collection (~10–20 docs for a 500-page report). Reads typed arrays directly (`contradictions[]`, `decisions[]`, `risks[]`, `key_entities[]`) and chapter summaries.
3. Returns concise answer.

**Route 3 — Grep route** (needle in haystack, exact phrase, "what's on page X")
1. GRA directly calls `grep(pattern, pages_filter?, limit=20)` and `get_page_text(pages)` against the preprocessed markdown.
2. Bounded match output: `[{page, line, snippet (±200 chars)}]`.
3. GRA synthesizes answer with page citations.

The GRA's classification of which route to use lives in its system prompt — no separate classifier agent.

## Grep substrate — preprocessed markdown file (not Mongo)

Raw text is **not** persisted in Mongo (and we deliberately don't change that). Grep and page-text retrieval read the preprocessed markdown at `output/preprocessed-output/{doc_id}.md` directly.

### Two distinct page numbering systems

- **Physical page index** — the markdown file contains explicit markers `{N}------------------------------------------------` between every page, with N being 0-indexed sequential.
- **Printed page number** — what appears in the page footer of the original document, e.g. `02 | Annual Report 2023-24` (left layout) or `Annual Report 2023-24 | 03` (right layout), occasionally bold-wrapped (`**06 | Annual Report 2023-24**`). **These are the page numbers stored in Mongo** (in `section.pages`, `chapter.pages`, `segment.pages`) and what the TOC (`{doc_id}.toc.json`) references.

The two diverge: front matter (covers, contents page) typically has no printed page number in the footer; printed page 1 may sit at physical page `{12}` or wherever the body starts. **Tools that accept page arguments accept printed page numbers** — internally they translate to physical via a mapping built at load time. This keeps the API consistent with what the sub-agents see from Mongo.

### Page-map construction

At markdown load (once per process per doc), build a `printed_to_physical: dict[int, int]` and inverse:

1. Walk the file once; tag each line with its current physical page index by latching on `^\{(\d+)\}-+$` markers.
2. Within each physical page slice, search for a footer match using two patterns (anchored at line start, allowing optional `**` wrappers and surrounding whitespace):
   - `^\*{0,2}\s*(\d+)\s*\|\s*Annual Report 2023-24\s*\*{0,2}$`  (left-side footer)
   - `^\*{0,2}\s*Annual Report 2023-24\s*\|\s*(\d+)\s*\*{0,2}$`  (right-side footer)
3. If found, record `printed → physical`. If a physical page has no footer (front matter, blank, financial pages with custom footers), it has no entry — `get_page_text(printed=X)` raises `PageNotFound` and the agent must retry with a nearby page.
4. Cache the map alongside the file in the module-level LRU cache.

The footer regex pair is doc-specific. For ICICI it's hardcoded as above. For generality, make the pair a setting (`FOOTER_REGEX_LEFT`, `FOOTER_REGEX_RIGHT`) so future docs only need a config change, not new code. TODO comment: future ingest could auto-detect the footer template per doc and persist it alongside the markdown.

### What the helpers do

- **`grep`**: scans the markdown for matches, walks each match's line back to its physical-page marker, looks up the printed page from the map, returns `{printed_page, physical_page, line, snippet}`. `pages_filter` accepts printed page numbers.
- **`get_page_text`**: takes printed page numbers, translates to physical, slices the file between the two `{N}---` markers for the requested physical range.

Zero ingest cost — the preprocessed markdown and `toc.json` already exist. Doc_id → filename mapping is direct (`ICICI Bank Report` → `output/preprocessed-output/ICICI Bank Report.md`).

## Files to modify

- **[qna-pipeline/qna_pipeline/tools/global_reasoning_tool.py](qna-pipeline/qna_pipeline/tools/global_reasoning_tool.py)** — rewrite. `global_reasoning(question, state)` builds a fresh ReAct orchestrator with 5 tools, all closures over `document_id`. Bind LLM with `parallel_tool_calls=True`. Pass `recursion_limit=25` to the inner `.ainvoke`. Drop direct MongoDB MCP wiring.
- **[qna-pipeline/config/prompts/global_reasoning_prompt.py](qna-pipeline/config/prompts/global_reasoning_prompt.py)** — replace `GLOBAL_REASONING_PROMPT` with new `GRA_ORCHESTRATOR_PROMPT` documenting the 3 routes, the 5 tools, classification heuristics, and the 128K discipline.
- **[qna-pipeline/config/settings.py](qna-pipeline/config/settings.py)** — add Mongo env vars (mirror doc-processing-pipeline conventions): `MONGODB_URI`, `MONGODB_DB`, `MONGODB_SEGMENTS_COLLECTION`, `MONGODB_SECTIONS_COLLECTION`, `MONGODB_CHAPTERS_COLLECTION`. Add `PREPROCESSED_OUTPUT_DIR` (path to `output/preprocessed-output/`). Add `GREP_MATCH_LIMIT=20`, `GREP_SNIPPET_CHARS=200`, `WORKER_PARALLEL_CAP=8`, `PLAN_MAX_TASKS=10`, `GET_PAGE_TEXT_TOKEN_CAP=8000`. Footer-regex pair: `FOOTER_REGEX_LEFT=r"^\*{0,2}\s*(\d+)\s*\|\s*Annual Report 2023-24\s*\*{0,2}$"`, `FOOTER_REGEX_RIGHT=r"^\*{0,2}\s*Annual Report 2023-24\s*\|\s*(\d+)\s*\*{0,2}$"` (doc-specific; configurable per document).
- **[qna-pipeline/.env](qna-pipeline/.env)** — add corresponding values.

## Files to add

**Typed Mongo helpers** (PyMongo, not MCP — predictable bounded output for the 128K budget):
- `qna-pipeline/qna_pipeline/db/__init__.py`
- `qna-pipeline/qna_pipeline/db/mongo.py` — module-level `MongoClient`, async wrappers via `asyncio.to_thread`. Functions:
  - `list_sections(doc_id) -> [{section_name, summary, pages}]` (projection-only)
  - `list_chapters(doc_id) -> [full AggregateAnalysis dicts]`
  - `get_section_full(doc_id, section_name) -> AggregateAnalysis dict`
  - `get_segments_meta(doc_id, section_name) -> [{seg_id, pages, summary, salient_quotes}]` (no raw text — that comes from markdown)
  - Reuse conventions from [doc-processing-pipeline/utils/mongo.py](doc-processing-pipeline/utils/mongo.py).

**Markdown-file helpers** (raw text + grep — no Mongo involvement):
- `qna-pipeline/qna_pipeline/db/markdown.py` — pure-Python file reader cached per `doc_id`. Functions:
  - `grep(doc_id, pattern, pages_filter: list[int] | None = None, regex: bool = False, limit: int = 20) -> [{printed_page, physical_page, line, snippet}]` — scans `{PREPROCESSED_OUTPUT_DIR}/{doc_id}.md` for matches; resolves printed page via the page map; `pages_filter` accepts **printed** page numbers (the same numbers Mongo uses); truncates snippet to `±GREP_SNIPPET_CHARS`; `re.escape(pattern)` when `regex=False`.
  - `get_page_text(doc_id, printed_pages: list[int]) -> str` — translates printed → physical via the map, slices the markdown between physical page markers, joins, truncates to `GET_PAGE_TEXT_TOKEN_CAP` (e.g. 8K).
  - `_load_markdown(doc_id) -> tuple[str, dict[int,int], list[tuple[int,int]]]` — module-level LRU cache. Returns `(file_text, printed_to_physical, physical_page_line_spans)`. Built once per process per doc.
  - `_build_page_map(text)` — walks the file, latches on `^\{(\d+)\}-+$`, applies the two footer regexes (`FOOTER_REGEX_LEFT`, `FOOTER_REGEX_RIGHT`) within each physical-page slice to extract the printed page number.
  - Raises `PageNotFound` when a requested printed page has no mapping (e.g. front matter without a footer).

**Sub-agent tool wrappers** (each exposed as a LangChain tool to the GRA):
- `qna-pipeline/qna_pipeline/tools/section_planner_tool.py` — `plan_sections(query, state)`. Single LLM call with `with_structured_output(Plan)`. No ReAct loop — deterministic input → output.
- `qna-pipeline/qna_pipeline/tools/section_worker_tool.py` — `run_section_worker(section_name, sub_query, state)`. Async tool that spins a fresh ReAct sub-agent with worker tools (closures over `document_id`).
- `qna-pipeline/qna_pipeline/tools/document_agent_tool.py` — `query_document(query, state)`. Spins fresh ReAct sub-agent with chapter-scoped helpers only.
- `qna-pipeline/qna_pipeline/tools/grep_tools.py` — `grep(pattern, ...)` and `get_page_text(pages)` exposed directly to GRA (no sub-agent indirection — these are bounded by construction). Closures over `document_id` and the resolved markdown path.

**Pydantic schemas**:
- `qna-pipeline/qna_pipeline/schemas/__init__.py`
- `qna-pipeline/qna_pipeline/schemas/plan.py` — `Task(section_name: str, sub_query: str)`, `Plan(tasks: list[Task], rationale: str)`.

**Prompts**:
- `qna-pipeline/config/prompts/section_planner_prompt.py`
- `qna-pipeline/config/prompts/section_worker_prompt.py`
- `qna-pipeline/config/prompts/document_agent_prompt.py`

## Files NOT to touch
[qna-pipeline/qna_pipeline/pipeline.py](qna-pipeline/qna_pipeline/pipeline.py), [state.py](qna-pipeline/qna_pipeline/state.py), [nodes/supervisor_agent.py](qna-pipeline/qna_pipeline/nodes/supervisor_agent.py), [nodes/finalize.py](qna-pipeline/qna_pipeline/nodes/finalize.py), [tools/search_tool.py](qna-pipeline/qna_pipeline/tools/search_tool.py), [config/prompts/supervisor_prompt.py](qna-pipeline/config/prompts/supervisor_prompt.py), [config/prompts/search_prompt.py](qna-pipeline/config/prompts/search_prompt.py). Supervisor still binds `[global_reasoning, search]`; PageIndex unchanged. [tools/mcp_clients.py](qna-pipeline/qna_pipeline/tools/mcp_clients.py) kept (PageIndex still uses it; Mongo MCP block becomes dormant but retained as escape hatch).

## Critical implementation rules

1. **128K discipline per agent.**
   - Planner: input ~30–40K (section summaries projection). If `len(sections) > 150`, planner first scans `chapters` to narrow to candidate chapters, then fetches sections only for those — fallback baked into planner prompt.
   - Section worker: starts with `get_section_full` (~3–8K dense aggregate). May call `get_page_text` ≤ 2 times within its section's page range. Token budget tracker required in prompt.
   - Document agent: only `chapters` collection; never touches segments/sections/markdown.
   - Grep: hard cap 20 matches × ~80 tokens.
   - `get_page_text`: hard cap on returned tokens (e.g. 8K — multiple sequential pages get truncated rather than overflow).
   - GRA orchestrator: only sees sub-agent string outputs and bounded grep results — never raw section/segment text in bulk.

2. **InjectedState propagation.** GRA's `state` (including `document_id`) reaches `global_reasoning` via LangGraph's `InjectedState`. When the GRA builds its own sub-tools internally, **close over `document_id` at tool-factory time** — do not rely on a nested injected state, which won't propagate through manually-built sub-tools.

3. **Parallel tool calls.** GRA's bound LLM uses `parallel_tool_calls=True`. Multiple `run_section_worker` calls fanned out in one assistant turn → `ToolNode` runs them via `asyncio.gather`. Note: supervisor stays at `parallel_tool_calls=False` (unchanged) — only the GRA's internal LLM binding flips this on.

4. **Dynamic K.** Planner emits 1–10 tasks. Zero tasks signals "route via `query_document` instead" — GRA respects this. After synthesis, if GRA detects gap, it may re-invoke `plan_sections` with a refined query (recursion bounded).

5. **Regex safety.** `grep(pattern, regex: bool = False)`. When false (default), apply `re.escape`. Snippet is `±GREP_SNIPPET_CHARS` around match.

6. **No new state channels.** Sub-agents are ephemeral fresh ReAct invocations; their messages live only inside the tool function. `state.py` untouched.

7. **Printed-page invariant.** All cross-component page references — Mongo `pages` arrays, TOC `page_number`, tool arguments (`grep.pages_filter`, `get_page_text.printed_pages`), final-answer citations — use **printed page numbers** (footer numbers, what the original document shows). Physical page indices `{N}` are an internal detail of the markdown reader and never surface in tool signatures or LLM-visible output. Worker prompts must say "cite the printed page number" explicitly.

## Reused functions / patterns

- `LLMWithRetry` from [qna-pipeline/utils/llm.py](qna-pipeline/utils/llm.py) for all sub-agent LLMs.
- `create_react_agent` invocation idiom from existing [global_reasoning_tool.py:44-49](qna-pipeline/qna_pipeline/tools/global_reasoning_tool.py#L44-L49).
- PyMongo + `asyncio.to_thread` idiom from [doc-processing-pipeline/utils/mongo.py](doc-processing-pipeline/utils/mongo.py).
- `AggregateAnalysis` schema from [doc-processing-pipeline/processing_pipeline/schemas/segment_analysis.py](doc-processing-pipeline/processing_pipeline/schemas/segment_analysis.py) — import for type hints in helpers (or duplicate types-only definitions if cross-package import is awkward).

## Verification

Run these 5 demo queries against the ICICI doc:

| Route | Query | Expected behavior |
|---|---|---|
| Section (R1) | "Compare the credit risk approach in the Risk Management chapter vs. the disclosures in the Auditor's Report." | Planner returns 2 tasks; 2 parallel `run_section_worker` ToolMessages visible in trace with simultaneous timestamps. |
| Document (R2) | "Give an executive summary of the entire annual report." | Single `query_document` call; chapters-only fetch; sub-agent total tokens < 30K. |
| Document (R2) | "List every operational risk and which entity it affects." | `query_document` scans chapter `risks[]` arrays; zero segment fetches. |
| Document (R2) | "Are there contradictions between MD&A and audited financials?" | `query_document` reads chapter `contradictions[]`. |
| Grep (R3) | "On which page does the document mention 'ESG bond' and what's the surrounding context?" | Direct `grep` returns `{printed_page, physical_page, line, snippet}`; GRA calls `get_page_text([printed_page])` for context; final answer cites the **printed** page number (the one the user can flip to in the original PDF). |

**Token-budget assertions.** Instrument every helper to log `tiktoken.encoding_for_model("cl100k_base").encode(result)` length. Assert: no tool return > 30K tokens; GRA's running context never exceeds 60K (well below 128K).

**Regression check.** Re-run any existing supervisor-routed test queries — supervisor + PageIndex search + finalize must be byte-identical (no code changed there).

**Manual smoke.** Bring up MongoDB locally (already containerized by doc-processing-pipeline) with the existing ICICI ingest results. Confirm `output/preprocessed-output/ICICI Bank Report.md` is present. Set `PREPROCESSED_OUTPUT_DIR` in `.env`. Run qna-pipeline against each query above and inspect LangSmith trace for parallel tool spans and per-tool token counts.

## Open implementation choices

1. **MongoDB MCP retention.** Keep the dormant Mongo MCP block in [mcp_clients.py](qna-pipeline/qna_pipeline/tools/mcp_clients.py) for an optional legacy escape hatch (zero cost) — recommend keep.
2. **Worker concurrency cap.** `WORKER_PARALLEL_CAP=8` is comfortable for K≤10 plans. Bump only if traces show planner consistently saturating it.
3. **Inner recursion limit.** Pass `{"recursion_limit": 25}` to the GRA's internal `gra_agent.ainvoke`. Separate from the outer `GRAPH_RECURSION_LIMIT=50` on the supervisor graph.
