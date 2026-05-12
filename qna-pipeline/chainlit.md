# Modus QnA — Chainlit observer

Watch the supervisor's calls into the Global Reasoning Agent and every nested sub-agent (section planner, section workers, document agent) stream as collapsible tool steps — like Claude Code's tool-use display.

**To use:**
1. Open the settings panel (cog icon) and set `document_id` (and/or `pageindex_doc_id`).
2. Type a question.
3. Watch tool calls stream live. After the answer, a per-scope token table prints; a complete hierarchical trace is written to `Agent-Traces/{run_id}.json`.

You can also paste a full JSON payload (`{"question":"…","document_id":"…"}`) as the message itself — fields there override the settings panel.
