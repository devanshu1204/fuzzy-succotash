# Modus QnA — Chainlit observer

Watch the supervisor's calls into the Global Reasoning Agent and every nested sub-agent (section planner, section workers, document agent) stream as collapsible tool steps — like Claude Code's tool-use display.

**First-run setup (one time):** Chainlit needs an auth secret to sign session tokens. From `qna-pipeline/`, run:

```bash
chainlit create-secret >> .env       # appends CHAINLIT_AUTH_SECRET=…
```

then start the app: `chainlit run chainlit_app/app.py -w`.

**Login:** type any username (e.g. `dev`) and any password — authentication is permissive in local-dev mode. Threads are scoped per-username, so logging back in as the same user brings prior conversations back into the sidebar.

**To use:**
1. Open the settings panel (cog icon) and set `document_id` (and/or `pageindex_doc_id`).
2. Type a question.
3. Watch tool calls stream live. After the answer, a per-scope token table prints; a complete hierarchical trace is written to `Agent-Traces/{run_id}.json`.

You can also paste a full JSON payload (`{"question":"…","document_id":"…"}`) as the message itself — fields there override the settings panel.

Conversations persist across restarts in `chainlit_data/chat_history.db`. Delete that file to start fresh.
