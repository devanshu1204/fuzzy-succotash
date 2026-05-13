# Modus QnA — Chainlit observer

Watch the supervisor's calls into the Global Reasoning Agent and every nested sub-agent stream as collapsible tool steps.

**First-run setup (one time):** Chainlit needs an auth secret to sign session tokens. From `qna-pipeline/`:

```bash
chainlit create-secret >> .env       # appends CHAINLIT_AUTH_SECRET=…
chainlit run chainlit_app/app.py -w
```

**Login:** any username + any password — auth is permissive in local-dev. Threads are scoped per-username, so logging back in as the same user restores prior conversations.

**To use:**
1. Open the settings panel (cog icon) and set `document_id`.
2. Type a question. Tool calls stream live; the final answer is followed by a per-scope token table, and a full hierarchical trace is written to `Agent-Traces/{run_id}.json`.

You can also paste a JSON payload (`{"question":"…","document_id":"…"}`) as the message — fields override the settings panel.

Conversations persist in `chainlit_data/chat_history.db`; delete to start fresh.
