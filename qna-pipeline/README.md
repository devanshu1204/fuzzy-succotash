# qna-pipeline

A LangGraph supervisor pipeline that answers questions by routing through two MCP-backed sub-agents:

1. **Supervisor agent** — orchestrates the two sub-agents (used as tools), then synthesises a final answer.
2. **Global reasoning agent** — connected to MongoDB Official MCP (read-only); reasons over structured data, scoped to a single document via `document_id` in input state.
3. **Search agent** — connected to PageIndex by Vectify AI MCP; searches and reasons over an indexed PDF, scoped via `pageindex_doc_id` in input state.

## Local dev

```bash
cp .env <fill in values>           # or edit .env directly
pip install -r ../requirements.txt
langgraph dev
```

`npx` (Node 18+) must be on PATH — both MCP servers are Node packages launched via `npx -y …`.

## Invocation shape

```json
{
  "run_id": "test-1",
  "question": "<the user's question>",
  "document_id": "<MongoDB document_id, or null if no MongoDB reasoning needed>",
  "pageindex_doc_id": "<PageIndex doc_id, or null if no document search needed>"
}
```

## Known limitations

- First invocation pays a 5–15s `npx` cold start for each MCP server.
- `langgraph dev` module reloads can orphan `npx` child processes. Kill manually if needed: `pkill -f mongodb-mcp-server; pkill -f @pageindex/mcp`.
- Changes to `.env` require a dev-server restart.
- Sub-agent LLM calls bypass the retry wrapper used by the supervisor.
