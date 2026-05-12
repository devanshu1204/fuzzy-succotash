Yes — there's a whole research lineage purpose-built for this, and it's worth showing awareness of it in your writeup even if you don't implement all of it.

## The landscape you should know

**GraphRAG (Microsoft, 2024)** — exactly the thing you asked about. Extracts entities + relations into a knowledge graph, runs community detection (Leiden algorithm), generates hierarchical summaries at each community level. Query-time you traverse community summaries (global) or specific subgraphs (local). The off-the-shelf version uses embeddings for some entity resolution, but the *core mechanism* — entity graph + hierarchical community summaries — doesn't require them. You can do entity resolution via LLM-based string matching and coreference. **This is probably the strongest fit for your contradiction-detection requirement** because conflicting facts about the same entity become structurally trivial to find when they're edges on a node.

**RAPTOR (Stanford, ICLR 2024)** — recursively clusters chunks and summarizes them into a tree. Standard version clusters with embeddings, but you can substitute structural clustering (which PageIndex already gives you for free). Same shape as your current map-reduce but with explicit summary nodes at every level.

**ReadAgent (Google DeepMind, 2024)** — models how humans read. Pages through the document, creates "gist memories" for each episode, then at query time the agent decides which full-resolution pages to pull back based on the gists. Pure LLM reasoning, no embeddings. Very clean fit for your constraints.

**MemWalker (Meta, 2024)** — tree of summaries with an agent that walks down the tree at query time, making routing decisions at each node. Philosophically very close to what PageIndex enables.

**Chain-of-Agents (Google, NeurIPS 2024)** — multiple worker agents process chunks sequentially, each passing a "communication unit" forward; a manager agent synthesizes. Explicitly multi-agent, explicitly for long context. Directly matches the brief's wording.

**GraphReader (ACL 2024)** — builds a graph from text, then an agent explores it step-by-step to answer queries. No embeddings required.

## What I'd actually build for this assignment

A **hybrid**: PageIndex for structural navigation + LLM-extracted knowledge graph for cross-section reasoning + multi-agent orchestration to tie them together.

Why this combination specifically:

- **PageIndex** handles the "navigate by topic/section" axis — great for summarization and section-scoped queries.
- **Knowledge graph** handles the "track entities and claims across the document" axis — which is what kills naive map-reduce on contradiction detection, entity extraction, and "compare distant sections" queries. Your current "DB of entities/risks/decisions" is already 60% of the way to this; you just need explicit relations between them, not flat records.
- **Multi-agent layer** routes queries to the right substrate.

Concrete shape:

- Ingestion agent → OCR + per-region confidence scoring
- Structure agent → PageIndex tree
- Section Analyzer agents (parallel) → produce `(entity, relation, entity, page, confidence)` triples + section summary
- Aggregator agent → roll summaries up the tree, RAPTOR-style
- Graph store → property graph in SQLite or NetworkX (don't reach for Neo4j, it'll read as over-engineering for a take-home)
- Query Router agent → classifies query type, dispatches to tree traversal, graph traversal, or both
- Verifier/Contradiction agent → for contradiction queries, scans the graph for same-entity nodes with conflicting attributes or relations

The single biggest seniority signal you can add to the README is a paragraph that says: *"I considered GraphRAG, RAPTOR, and ReadAgent. GraphRAG's community-summary structure is ideal for contradiction queries but its embedding-based entity resolution conflicts with the constraint; I substituted LLM-based coreference. RAPTOR's clustering layer is replaced by PageIndex's deterministic structure. ReadAgent's gist memory pattern informed my section summaries."* — that one paragraph reframes you from "implementer" to "architect."

One caveat: don't build all of this. Pick the hybrid above, build it cleanly, and *cite* the others in the trade-off section. A working focused system beats a sprawling half-built one every time.