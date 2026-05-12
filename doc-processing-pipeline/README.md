# doc-processing-pipeline

A LangGraph document processing pipeline with 5 nodes:

1. **Extraction** — extracts raw content from input documents
2. **Preprocessing** — cleans and normalises extracted content
3. **Segmentation** — splits content into logical segments
4. **Segment Analyzer Agent** — LLM agent that analyses each segment
5. **Aggregation Agent** — LLM agent that aggregates segment analyses into a final result

## Local dev

```bash
cp .env.example .env  # fill in values
pip install -r requirements.txt
langgraph dev
```
