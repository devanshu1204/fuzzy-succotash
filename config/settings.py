import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Shared: LiteLLM
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME")
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

# ---------------------------------------------------------------------------
# Shared: MongoDB (typed PyMongo helpers used by both pipelines)
# ---------------------------------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "modus")
MONGODB_SEGMENTS_COLLECTION = os.getenv("MONGODB_SEGMENTS_COLLECTION", "segments")
MONGODB_SECTIONS_COLLECTION = os.getenv("MONGODB_SECTIONS_COLLECTION", "sections")
MONGODB_CHAPTERS_COLLECTION = os.getenv("MONGODB_CHAPTERS_COLLECTION", "chapters")

# ---------------------------------------------------------------------------
# Shared: Token-count encoding
# ---------------------------------------------------------------------------
TOKEN_COUNT_ENCODING = os.getenv("TOKEN_COUNT_ENCODING", "cl100k_base")

# ---------------------------------------------------------------------------
# Shared: Graph
# Note: previously the doc-processing pipeline defaulted to 100 and the qna
# pipeline to 50. We keep the higher value (100); override via .env if needed.
# ---------------------------------------------------------------------------
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "100"))

# ---------------------------------------------------------------------------
# Shared: Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Doc-processing pipeline
# ---------------------------------------------------------------------------
# Datalab
DATALAB_API_KEY = os.getenv("DATALAB_API_KEY")

# Paths
INPUT_DIR = Path(os.getenv("INPUT_DIR", str(_REPO_ROOT / "Input")))
EXTRACTION_OUTPUT_DIR = Path(
    os.getenv("EXTRACTION_OUTPUT_DIR", str(_REPO_ROOT / "output" / "extraction-output"))
)
PREPROCESSED_OUTPUT_DIR = Path(
    os.getenv(
        "PREPROCESSED_OUTPUT_DIR",
        str(_REPO_ROOT / "output" / "preprocessed-output"),
    )
)
SEGMENTATION_OUTPUT_DIR = Path(
    os.getenv(
        "SEGMENTATION_OUTPUT_DIR", str(_REPO_ROOT / "output" / "segmentation-output")
    )
)

# Segmentation
SEGMENT_CHUNK_SIZE = int(os.getenv("SEGMENT_CHUNK_SIZE", "5000"))
SEGMENT_CHUNK_OVERLAP = int(os.getenv("SEGMENT_CHUNK_OVERLAP", "500"))

# Segment analyzer
SEGMENT_ANALYZER_CONCURRENCY = int(os.getenv("SEGMENT_ANALYZER_CONCURRENCY", "10"))

# Aggregation (section + chapter reduction)
SECTION_AGGREGATOR_CONCURRENCY = int(os.getenv("SECTION_AGGREGATOR_CONCURRENCY", "1"))
CHAPTER_AGGREGATOR_CONCURRENCY = int(os.getenv("CHAPTER_AGGREGATOR_CONCURRENCY", "1"))

# ---------------------------------------------------------------------------
# QnA pipeline
# ---------------------------------------------------------------------------
# MongoDB MCP (legacy; kept for fallback)
MDB_MCP_CONNECTION_STRING = os.getenv("MDB_MCP_CONNECTION_STRING")

# Footer regex pair — per-doc configurable. Defaults match ICICI annual report
# layout: "02 | Annual Report 2023-24" (left) and
# "Annual Report 2023-24 | 03" (right), optionally bold-wrapped (**...**).
FOOTER_REGEX_LEFT = os.getenv(
    "FOOTER_REGEX_LEFT",
    r"^\*{0,2}\s*(\d+)\s*\|\s*Annual Report 2023-24\s*\*{0,2}$",
)
FOOTER_REGEX_RIGHT = os.getenv(
    "FOOTER_REGEX_RIGHT",
    r"^\*{0,2}\s*Annual Report 2023-24\s*\|\s*(\d+)\s*\*{0,2}$",
)

# GRA budgets and caps
GREP_MATCH_LIMIT = int(os.getenv("GREP_MATCH_LIMIT", "20"))
GREP_SNIPPET_CHARS = int(os.getenv("GREP_SNIPPET_CHARS", "200"))
GET_PAGE_TEXT_TOKEN_CAP = int(os.getenv("GET_PAGE_TEXT_TOKEN_CAP", "8000"))
WORKER_PARALLEL_CAP = int(os.getenv("WORKER_PARALLEL_CAP", "8"))
PLAN_MAX_TASKS = int(os.getenv("PLAN_MAX_TASKS", "10"))
GRA_INNER_RECURSION_LIMIT = int(os.getenv("GRA_INNER_RECURSION_LIMIT", "25"))
WORKER_INNER_RECURSION_LIMIT = int(os.getenv("WORKER_INNER_RECURSION_LIMIT", "15"))
DOCUMENT_AGENT_RECURSION_LIMIT = int(os.getenv("DOCUMENT_AGENT_RECURSION_LIMIT", "15"))

# Lookup agent (orchestrator + parallel workers)
LOOKUP_INNER_RECURSION_LIMIT = int(os.getenv("LOOKUP_INNER_RECURSION_LIMIT", "20"))
LOOKUP_WORKER_PARALLEL_CAP = int(os.getenv("LOOKUP_WORKER_PARALLEL_CAP", "8"))

# PageIndex MCP
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")

# Supervisor prompt version: "v1" (default, original) or "v2" (refined
# routing — explicit tiebreaker, failover, few-shot examples, DON'Ts).
SUPERVISOR_PROMPT_VERSION = os.getenv("SUPERVISOR_PROMPT_VERSION", "v1")
