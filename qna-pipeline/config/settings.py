import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# LiteLLM
MODEL_NAME = os.getenv("MODEL_NAME")
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

# MongoDB MCP (legacy; kept for fallback)
MDB_MCP_CONNECTION_STRING = os.getenv("MDB_MCP_CONNECTION_STRING")

# MongoDB (typed PyMongo helpers used by the GRA sub-agents)
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "modus")
MONGODB_SEGMENTS_COLLECTION = os.getenv("MONGODB_SEGMENTS_COLLECTION", "segments")
MONGODB_SECTIONS_COLLECTION = os.getenv("MONGODB_SECTIONS_COLLECTION", "sections")
MONGODB_CHAPTERS_COLLECTION = os.getenv("MONGODB_CHAPTERS_COLLECTION", "chapters")

# Preprocessed markdown (grep + page-text substrate)
PREPROCESSED_OUTPUT_DIR = Path(
    os.getenv(
        "PREPROCESSED_OUTPUT_DIR",
        str(Path(__file__).resolve().parents[2] / "output" / "preprocessed-output"),
    )
)

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

# Token-count encoding (matches doc-processing-pipeline for consistency)
TOKEN_COUNT_ENCODING = os.getenv("TOKEN_COUNT_ENCODING", "cl100k_base")

# PageIndex MCP
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")

# Graph
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "50"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
