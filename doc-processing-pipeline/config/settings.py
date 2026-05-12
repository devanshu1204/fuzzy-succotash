import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# LiteLLM
MODEL_NAME = os.getenv("MODEL_NAME")
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

# Datalab
DATALAB_API_KEY = os.getenv("DATALAB_API_KEY")

# Paths
INPUT_DIR = Path(os.getenv("INPUT_DIR", str(Path(__file__).resolve().parents[2] / "Input")))
EXTRACTION_OUTPUT_DIR = Path(os.getenv("EXTRACTION_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "output" / "extraction-output")))
PREPROCESSED_OUTPUT_DIR = Path(os.getenv("PREPROCESSED_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "output" / "preprocessed-output")))
SEGMENTATION_OUTPUT_DIR = Path(os.getenv("SEGMENTATION_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "output" / "segmentation-output")))

# Segmentation
SEGMENT_CHUNK_SIZE = int(os.getenv("SEGMENT_CHUNK_SIZE", "5000"))
SEGMENT_CHUNK_OVERLAP = int(os.getenv("SEGMENT_CHUNK_OVERLAP", "500"))
TOKEN_COUNT_ENCODING = os.getenv("TOKEN_COUNT_ENCODING", "cl100k_base")

# Segment analyzer
SEGMENT_ANALYZER_CONCURRENCY = int(os.getenv("SEGMENT_ANALYZER_CONCURRENCY", "10"))

# Aggregation (section + chapter reduction)
SECTION_AGGREGATOR_CONCURRENCY = int(os.getenv("SECTION_AGGREGATOR_CONCURRENCY", "1"))
CHAPTER_AGGREGATOR_CONCURRENCY = int(os.getenv("CHAPTER_AGGREGATOR_CONCURRENCY", "1"))

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "modus")
MONGODB_SEGMENTS_COLLECTION = os.getenv("MONGODB_SEGMENTS_COLLECTION", "segments")
MONGODB_SECTIONS_COLLECTION = os.getenv("MONGODB_SECTIONS_COLLECTION", "sections")
MONGODB_CHAPTERS_COLLECTION = os.getenv("MONGODB_CHAPTERS_COLLECTION", "chapters")

# Graph
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "100"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
