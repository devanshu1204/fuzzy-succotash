import os

from dotenv import load_dotenv

load_dotenv()

# LiteLLM
MODEL_NAME = os.getenv("MODEL_NAME")
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

# MongoDB MCP
MDB_MCP_CONNECTION_STRING = os.getenv("MDB_MCP_CONNECTION_STRING")

# PageIndex MCP
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")

# Graph
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "50"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
