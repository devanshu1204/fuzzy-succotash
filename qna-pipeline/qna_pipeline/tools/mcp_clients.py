import asyncio
from typing import Optional

from langchain_mcp_adapters.client import MultiServerMCPClient

from config.settings import MDB_MCP_CONNECTION_STRING, PAGEINDEX_API_KEY

_client: Optional[MultiServerMCPClient] = None
_mongo_tools: Optional[list] = None
_pageindex_tools: Optional[list] = None
_lock = asyncio.Lock()


def _build_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "mongodb": {
                "command": "npx",
                "args": ["-y", "mongodb-mcp-server@latest", "--readOnly"],
                "env": {"MDB_MCP_CONNECTION_STRING": MDB_MCP_CONNECTION_STRING or ""},
                "transport": "stdio",
            },
            "pageindex": {
                "command": "npx",
                "args": ["-y", "@pageindex/mcp"],
                "env": {"PAGEINDEX_API_KEY": PAGEINDEX_API_KEY or ""},
                "transport": "stdio",
            },
        }
    )


async def get_mongodb_tools() -> list:
    global _client, _mongo_tools
    async with _lock:
        if _mongo_tools is None:
            if _client is None:
                _client = _build_client()
            _mongo_tools = await _client.get_tools(server_name="mongodb")
    return _mongo_tools


async def get_pageindex_tools() -> list:
    global _client, _pageindex_tools
    async with _lock:
        if _pageindex_tools is None:
            if _client is None:
                _client = _build_client()
            _pageindex_tools = await _client.get_tools(server_name="pageindex")
    return _pageindex_tools
