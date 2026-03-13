"""
MCP Client Wrapper — Connects to Assistant Hub via Model Context Protocol.

Wraps `langchain-mcp-adapters` to auto-discover and load tools from the
MCP server with authentication, caching, and optional x402 payment support.

Usage:
    # Option A: Direct MCP adapter (auto-discovers all tools)
    from langchain_assistanthub.client import AssistantHubMCPClient
    client = AssistantHubMCPClient(api_key="your-jwt")
    tools = await client.get_tools()

    # Option B: Use the toolkit (recommended — includes typed tools)
    from langchain_assistanthub import AssistantHubToolkit
    toolkit = AssistantHubToolkit(api_key="your-jwt")
    tools = toolkit.get_tools()
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool


class AssistantHubMCPClient:
    """
    MCP client that connects to the Assistant Hub MCP server.

    Auto-discovers all available tools via the MCP protocol and returns
    them as LangChain BaseTool instances. Handles authentication via
    JWT Bearer token or API key header.

    Args:
        url: MCP server URL (default: https://rmassistanthub.io/mcp).
        api_key: JWT Bearer token or Hub API key (ahk_xxx). Falls back
                 to ASSISTANT_HUB_API_KEY env var.
        cache_tools: Whether to cache tools after first fetch (default: True).
        transport: MCP transport type (default: "streamable_http").

    Example:
        client = AssistantHubMCPClient(api_key="ahk_abc123")
        tools = await client.get_tools()
        # → [StructuredTool(get_prices), StructuredTool(get_fear_greed), ...]
    """

    def __init__(
        self,
        url: str = "https://rmassistanthub.io/mcp",
        api_key: Optional[str] = None,
        cache_tools: bool = True,
        transport: str = "streamable_http",
    ):
        self.url = url
        self.api_key = api_key or os.environ.get("ASSISTANT_HUB_API_KEY", "")
        self.cache_tools = cache_tools
        self.transport = transport
        self._tools: Optional[List[BaseTool]] = None
        self._client: Any = None

    def _build_headers(self) -> Dict[str, str]:
        """Build auth headers based on API key format."""
        if not self.api_key:
            return {}
        if self.api_key.startswith("ahk_"):
            return {"X-API-Key": self.api_key}
        return {"Authorization": f"Bearer {self.api_key}"}

    async def get_tools(self) -> List[BaseTool]:
        """
        Fetch and return all available tools from the MCP server.

        Uses langchain-mcp-adapters to auto-discover tools via the MCP protocol.
        Results are cached by default for subsequent calls.

        Returns:
            List of LangChain BaseTool instances ready for agent use.

        Raises:
            ImportError: If langchain-mcp-adapters is not installed.
        """
        if self._tools is not None and self.cache_tools:
            return self._tools

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            raise ImportError(
                "langchain-mcp-adapters is required for MCP mode. "
                "Install it with: pip install langchain-mcp-adapters"
            )

        client = MultiServerMCPClient(
            {
                "assistant-hub": {
                    "transport": self.transport,
                    "url": self.url,
                    "headers": self._build_headers(),
                }
            }
        )

        self._client = client
        tools = await client.get_tools()
        self._tools = tools
        return tools

    async def close(self) -> None:
        """Close the MCP client connection."""
        if self._client and hasattr(self._client, "close"):
            await self._client.close()
        self._client = None
        self._tools = None

    @classmethod
    def from_api_key(cls, api_key: str, **kwargs) -> "AssistantHubMCPClient":
        """Convenience constructor from an API key."""
        return cls(api_key=api_key, **kwargs)

    @classmethod
    def from_env(cls, **kwargs) -> "AssistantHubMCPClient":
        """Create client using ASSISTANT_HUB_API_KEY environment variable."""
        return cls(**kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
