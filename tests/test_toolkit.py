"""
Tests for langchain-assistanthub toolkit.

Run with: pytest tests/
"""

import os
from unittest.mock import patch


class TestAssistantHubToolkit:
    """Tests for the main AssistantHubToolkit class."""

    def test_import(self):
        """Package imports correctly."""
        from langchain_assistanthub import AssistantHubToolkit

        assert AssistantHubToolkit is not None

    def test_version(self):
        """Version is accessible."""
        from langchain_assistanthub import __version__

        assert __version__ == "0.1.0"

    def test_toolkit_init_defaults(self):
        """Toolkit initializes with defaults."""
        from langchain_assistanthub import AssistantHubToolkit

        toolkit = AssistantHubToolkit()
        assert toolkit.base_url == "https://rmassistanthub.io"
        assert toolkit.include_premium is True
        assert toolkit.max_retries == 2
        assert toolkit.timeout == 30

    def test_toolkit_init_with_api_key(self):
        """Toolkit accepts API key."""
        from langchain_assistanthub import AssistantHubToolkit

        toolkit = AssistantHubToolkit(api_key="ahk_test123")
        assert toolkit.api_key == "ahk_test123"

    def test_toolkit_init_from_env(self):
        """Toolkit reads API key from environment."""
        from langchain_assistanthub import AssistantHubToolkit

        with patch.dict(os.environ, {"ASSISTANT_HUB_API_KEY": "ahk_env_key"}):
            toolkit = AssistantHubToolkit()
            assert toolkit.api_key == "ahk_env_key"

    def test_toolkit_custom_base_url(self):
        """Toolkit strips trailing slash from base URL."""
        from langchain_assistanthub import AssistantHubToolkit

        toolkit = AssistantHubToolkit(base_url="http://localhost:3000/")
        assert toolkit.base_url == "http://localhost:3000"

    def test_get_tools_returns_list(self):
        """get_tools() returns a list of BaseTool instances."""
        from langchain_assistanthub import AssistantHubToolkit

        toolkit = AssistantHubToolkit(api_key="test")
        tools = toolkit.get_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_tools_free_only(self):
        """include_premium=False excludes premium tools."""
        from langchain_assistanthub import AssistantHubToolkit

        toolkit = AssistantHubToolkit(api_key="test", include_premium=False)
        tools = toolkit.get_tools()
        for tool in tools:
            assert not getattr(tool, "hub_premium", False), (
                f"Premium tool {tool.name} should not be included"
            )

    def test_get_tools_filter(self):
        """tools= filter limits returned tools."""
        from langchain_assistanthub import AssistantHubToolkit

        toolkit = AssistantHubToolkit(
            api_key="test",
            tools=["live_prices", "fear_greed"],
        )
        tools = toolkit.get_tools()
        tool_ids = [getattr(t, "hub_tool_id", "") for t in tools]
        assert "live_prices" in tool_ids
        assert "fear_greed" in tool_ids
        assert "crypto_news" not in tool_ids

    def test_available_tools_list(self):
        """available_tools returns list of IDs."""
        from langchain_assistanthub import AssistantHubToolkit

        toolkit = AssistantHubToolkit(api_key="test")
        ids = toolkit.available_tools
        assert "live_prices" in ids
        assert "fear_greed" in ids
        assert "ai_forecast" in ids  # premium included by default

    def test_available_tools_no_premium(self):
        """available_tools excludes premium when disabled."""
        from langchain_assistanthub import AssistantHubToolkit

        toolkit = AssistantHubToolkit(api_key="test", include_premium=False)
        ids = toolkit.available_tools
        assert "live_prices" in ids
        assert "ai_forecast" not in ids

    def test_get_tool_by_id(self):
        """get_tool() returns a single tool."""
        from langchain_assistanthub import AssistantHubToolkit

        toolkit = AssistantHubToolkit(api_key="test")
        tool = toolkit.get_tool("live_prices")
        assert tool is not None
        assert tool.name == "assistant_hub_live_prices"

    def test_get_tool_missing(self):
        """get_tool() returns None for unknown ID."""
        from langchain_assistanthub import AssistantHubToolkit

        toolkit = AssistantHubToolkit(api_key="test")
        tool = toolkit.get_tool("nonexistent_tool")
        assert tool is None


class TestToolExports:
    """Test that all tool classes are exported."""

    def test_core_tools_importable(self):
        from langchain_assistanthub import (
            AssistantHubLivePrices,
        )

        assert AssistantHubLivePrices is not None

    def test_premium_tools_importable(self):
        from langchain_assistanthub import (
            AssistantHubAIForecast,
        )

        assert AssistantHubAIForecast is not None

    def test_strategy_importable(self):
        from langchain_assistanthub import (
            AssistantHubStrategyAnalysis,
        )

        assert AssistantHubStrategyAnalysis is not None

    def test_execution_importable(self):
        from langchain_assistanthub import (
            AssistantHubExecuteTrade,
        )

        assert AssistantHubExecuteTrade is not None

    def test_price_feed_importable(self):
        from langchain_assistanthub import (
            PriceBuffer,
        )

        assert PriceBuffer is not None


class TestMCPClient:
    """Test the MCP client wrapper."""

    def test_client_importable(self):
        from langchain_assistanthub.client import AssistantHubMCPClient

        assert AssistantHubMCPClient is not None

    def test_client_init_defaults(self):
        from langchain_assistanthub.client import AssistantHubMCPClient

        client = AssistantHubMCPClient()
        assert client.url == "https://rmassistanthub.io/mcp"
        assert client.cache_tools is True

    def test_client_builds_jwt_headers(self):
        from langchain_assistanthub.client import AssistantHubMCPClient

        client = AssistantHubMCPClient(api_key="my-jwt-token")
        headers = client._build_headers()
        assert headers == {"Authorization": "Bearer my-jwt-token"}

    def test_client_builds_apikey_headers(self):
        from langchain_assistanthub.client import AssistantHubMCPClient

        client = AssistantHubMCPClient(api_key="ahk_abc123")
        headers = client._build_headers()
        assert headers == {"X-API-Key": "ahk_abc123"}

    def test_client_from_api_key(self):
        from langchain_assistanthub.client import AssistantHubMCPClient

        client = AssistantHubMCPClient.from_api_key("ahk_test")
        assert client.api_key == "ahk_test"

    def test_client_from_env(self):
        from langchain_assistanthub.client import AssistantHubMCPClient

        with patch.dict(os.environ, {"ASSISTANT_HUB_API_KEY": "ahk_env"}):
            client = AssistantHubMCPClient.from_env()
            assert client.api_key == "ahk_env"


class TestPriceBuffer:
    """Test the PriceBuffer data structure."""

    def test_buffer_update_and_latest(self):
        from langchain_assistanthub import PriceBuffer

        buf = PriceBuffer()
        buf.update("BTC", 50000.0, 1000)
        assert buf.latest("BTC") == 50000.0

    def test_buffer_unknown_coin(self):
        from langchain_assistanthub import PriceBuffer

        buf = PriceBuffer()
        assert buf.latest("UNKNOWN") is None

    def test_buffer_max_entries(self):
        from langchain_assistanthub import PriceBuffer

        buf = PriceBuffer(max_entries=5)
        for i in range(10):
            buf.update("BTC", float(i), i * 1000)
        assert buf.latest("BTC") == 9.0
        assert len(buf.history("BTC", minutes=999)) <= 5

    def test_buffer_pct_change(self):
        import time

        from langchain_assistanthub import PriceBuffer

        now = int(time.time() * 1000)
        buf = PriceBuffer()
        buf.update("BTC", 100.0, now - 60_000)  # 1 min ago
        buf.update("BTC", 110.0, now)  # now
        change = buf.pct_change("BTC", minutes=5)
        assert change is not None
        assert abs(change - 10.0) < 0.01

    def test_buffer_tracked_coins(self):
        from langchain_assistanthub import PriceBuffer

        buf = PriceBuffer()
        buf.update("BTC", 50000.0, 1000)
        buf.update("ETH", 3000.0, 1000)
        assert set(buf.tracked_coins) == {"BTC", "ETH"}
