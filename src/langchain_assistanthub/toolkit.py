"""
AssistantHubToolkit — LangChain Community Toolkit

Wraps the Assistant Hub MCP server and exposes all tools with
human-readable names, descriptions, and typed input schemas.

Supports two authentication modes:
  1. JWT (api_key) — for Pro/Premium subscribers
  2. x402 (no key) — pay-per-call with USDC on Base

Usage:
    from langchain_assistanthub import AssistantHubToolkit
    toolkit = AssistantHubToolkit(api_key="your-jwt")
    tools = toolkit.get_tools()
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

from langchain_core.tools import BaseTool

from langchain_assistanthub.execution import (
    AssistantHubCheckApproval,
    AssistantHubExecuteTrade,
)
from langchain_assistanthub.price_monitor import AssistantHubPriceMonitor
from langchain_assistanthub.strategy import AssistantHubStrategyAnalysis
from langchain_assistanthub.tools import (
    AssistantHubAIForecast,
    AssistantHubCreateAlert,
    AssistantHubCryptoNews,
    AssistantHubDailyPulse,
    AssistantHubFearGreed,
    AssistantHubLivePrices,
    AssistantHubMonteCarloBacktest,
    AssistantHubRiskScores,
    AssistantHubSlippageEstimate,
)

# ── Tool Registry ────────────────────────────────────────────────────

# (tool_class, is_premium)
_ALL_TOOLS: List[tuple[type[BaseTool], bool]] = [
    # Free tools
    (AssistantHubLivePrices, False),
    (AssistantHubFearGreed, False),
    (AssistantHubCryptoNews, False),
    (AssistantHubRiskScores, False),
    (AssistantHubDailyPulse, False),
    # Premium tools
    (AssistantHubAIForecast, True),
    (AssistantHubMonteCarloBacktest, True),
    (AssistantHubSlippageEstimate, True),
    (AssistantHubCreateAlert, True),
    (AssistantHubStrategyAnalysis, True),
    (AssistantHubExecuteTrade, True),
    (AssistantHubCheckApproval, True),
]


class AssistantHubToolkit:
    """
    LangChain toolkit that loads all Assistant Hub crypto intelligence tools.

    Args:
        api_key:  JWT token or Hub API key. Falls back to ASSISTANT_HUB_API_KEY env var.
        base_url: Hub instance URL (default: https://rmassistanthub.io).
        include_premium: Whether to include premium (x402) tools (default: True).
        tools:    Explicit list of tool names to load (default: all).
        max_retries: Number of retries on transient failures (default: 2).
        timeout:  Request timeout in seconds (default: 30).
        enable_price_feed: Start WebSocket price feed for live monitoring (default: False).
        price_feed_coins: Coins to subscribe to when price feed is enabled (default: all).

    Example:
        toolkit = AssistantHubToolkit(api_key="ahk_abc123")
        tools = toolkit.get_tools()

        # Only free tools:
        toolkit = AssistantHubToolkit(api_key="...", include_premium=False)

        # With live price monitoring:
        toolkit = AssistantHubToolkit(
            api_key="...",
            enable_price_feed=True,
            price_feed_coins=["BTC", "ETH", "SOL"],
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://rmassistanthub.io",
        include_premium: bool = True,
        tools: Optional[Sequence[str]] = None,
        max_retries: int = 2,
        timeout: int = 30,
        enable_price_feed: bool = False,
        price_feed_coins: Optional[List[str]] = None,
    ):
        self.api_key = api_key or os.environ.get("ASSISTANT_HUB_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.include_premium = include_premium
        self.tool_filter = set(tools) if tools else None
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_price_feed = enable_price_feed
        self.price_feed_coins = price_feed_coins

        # Price feed state (lazy-initialized)
        self._price_feed = None
        self._price_buffer = None

        # Anonymous telemetry (opt out: ASSISTANT_HUB_TELEMETRY_OPT_OUT=1)
        from langchain_assistanthub._telemetry import _send_telemetry

        _send_telemetry(self.base_url, has_api_key=bool(self.api_key))

    def get_tools(self) -> List[BaseTool]:
        """
        Return instantiated LangChain tools.

        Each tool wraps an HTTP call to the Assistant Hub API
        with authentication, retry logic, and error handling.

        When enable_price_feed=True, also includes the PriceMonitor tool
        and starts the WebSocket price feed.
        """
        result: List[BaseTool] = []

        for tool_cls, is_premium in _ALL_TOOLS:
            if is_premium and not self.include_premium:
                continue
            if (
                self.tool_filter
                and tool_cls.model_fields["hub_tool_id"].default not in self.tool_filter
            ):
                continue

            tool = tool_cls(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
            result.append(tool)

        # Add PriceMonitor if price feed is enabled
        if self.enable_price_feed:
            monitor = AssistantHubPriceMonitor(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
            # Lazy-start the price feed
            if self._price_buffer is None:
                self._init_price_feed()
            if self._price_buffer is not None:
                monitor.set_price_buffer(self._price_buffer)
            result.append(monitor)

        return result

    def _init_price_feed(self) -> None:
        """Initialize the WebSocket price feed (lazy, called once)."""
        try:
            from langchain_assistanthub.price_feed import PriceFeedRunnable

            self._price_feed = PriceFeedRunnable(
                api_key=self.api_key,
                base_url=self.base_url,
                coins=self.price_feed_coins,
            )
            self._price_buffer = self._price_feed.buffer
            self._price_feed.start()
        except ImportError:
            pass  # websockets not installed

    @property
    def price_feed(self):
        """Access the PriceFeedRunnable (if enabled)."""
        return self._price_feed

    @property
    def price_buffer(self):
        """Access the shared PriceBuffer (if price feed enabled)."""
        return self._price_buffer

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a single tool by its Hub tool ID (e.g. 'live_prices')."""
        for tool in self.get_tools():
            if getattr(tool, "hub_tool_id", None) == name:
                return tool
        return None

    @property
    def available_tools(self) -> List[str]:
        """List all available tool IDs."""
        ids = [
            cls.model_fields["hub_tool_id"].default
            for cls, is_premium in _ALL_TOOLS
            if (not is_premium or self.include_premium)
        ]
        if self.enable_price_feed:
            ids.append("price_monitor")
        return ids

    def get_tool_metadata(self, tool_name: str) -> dict:
        """Get metadata for a specific tool: tier, daily limits, x402 price.

        Lets agents self-query "is this premium? do I need to pay?"
        before invoking a tool.

        Args:
            tool_name: Hub tool ID (e.g. 'live_prices', 'ai_forecast')
                       or full tool name (e.g. 'assistant_hub_live_prices').

        Returns:
            Dict with tier_required, daily_limits, x402_price_usdc,
            staking_discount_pct, and description.

        Raises:
            ValueError: If the tool is not found.

        Example:
            meta = toolkit.get_tool_metadata("ai_forecast")
            if meta["tier_required"]:
                print(f"Premium tool — x402 price: ${meta['x402_price_usdc']}")
        """
        # Check against registry first (no instantiation needed)
        for tool_cls, is_premium in _ALL_TOOLS:
            hub_id = tool_cls.model_fields["hub_tool_id"].default
            tool_full_name = tool_cls.model_fields.get("name", None)
            full_name = tool_full_name.default if tool_full_name else ""

            if tool_name not in (hub_id, full_name):
                continue

            # Skip if premium excluded
            if is_premium and not self.include_premium:
                continue

            return {
                "hub_tool_id": hub_id,
                "name": full_name,
                "tier_required": is_premium,
                "daily_limits": {
                    "anonymous": 10,
                    "free": 50,
                    "pro": 200,
                    "premium": "unlimited",
                },
                "x402_price_usdc": 0.01 if is_premium else 0.0,
                "staking_discount_pct": 50,
                "description": tool_cls.model_fields.get("description", None).default
                if tool_cls.model_fields.get("description")
                else "",
            }

        available = [cls.model_fields["hub_tool_id"].default for cls, _ in _ALL_TOOLS]
        raise ValueError(f"Tool '{tool_name}' not found. Available: {available}")

    # ── Convenience Constructors ────────────────────────────────────

    @classmethod
    def from_api_key(cls, api_key: str, **kwargs) -> "AssistantHubToolkit":
        """Create toolkit from a Hub API key (JWT or ahk_* key).

        Example:
            toolkit = AssistantHubToolkit.from_api_key("ahk_abc123")
            tools = toolkit.get_tools()
        """
        return cls(api_key=api_key, **kwargs)

    @classmethod
    def from_env(cls, **kwargs) -> "AssistantHubToolkit":
        """Create toolkit from ASSISTANT_HUB_API_KEY env var.

        Example:
            export ASSISTANT_HUB_API_KEY="ahk_abc123"
            toolkit = AssistantHubToolkit.from_env()
        """
        api_key = os.environ.get("ASSISTANT_HUB_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ASSISTANT_HUB_API_KEY not set. Get your key at https://rmassistanthub.io/#payments"
            )
        return cls(api_key=api_key, **kwargs)

    @classmethod
    def from_hub_login(cls, email: str, password: str, **kwargs) -> "AssistantHubToolkit":
        """Create toolkit by logging into Assistant Hub with email/password.

        Authenticates against the Hub API and uses the returned JWT token.
        Useful for notebook / script workflows without managing API keys.

        Args:
            email: Your Assistant Hub account email.
            password: Your account password.
            **kwargs: Additional arguments passed to AssistantHubToolkit().

        Example:
            toolkit = AssistantHubToolkit.from_hub_login(
                "user@example.com", "mypassword"
            )
            tools = toolkit.get_tools()

        Raises:
            ValueError: If login fails (wrong credentials, network error).
        """
        import json
        import urllib.error
        import urllib.request

        base_url = kwargs.pop("base_url", "https://rmassistanthub.io").rstrip("/")
        url = f"{base_url}/api/auth/login"

        payload = json.dumps({"email": email, "password": password}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read())
                detail = body.get("error", str(e))
            except Exception:
                detail = str(e)
            raise ValueError(
                f"Login failed: {detail}. Check credentials or sign up at https://rmassistanthub.io"
            ) from None
        except Exception as e:
            raise ValueError(f"Login failed: {e}") from None

        token = data.get("token")
        if not token:
            raise ValueError("Login response missing token. Check API version.")

        return cls(api_key=token, base_url=base_url, **kwargs)

    @classmethod
    async def from_mcp(
        cls,
        url: str = "https://rmassistanthub.io/mcp",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> List[BaseTool]:
        """Auto-discover tools via MCP protocol (uses langchain-mcp-adapters).

        Returns raw LangChain tools from the MCP server instead of the
        pre-defined tool wrappers. Useful when the server adds new tools
        that aren't yet in this package.

        Example:
            tools = await AssistantHubToolkit.from_mcp(api_key="ahk_abc123")
            agent = create_react_agent(model, tools)

        Requires:
            pip install langchain-mcp-adapters
        """
        from langchain_assistanthub.client import AssistantHubMCPClient

        client = AssistantHubMCPClient(url=url, api_key=api_key, **kwargs)
        return await client.get_tools()
