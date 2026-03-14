"""
Individual LangChain tools wrapping Assistant Hub API endpoints.

Each tool:
  - Has a human-readable name + description for LLM function calling
  - Defines a Pydantic input schema for structured args
  - Makes HTTP calls with JWT auth + retry logic
  - Returns JSON results directly usable by LangGraph agents

Premium tools are flagged — they require Pro/Premium JWT or x402 payment.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type

import aiohttp
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_assistanthub.exceptions import (
    AssistantHubForbiddenError,
    AssistantHubPaymentRequiredError,
    AssistantHubRateLimitError,
    AssistantHubServerError,
)

# ── Base Tool ────────────────────────────────────────────────────────


class AssistantHubBaseTool(BaseTool):
    """Base class for all Assistant Hub tools with shared HTTP logic."""

    # Subclass metadata (set by each tool)
    hub_tool_id: str = ""
    hub_endpoint: str = ""
    hub_method: str = "GET"
    hub_premium: bool = False

    # Config (set by toolkit)
    api_key: str = ""
    base_url: str = "https://rmassistanthub.io"
    max_retries: int = 2
    timeout: int = 30

    class Config:
        arbitrary_types_allowed = True

    async def _hub_request(
        self,
        path: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated request to the Hub API with retry."""
        url = f"{self.base_url}{path or self.hub_endpoint}"
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        if self.api_key:
            # Support both JWT and API key formats
            if self.api_key.startswith("ahk_"):
                headers["X-API-Key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    kwargs: Dict[str, Any] = {"headers": headers}
                    if params:
                        kwargs["params"] = params
                    if body and self.hub_method in ("POST", "PUT"):
                        kwargs["json"] = body

                    async with session.request(self.hub_method, url, **kwargs) as resp:
                        if resp.status == 402:
                            # Premium tool — raise with x402 payment info
                            try:
                                err_body = await resp.json()
                                detail = err_body.get("detail", "Premium tool requires payment.")
                            except Exception:
                                detail = "Premium tool requires payment."
                            raise AssistantHubPaymentRequiredError(str(detail))

                        if resp.status == 403:
                            try:
                                err_body = await resp.json()
                                detail = err_body.get("detail", "Access forbidden.")
                            except Exception:
                                detail = "Access forbidden."
                            raise AssistantHubForbiddenError(str(detail))

                        if resp.status == 429:
                            # Rate limited — retry after delay
                            if attempt < self.max_retries:
                                import asyncio

                                await asyncio.sleep(2**attempt)
                                continue
                            # All retries exhausted — raise with upgrade CTA
                            try:
                                err_body = await resp.json()
                                detail = err_body.get(
                                    "detail",
                                    err_body.get("error", "Rate limit exceeded."),
                                )
                            except Exception:
                                detail = "Rate limit exceeded."
                            raise AssistantHubRateLimitError(str(detail))

                        if resp.status >= 500:
                            try:
                                err_body = await resp.json()
                                detail = err_body.get("detail", "Internal server error.")
                            except Exception:
                                detail = await resp.text()
                            raise AssistantHubServerError(str(detail)[:500])

                        if resp.status >= 400:
                            text = await resp.text()
                            return {
                                "error": f"http_{resp.status}",
                                "message": text[:500],
                            }

                        return await resp.json()

            except (aiohttp.ClientError, TimeoutError) as e:
                last_error = e
                if attempt < self.max_retries:
                    import asyncio

                    await asyncio.sleep(2**attempt)
                    continue

        return {"error": "connection_failed", "message": str(last_error)}

    def _run(self, **kwargs: Any) -> str:
        """Sync wrapper — runs the async implementation."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — create a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, self._arun(**kwargs)).result()
            return result
        else:
            return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: Any) -> str:
        """Override in subclasses."""
        raise NotImplementedError


# ── Input Schemas ────────────────────────────────────────────────────


class CoinInput(BaseModel):
    """Input for coin-specific queries."""

    coin: str = Field(
        default="BTC",
        description="Cryptocurrency symbol (e.g., BTC, ETH, SOL, DOGE, AVAX, LINK, ADA, DOT)",
    )


class OptionalCoinInput(BaseModel):
    """Input that optionally takes a coin."""

    coin: Optional[str] = Field(
        default=None,
        description="Optional cryptocurrency symbol to filter results",
    )


class BacktestInput(BaseModel):
    """Input for Monte Carlo backtesting."""

    coin: str = Field(description="Cryptocurrency symbol (e.g., BTC, ETH, SOL)")
    strategy: str = Field(
        default="momentum",
        description="Strategy type: momentum, mean_reversion, breakout, or rsi",
    )
    period_days: int = Field(default=90, description="Backtest period in days (30-365)")
    simulations: int = Field(default=1000, description="Monte Carlo simulations (100-10000)")


class SlippageInput(BaseModel):
    """Input for slippage estimation."""

    coin: str = Field(description="Cryptocurrency symbol")
    amount_usd: float = Field(description="Trade size in USD")
    side: str = Field(default="buy", description="Trade side: buy or sell")


class AlertInput(BaseModel):
    """Input for creating price alerts."""

    coin: str = Field(description="Cryptocurrency symbol")
    condition: str = Field(description="Alert condition: above, below, or change_pct")
    value: float = Field(description="Trigger value (price in USD or percentage)")


class EmptyInput(BaseModel):
    """No input required."""

    pass


# ── Individual Tools ─────────────────────────────────────────────────


class AssistantHubLivePrices(AssistantHubBaseTool):
    """Get live cryptocurrency prices for 8 major coins."""

    name: str = "assistant_hub_live_prices"
    description: str = (
        "Get real-time cryptocurrency prices including BTC, ETH, SOL, DOGE, AVAX, LINK, ADA, DOT. "
        "Returns current price, 24h change, volume, and market cap for each coin."
    )
    hub_tool_id: str = "live_prices"
    hub_endpoint: str = "/api/crypto/prices"
    hub_method: str = "GET"
    hub_premium: bool = False
    args_schema: Type[BaseModel] = EmptyInput

    async def _arun(self, **kwargs: Any) -> str:
        result = await self._hub_request()
        return json.dumps(result, indent=2)


class AssistantHubFearGreed(AssistantHubBaseTool):
    """Get the Crypto Fear & Greed Index."""

    name: str = "assistant_hub_fear_greed"
    description: str = (
        "Get the current Crypto Fear & Greed Index (0-100). "
        "0 = Extreme Fear, 100 = Extreme Greed. Useful for gauging market sentiment."
    )
    hub_tool_id: str = "fear_greed"
    hub_endpoint: str = "/api/crypto/fear-greed"
    hub_method: str = "GET"
    hub_premium: bool = False
    args_schema: Type[BaseModel] = EmptyInput

    async def _arun(self, **kwargs: Any) -> str:
        result = await self._hub_request()
        return json.dumps(result, indent=2)


class AssistantHubCryptoNews(AssistantHubBaseTool):
    """Get latest crypto news headlines."""

    name: str = "assistant_hub_crypto_news"
    description: str = (
        "Get the latest cryptocurrency news headlines from CryptoCompare. "
        "Returns 12 recent headlines with titles, sources, and timestamps."
    )
    hub_tool_id: str = "crypto_news"
    hub_endpoint: str = "/api/crypto/news"
    hub_method: str = "GET"
    hub_premium: bool = False
    args_schema: Type[BaseModel] = EmptyInput

    async def _arun(self, **kwargs: Any) -> str:
        result = await self._hub_request()
        return json.dumps(result, indent=2)


class AssistantHubRiskScores(AssistantHubBaseTool):
    """Get AI-computed risk scores for cryptocurrencies."""

    name: str = "assistant_hub_risk_scores"
    description: str = (
        "Get composite risk scores (0-100) for cryptocurrencies combining technical, "
        "economic, and sentiment analysis. Higher score = higher risk. "
        "Includes risk level classification (LOW/MEDIUM/HIGH/EXTREME)."
    )
    hub_tool_id: str = "risk_scores"
    hub_endpoint: str = "/api/risk/scores"
    hub_method: str = "GET"
    hub_premium: bool = False
    args_schema: Type[BaseModel] = OptionalCoinInput

    async def _arun(self, coin: Optional[str] = None, **kwargs: Any) -> str:
        params = {}
        if coin:
            params["coin"] = coin.upper()
        result = await self._hub_request(params=params)
        return json.dumps(result, indent=2)


class AssistantHubDailyPulse(AssistantHubBaseTool):
    """Get the Daily Macro Pulse — top threats and opportunities."""

    name: str = "assistant_hub_daily_pulse"
    description: str = (
        "Get today's Daily Macro Pulse: AI-generated analysis of the top 3 macro threats "
        "and top 3 crypto opportunities. Updated daily at 06:00 UTC."
    )
    hub_tool_id: str = "daily_pulse"
    hub_endpoint: str = "/api/pulse"
    hub_method: str = "GET"
    hub_premium: bool = False
    args_schema: Type[BaseModel] = EmptyInput

    async def _arun(self, **kwargs: Any) -> str:
        result = await self._hub_request()
        return json.dumps(result, indent=2)


class AssistantHubAIForecast(AssistantHubBaseTool):
    """Get AI-powered price forecast for a cryptocurrency. (Premium)"""

    name: str = "assistant_hub_ai_forecast"
    description: str = (
        "Get an AI-powered price forecast for a specific cryptocurrency. "
        "Uses historical price patterns and market indicators to generate "
        "short-term (24h) and medium-term (7d) predictions with confidence levels. "
        "PREMIUM: Requires Pro/Premium tier or x402 USDC payment."
    )
    hub_tool_id: str = "ai_forecast"
    hub_endpoint: str = "/api/crypto/forecast"
    hub_method: str = "GET"
    hub_premium: bool = True
    args_schema: Type[BaseModel] = CoinInput

    async def _arun(self, coin: str = "BTC", **kwargs: Any) -> str:
        result = await self._hub_request(params={"coin": coin.upper()})
        return json.dumps(result, indent=2)


class AssistantHubMonteCarloBacktest(AssistantHubBaseTool):
    """Run a Monte Carlo strategy backtest. (Premium)"""

    name: str = "assistant_hub_monte_carlo_backtest"
    description: str = (
        "Run a Monte Carlo simulation backtest on a cryptocurrency trading strategy. "
        "Tests strategies (momentum, mean_reversion, breakout, rsi) with configurable "
        "period and simulation count. Returns expected returns, VaR, Sharpe ratio, "
        "win rate, and distribution statistics. "
        "PREMIUM: Requires Pro/Premium tier or x402 USDC payment."
    )
    hub_tool_id: str = "monte_carlo_backtest"
    hub_endpoint: str = "/api/backtest/run"
    hub_method: str = "POST"
    hub_premium: bool = True
    args_schema: Type[BaseModel] = BacktestInput

    async def _arun(
        self,
        coin: str = "BTC",
        strategy: str = "momentum",
        period_days: int = 90,
        simulations: int = 1000,
        **kwargs: Any,
    ) -> str:
        result = await self._hub_request(
            body={
                "coin": coin.upper(),
                "strategy": strategy,
                "periodDays": period_days,
                "simulations": simulations,
            }
        )
        return json.dumps(result, indent=2)


class AssistantHubSlippageEstimate(AssistantHubBaseTool):
    """Estimate trade slippage for a given order size. (Premium)"""

    name: str = "assistant_hub_slippage_estimate"
    description: str = (
        "Estimate the expected slippage for a cryptocurrency trade of a given size. "
        "Uses order book depth and daily volume data to predict price impact. "
        "Returns slippage percentage, effective price, and fee breakdown. "
        "PREMIUM: Requires Pro/Premium tier or x402 USDC payment."
    )
    hub_tool_id: str = "slippage_estimate"
    hub_endpoint: str = "/api/v1/slippage"
    hub_method: str = "GET"
    hub_premium: bool = True
    args_schema: Type[BaseModel] = SlippageInput

    async def _arun(
        self,
        coin: str = "BTC",
        amount_usd: float = 1000.0,
        side: str = "buy",
        **kwargs: Any,
    ) -> str:
        result = await self._hub_request(
            params={
                "coin": coin.upper(),
                "amountUsd": str(amount_usd),
                "side": side,
            }
        )
        return json.dumps(result, indent=2)


class AssistantHubCreateAlert(AssistantHubBaseTool):
    """Create a price or change alert. (Premium)"""

    name: str = "assistant_hub_create_alert"
    description: str = (
        "Create a price alert for a cryptocurrency. Triggers when the price crosses "
        "above or below a threshold, or when 24h change exceeds a percentage. "
        "PREMIUM: Requires Pro/Premium tier or x402 USDC payment."
    )
    hub_tool_id: str = "create_alert"
    hub_endpoint: str = "/api/v1/alerts"
    hub_method: str = "POST"
    hub_premium: bool = True
    args_schema: Type[BaseModel] = AlertInput

    async def _arun(
        self,
        coin: str = "BTC",
        condition: str = "above",
        value: float = 100000.0,
        **kwargs: Any,
    ) -> str:
        result = await self._hub_request(
            body={
                "coin": coin.upper(),
                "condition": condition,
                "value": value,
            }
        )
        return json.dumps(result, indent=2)
