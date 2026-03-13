"""
Price Monitor Tool — LangChain tool for continuous price monitoring.

Enables agents to check for significant price movements without polling.
Reads from the shared PriceBuffer maintained by PriceFeedRunnable.

Usage:
    toolkit = AssistantHubToolkit(api_key="...", enable_price_feed=True)
    tools = toolkit.get_tools()
    # AssistantHubPriceMonitor is automatically included

    # Agent can call: "Watch BTC for >5% move in 5 minutes"
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from langchain_assistanthub.price_feed import PriceBuffer
from langchain_assistanthub.tools import AssistantHubBaseTool


class PriceMonitorInput(BaseModel):
    """Input for price monitoring checks."""

    coin: str = Field(description="Cryptocurrency symbol (e.g., BTC, ETH, SOL)")
    threshold_pct: float = Field(
        default=5.0,
        description="Price change threshold in percent to trigger alert (e.g., 5.0 for 5%)",
    )
    window_minutes: int = Field(
        default=5,
        description="Time window in minutes to check for price change (1-60)",
    )
    direction: str = Field(
        default="any",
        description=("Direction: 'up' (increase only), 'down' (decrease only), 'any' (either)"),
    )


class AssistantHubPriceMonitor(AssistantHubBaseTool):
    """
    Check for significant price movements from the live WebSocket feed.

    Reads from the shared PriceBuffer (maintained by PriceFeedRunnable)
    to detect if a coin has moved by more than a threshold percentage
    within a given time window.

    Returns structured result indicating whether the threshold was
    triggered, the actual change percentage, and current price.
    """

    name: str = "assistant_hub_price_monitor"
    description: str = (
        "Check if a cryptocurrency has moved by more than a threshold percentage "
        "within a time window. Uses live WebSocket price data (no polling). "
        "Example: 'Has BTC moved more than 5% in the last 5 minutes?' "
        "Requires enable_price_feed=True in the toolkit."
    )
    hub_tool_id: str = "price_monitor"
    hub_endpoint: str = ""
    hub_method: str = "GET"
    hub_premium: bool = False
    args_schema: Type[BaseModel] = PriceMonitorInput

    # Shared price buffer (set by toolkit)
    _price_buffer: Optional[PriceBuffer] = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def set_price_buffer(self, buffer: PriceBuffer) -> None:
        """Set the shared price buffer (called by toolkit)."""
        self._price_buffer = buffer

    async def _arun(
        self,
        coin: str = "BTC",
        threshold_pct: float = 5.0,
        window_minutes: int = 5,
        direction: str = "any",
        **kwargs: Any,
    ) -> str:
        if self._price_buffer is None:
            return json.dumps(
                {
                    "error": "price_feed_not_enabled",
                    "message": (
                        "Price feed is not running. "
                        "Enable with: AssistantHubToolkit(enable_price_feed=True)"
                    ),
                }
            )

        coin = coin.upper()
        current_price = self._price_buffer.latest(coin)

        if current_price is None:
            return json.dumps(
                {
                    "error": "no_data",
                    "message": (
                        f"No price data for {coin}. "
                        "Coin may not be tracked or feed hasn't received data."
                    ),
                    "tracked_coins": self._price_buffer.tracked_coins,
                }
            )

        change_pct = self._price_buffer.pct_change(coin, window_minutes)
        history = self._price_buffer.history(coin, window_minutes)

        if change_pct is None:
            return json.dumps(
                {
                    "coin": coin,
                    "current_price": current_price,
                    "triggered": False,
                    "message": (
                        f"Insufficient history for {coin} "
                        f"in {window_minutes}min window ({len(history)} ticks)."
                    ),
                    "data_points": len(history),
                }
            )

        # Check threshold
        triggered = False
        if direction == "up":
            triggered = change_pct >= threshold_pct
        elif direction == "down":
            triggered = change_pct <= -threshold_pct
        else:  # "any"
            triggered = abs(change_pct) >= threshold_pct

        # Price range in window
        prices_in_window = [p for _, p in history]
        high = max(prices_in_window) if prices_in_window else current_price
        low = min(prices_in_window) if prices_in_window else current_price

        return json.dumps(
            {
                "coin": coin,
                "triggered": triggered,
                "current_price": current_price,
                "change_pct": round(change_pct, 4),
                "threshold_pct": threshold_pct,
                "direction": direction,
                "window_minutes": window_minutes,
                "window_high": high,
                "window_low": low,
                "data_points": len(history),
                "ts": int(time.time() * 1000),
            },
            indent=2,
        )
