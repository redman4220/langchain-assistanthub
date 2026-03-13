"""
Real-Time Price Feed — LangChain Memory / Runnable

Connects to the Assistant Hub WebSocket price stream and maintains
a rolling price buffer. Agents get live prices injected into context
without polling.

Components:
  PriceBuffer        — Thread-safe rolling price buffer with pct_change()
  PriceFeedRunnable  — LangChain Runnable that connects to WS, returns snapshots
  PriceFeedCallbackHandler — Auto-injects latest prices into agent prompts

Usage:
    from langchain_assistanthub import AssistantHubToolkit

    toolkit = AssistantHubToolkit(
        api_key="ahk_xxx",
        enable_price_feed=True,
        price_feed_coins=["BTC", "ETH", "SOL"],
    )

    # Get callback handler for auto-injection into agent context
    handler = toolkit.get_price_feed_handler()

    # Or access the feed directly
    feed = toolkit.price_feed
    await feed.start()
    snapshot = await feed.ainvoke({})
    # → {"prices": {"BTC": 65000, "ETH": 3200}, "changes_5m": {"BTC": 1.2}, "ts": ...}
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

try:
    import websockets
    import websockets.client
except ImportError:
    websockets = None  # type: ignore

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableSerializable

# ── PriceBuffer ──────────────────────────────────────────────────


class PriceBuffer:
    """
    Thread-safe rolling price buffer.

    Stores (timestamp_ms, price) tuples per coin in a deque with
    configurable max length. Default 300 entries ≈ 5 minutes of
    1-second price ticks.
    """

    def __init__(self, max_entries: int = 300):
        self._data: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._max = max_entries

    def update(self, coin: str, price: float, ts: int) -> None:
        """Record a price tick."""
        with self._lock:
            if coin not in self._data:
                self._data[coin] = deque(maxlen=self._max)
            self._data[coin].append((ts, price))

    def latest(self, coin: str) -> Optional[float]:
        """Get the most recent price for a coin."""
        with self._lock:
            buf = self._data.get(coin)
            if not buf:
                return None
            return buf[-1][1]

    def all_latest(self) -> Dict[str, float]:
        """Get the most recent price for all tracked coins."""
        with self._lock:
            return {coin: buf[-1][1] for coin, buf in self._data.items() if buf}

    def history(self, coin: str, minutes: int = 5) -> List[Tuple[int, float]]:
        """Get price history for a coin within the last N minutes."""
        cutoff = int(time.time() * 1000) - (minutes * 60_000)
        with self._lock:
            buf = self._data.get(coin)
            if not buf:
                return []
            return [(ts, p) for ts, p in buf if ts >= cutoff]

    def pct_change(self, coin: str, minutes: int = 5) -> Optional[float]:
        """
        Calculate percentage price change over the last N minutes.
        Returns None if insufficient data.
        """
        hist = self.history(coin, minutes)
        if len(hist) < 2:
            return None
        first_price = hist[0][1]
        last_price = hist[-1][1]
        if first_price == 0:
            return None
        return ((last_price - first_price) / first_price) * 100

    def all_changes(self, minutes: int = 5) -> Dict[str, float]:
        """Get pct_change for all tracked coins."""
        result = {}
        with self._lock:
            coins = list(self._data.keys())
        for coin in coins:
            change = self.pct_change(coin, minutes)
            if change is not None:
                result[coin] = round(change, 4)
        return result

    @property
    def tracked_coins(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())


# ── PriceFeedRunnable ────────────────────────────────────────────


class PriceFeedRunnable(RunnableSerializable[Dict, Dict]):
    """
    LangChain Runnable that connects to the Assistant Hub WebSocket
    price stream and returns live price snapshots on invoke().

    Maintains a background asyncio task for the WS connection.
    Call .start() before use, .stop() when done.
    """

    api_key: str = ""
    base_url: str = "https://rmassistanthub.io"
    coins: Optional[List[str]] = None
    buffer_minutes: int = 5
    reconnect_delay: float = 3.0

    # Internal state (not serialized)
    _buffer: Optional[PriceBuffer] = None
    _task: Optional[asyncio.Task] = None
    _running: bool = False

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        max_entries = self.buffer_minutes * 60  # ~1 tick/sec
        self._buffer = PriceBuffer(max_entries=max_entries)
        self._running = False
        self._task = None

    @property
    def buffer(self) -> PriceBuffer:
        """Access the underlying price buffer."""
        if self._buffer is None:
            self._buffer = PriceBuffer(max_entries=self.buffer_minutes * 60)
        return self._buffer

    async def start(self) -> None:
        """Start the background WebSocket connection."""
        if self._running:
            return
        if websockets is None:
            raise ImportError(
                "websockets is required for PriceFeedRunnable. Install with: pip install websockets"
            )
        self._running = True
        self._task = asyncio.create_task(self._ws_loop())

    async def stop(self) -> None:
        """Stop the background WebSocket connection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _ws_loop(self) -> None:
        """Background loop: connect, subscribe, read messages, reconnect."""
        ws_url = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        uri = f"{ws_url}/api/v1/ws/prices?apiKey={self.api_key}"

        while self._running:
            try:
                async with websockets.client.connect(uri) as ws:
                    # Wait for welcome message
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    json.loads(raw)  # consume welcome message

                    # Subscribe to specific coins if requested
                    if self.coins:
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "subscribe",
                                    "coins": [c.upper() for c in self.coins],
                                }
                            )
                        )

                    # Read messages
                    while self._running:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=45)
                            data = json.loads(raw)

                            event = data.get("event")

                            if event == "prices":
                                prices = data.get("data", {})
                                ts = data.get("ts", int(time.time() * 1000))
                                for coin, price in prices.items():
                                    if isinstance(price, (int, float)):
                                        self.buffer.update(coin, float(price), ts)

                            elif event == "ping":
                                await ws.send(json.dumps({"type": "pong"}))

                        except asyncio.TimeoutError:
                            # No message in 45s — server may have disconnected
                            break

            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            # Reconnect delay
            if self._running:
                await asyncio.sleep(self.reconnect_delay)

    def invoke(self, input: Dict, config: Any = None, **kwargs: Any) -> Dict:
        """Return current price snapshot (sync)."""
        return {
            "prices": self.buffer.all_latest(),
            "changes_5m": self.buffer.all_changes(5),
            "tracked_coins": self.buffer.tracked_coins,
            "ts": int(time.time() * 1000),
        }

    async def ainvoke(self, input: Dict, config: Any = None, **kwargs: Any) -> Dict:
        """Return current price snapshot (async)."""
        return self.invoke(input)

    # Context manager support
    async def __aenter__(self) -> "PriceFeedRunnable":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()


# ── PriceFeedCallbackHandler ─────────────────────────────────────


class PriceFeedCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that injects latest prices into
    agent prompts automatically.

    Prepends a system message with current prices and 5-minute
    changes at the start of each chain invocation.

    Usage:
        handler = PriceFeedCallbackHandler(feed=price_feed_runnable)
        agent = create_react_agent(model, tools, callbacks=[handler])
    """

    def __init__(self, feed: PriceFeedRunnable):
        super().__init__()
        self._feed = feed

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Inject latest prices into the chain input messages."""
        snapshot = self._feed.invoke({})
        prices = snapshot.get("prices", {})
        changes = snapshot.get("changes_5m", {})

        if not prices:
            return

        # Build price summary string
        lines = []
        for coin, price in sorted(prices.items()):
            change = changes.get(coin)
            change_str = f" ({change:+.2f}% 5m)" if change is not None else ""
            lines.append(f"  {coin}: ${price:,.2f}{change_str}")

        price_text = (
            f"[Live Market Data — {time.strftime('%H:%M:%S UTC', time.gmtime())}]\n"
            + "\n".join(lines)
        )

        # Inject into messages if present
        if "messages" in inputs and isinstance(inputs["messages"], list):
            from langchain_core.messages import SystemMessage

            inputs["messages"].insert(0, SystemMessage(content=price_text))
        elif "input" in inputs and isinstance(inputs["input"], str):
            inputs["input"] = f"{price_text}\n\n{inputs['input']}"
