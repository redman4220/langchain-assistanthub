# langchain-assistanthub

[![PyPI version](https://img.shields.io/pypi/v/langchain-assistanthub)](https://pypi.org/project/langchain-assistanthub/)
[![Downloads](https://img.shields.io/pypi/dm/langchain-assistanthub)](https://pypi.org/project/langchain-assistanthub/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/redman4220/langchain-assistanthub/actions/workflows/ci.yml/badge.svg)](https://github.com/redman4220/langchain-assistanthub/actions/workflows/ci.yml)

Crypto intelligence toolkit for LangChain agents. Real-time prices, risk scores, Monte Carlo backtests, AI forecasts, and more — powered by [Assistant Hub](https://rmassistanthub.io).

## Install

```bash
# Recommended: pin to current stable version
pip install langchain-assistanthub==0.1.2

# Or latest (for bleeding-edge users)
# pip install langchain-assistanthub
```

> We're at v0.1.2 — pinned install recommended for stability. Check the [changelog](https://github.com/redman4220/langchain-assistanthub/releases) for updates!

**Optional extras:**

```bash
pip install langchain-assistanthub[x402]      # USDC on-chain payments
pip install langchain-assistanthub[swarm]     # LangGraph multi-agent
pip install langchain-assistanthub[notebook]  # JupyterLab + pandas + matplotlib for examples
pip install langchain-assistanthub[dev]       # pytest, ruff, mypy
```

## Quick Start

### Option A: Pre-built toolkit (recommended)

```python
from langchain_assistanthub import AssistantHubToolkit
from langgraph.prebuilt import create_react_agent

toolkit = AssistantHubToolkit.from_api_key("ahk_your_key")
# or: toolkit = AssistantHubToolkit.from_env()   # reads ASSISTANT_HUB_API_KEY
tools = toolkit.get_tools()

agent = create_react_agent(model, tools)
result = agent.invoke({"messages": [
    {"role": "user", "content": "What's the risk score for SOL?"}
]})
```

### Option B: Login with Hub credentials

```python
from langchain_assistanthub import AssistantHubToolkit

toolkit = AssistantHubToolkit.from_hub_login("you@email.com", "password")
tools = toolkit.get_tools()
```

### Option C: MCP auto-discovery

Dynamically discover tools from the MCP server — picks up new tools
without upgrading the package:

```python
from langchain_assistanthub import AssistantHubToolkit

tools = await AssistantHubToolkit.from_mcp(api_key="ahk_your_key")
agent = create_react_agent(model, tools)
```

### Option D: MCP client directly

```python
from langchain_assistanthub import AssistantHubMCPClient

client = AssistantHubMCPClient.from_api_key("ahk_your_key")
tools = await client.get_tools()
```

## Available Tools

| Tool | Description | Tier |
|------|-------------|------|
| `live_prices` | Real-time prices for 8 major coins | Free |
| `fear_greed` | Crypto Fear & Greed Index | Free |
| `crypto_news` | Latest crypto headlines | Free |
| `risk_scores` | AI-computed composite risk scores | Free |
| `daily_pulse` | Daily macro threats + opportunities | Free |
| `ai_forecast` | AI price predictions (24h/7d) | Premium |
| `monte_carlo_backtest` | Strategy backtesting with Monte Carlo | Premium |
| `slippage_estimate` | Trade slippage estimation | Premium |
| `create_alert` | Price/change alerts | Premium |
| `strategy_analysis` | Full backtest + Monte Carlo + walk-forward | Premium |
| `execute_trade` | Paper/live trade execution | Premium |
| `check_approval` | Trade approval status | Premium |
| `price_monitor` | Live WebSocket price feed + alerts | Free* |

\* Requires `enable_price_feed=True` and `websockets` package.

## Advanced Usage

```python
# Free tools only
toolkit = AssistantHubToolkit(api_key="...", include_premium=False)

# Specific tools
toolkit = AssistantHubToolkit(api_key="...", tools=["live_prices", "risk_scores"])

# Custom instance (local dev)
toolkit = AssistantHubToolkit(api_key="...", base_url="http://localhost:3000")

# With live price monitoring
toolkit = AssistantHubToolkit(
    api_key="...",
    enable_price_feed=True,
    price_feed_coins=["BTC", "ETH", "SOL"],
)
```

## Anonymous Free Tier

No API key? You get **10 free calls per day** on non-premium tools:

```python
toolkit = AssistantHubToolkit()  # no api_key needed
tools = toolkit.get_tools()     # 10 calls/day, then 429 with upgrade CTA
```

## Payment

Free tools work instantly. Premium tools require:
- **Pro/Premium JWT** — unlimited calls
- **x402 USDC on Base** — pay per call (no subscription needed)
- **HUB stakers** — 50% off all x402 calls

## Package Structure

```
sdk/langchain-python/
  src/langchain_assistanthub/
    __init__.py          # Public API + __all__
    _version.py          # Single source of truth for version
    toolkit.py           # AssistantHubToolkit (main entry)
    client.py            # AssistantHubMCPClient (MCP adapter)
    tools.py             # Individual tool wrappers
    price_feed.py        # WebSocket price feed
    price_monitor.py     # Price alert monitor
    strategy.py          # Strategy analysis tool
    execution.py         # Trade execution tools
  tests/
    test_toolkit.py      # Comprehensive test suite
  pyproject.toml         # hatchling build config
```

## Example Notebooks

See [`sdk/notebooks/`](https://github.com/redman4220/assistant-hub/tree/master/sdk/notebooks) for runnable examples:

| Notebook | Description | Difficulty |
|----------|-------------|------------|
| `01_sentiment_trader.py` | ReAct agent: sentiment → news → risk → trade | Beginner |
| `02_backtest_optimizer.py` | Grid search 4 strategies × 3 coins | Intermediate |
| `03_live_monitor.py` | WebSocket feed + auto AI analysis on big moves | Advanced |

## Telemetry

This package sends a single anonymous ping on toolkit init to help us gauge adoption.
No PII is collected — just a random ID, package version, and auth type (keyed vs anonymous).

**Opt out:**

```bash
export ASSISTANT_HUB_TELEMETRY_OPT_OUT=1
```

## Links

- [Documentation](https://rmassistanthub.io/docs#langchain)
- [MCP Discovery](https://rmassistanthub.io/.well-known/mcp.json)
- [GitHub](https://github.com/redman4220/assistant-hub)
