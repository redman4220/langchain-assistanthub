"""
langchain-assistanthub — Crypto Intelligence Toolkit for LangChain Agents

The easiest way to add real-time crypto intelligence to any LangChain /
LangGraph agent. Connects to Assistant Hub via MCP (Model Context Protocol)
and exposes tools with human-readable names.

Quick start:
    from langchain_assistanthub import AssistantHubToolkit

    toolkit = AssistantHubToolkit(api_key="your-hub-api-key")
    tools = toolkit.get_tools()
    # → [AssistantHubLivePrices, AssistantHubFearGreed, AssistantHubRiskScores, ...]

    # Plug into any LangGraph agent:
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(model, tools)

Payment:
    Free tools work instantly. Premium tools (ai_forecast, run_backtest, etc.)
    require a Pro/Premium JWT or pay-per-call with USDC on Base via x402.

    HUB token stakers get 50% off all x402 calls.
"""

from langchain_assistanthub._version import __version__  # noqa: F401
from langchain_assistanthub.client import AssistantHubMCPClient
from langchain_assistanthub.execution import (
    AssistantHubCheckApproval,
    AssistantHubExecuteTrade,
)
from langchain_assistanthub.price_feed import (
    PriceBuffer,
    PriceFeedCallbackHandler,
    PriceFeedRunnable,
)
from langchain_assistanthub.price_monitor import AssistantHubPriceMonitor
from langchain_assistanthub.strategy import (
    AssistantHubStrategyAnalysis,
    StrategyAnalysisResult,
)
from langchain_assistanthub.toolkit import AssistantHubToolkit
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

__all__ = [
    "AssistantHubToolkit",
    "AssistantHubMCPClient",
    # Core tools
    "AssistantHubLivePrices",
    "AssistantHubFearGreed",
    "AssistantHubCryptoNews",
    "AssistantHubRiskScores",
    "AssistantHubDailyPulse",
    "AssistantHubAIForecast",
    "AssistantHubMonteCarloBacktest",
    "AssistantHubSlippageEstimate",
    "AssistantHubCreateAlert",
    # Price feed (Feature 1)
    "PriceBuffer",
    "PriceFeedRunnable",
    "PriceFeedCallbackHandler",
    "AssistantHubPriceMonitor",
    # Strategy lab (Feature 2)
    "AssistantHubStrategyAnalysis",
    "StrategyAnalysisResult",
    # Execution (Feature 3)
    "AssistantHubExecuteTrade",
    "AssistantHubCheckApproval",
]
