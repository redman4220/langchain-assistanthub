"""
Strategy Lab — Structured Tool Chain for LangChain agents.

Chains /backtest/analyze → Monte Carlo → walk-forward → slippage
and returns strongly-typed structured output with a clear
robust / marginal / overfit verdict.

Server API consumed:
  POST /backtest/analyze  (combo endpoint)
  GET  /backtest/slippage-estimate

Usage:
    toolkit = AssistantHubToolkit(api_key="...")
    tools = toolkit.get_tools()
    # AssistantHubStrategyAnalysis is included in premium tools
"""

from __future__ import annotations

import json
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from langchain_assistanthub.tools import AssistantHubBaseTool


# ── Output Models ─────────────────────────────────────────────


class BacktestSummary(BaseModel):
    total_return_pct: float = Field(description="Cumulative return %")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    max_drawdown_pct: float = Field(description="Maximum drawdown %")
    win_rate: float = Field(description="Win rate (0-1)")
    total_trades: int = Field(description="Number of trades executed")


class MonteCarloSummary(BaseModel):
    simulations: int = Field(description="Number of Monte Carlo runs")
    median_return_pct: float = Field(description="Median simulated return %")
    prob_profit: float = Field(description="Probability of profit (0-1)")
    var_95: float = Field(description="95% Value at Risk %")
    best_case_pct: float = Field(description="Best-case return %")
    worst_case_pct: float = Field(description="Worst-case return %")


class WalkForwardSummary(BaseModel):
    assessment: str = Field(description="'robust', 'marginal', or 'overfit'")
    in_sample_return: float = Field(description="In-sample return %")
    out_of_sample_return: float = Field(description="Out-of-sample return %")
    degradation_ratio: float = Field(description="OOS / IS return ratio (>0.5 = robust)")


class SlippageEstimate(BaseModel):
    slippage_pct: float = Field(description="Estimated slippage %")
    effective_price: float = Field(description="Effective execution price")
    price_impact: float = Field(description="Market impact %")
    fees_usd: float = Field(description="Estimated fees in USD")


class StrategyAnalysisResult(BaseModel):
    coin: str
    strategy: str
    period_days: int
    backtest: Optional[BacktestSummary] = None
    monte_carlo: Optional[MonteCarloSummary] = None
    walk_forward: Optional[WalkForwardSummary] = None
    slippage: Optional[SlippageEstimate] = None
    recommendation: str = Field(
        description="AI recommendation: deploy / paper-trade-more / avoid"
    )


# ── Input Model ───────────────────────────────────────────────


class StrategyAnalysisInput(BaseModel):
    """Input for end-to-end strategy analysis."""

    coin: str = Field(description="Cryptocurrency symbol (e.g., BTC, ETH, SOL)")
    strategy: str = Field(
        default="momentum",
        description="Strategy type: momentum, mean_reversion, breakout, rsi",
    )
    period_days: int = Field(
        default=90,
        description="Backtest period in days (30-365)",
    )
    capital: float = Field(
        default=10_000,
        description="Starting capital in USD for slippage sizing",
    )
    include_slippage: bool = Field(
        default=True,
        description="Whether to chain a slippage estimate for realistic sizing",
    )
    simulations: int = Field(
        default=1000,
        description="Number of Monte Carlo simulations (100-10000)",
    )
    venue: str = Field(
        default="binance",
        description="Exchange venue for slippage estimate",
    )


# ── Tool ──────────────────────────────────────────────────────


class AssistantHubStrategyAnalysis(AssistantHubBaseTool):
    """
    End-to-end strategy analysis: backtest → Monte Carlo → walk-forward
    → optional slippage estimate. Returns a structured verdict
    (robust / marginal / overfit) and recommendation.

    PREMIUM: Requires Pro/Premium tier or x402 USDC payment.
    """

    name: str = "assistant_hub_strategy_analysis"
    description: str = (
        "Run a full strategy analysis pipeline: backtest → Monte Carlo simulation "
        "→ walk-forward validation → slippage estimate. Returns a structured result "
        "with a robust/marginal/overfit verdict and deploy/paper-trade/avoid "
        "recommendation. Input: coin, strategy type, period, capital. PREMIUM."
    )
    hub_tool_id: str = "strategy_analysis"
    hub_endpoint: str = "/backtest/analyze"
    hub_method: str = "POST"
    hub_premium: bool = True
    args_schema: Type[BaseModel] = StrategyAnalysisInput

    async def _arun(
        self,
        coin: str = "BTC",
        strategy: str = "momentum",
        period_days: int = 90,
        capital: float = 10_000,
        include_slippage: bool = True,
        simulations: int = 1000,
        venue: str = "binance",
        **kwargs: Any,
    ) -> str:
        coin = coin.upper()

        # Step 1: Call /backtest/analyze (combo endpoint)
        analyze_payload = {
            "coin": coin,
            "strategy": strategy,
            "periodDays": period_days,
            "simulations": simulations,
            "walkForward": True,
        }
        raw = await self._hub_request(
            "/backtest/analyze", "POST", body=analyze_payload
        )

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "parse_error", "raw": str(raw)[:500]})

        if "error" in data:
            return json.dumps(data)

        # Parse backtest summary
        bt = data.get("backtest", {})
        backtest = BacktestSummary(
            total_return_pct=bt.get("totalReturnPct", 0),
            sharpe_ratio=bt.get("sharpeRatio", 0),
            max_drawdown_pct=bt.get("maxDrawdownPct", 0),
            win_rate=bt.get("winRate", 0),
            total_trades=bt.get("totalTrades", 0),
        )

        # Parse Monte Carlo
        mc = data.get("monteCarlo", {})
        monte_carlo = MonteCarloSummary(
            simulations=mc.get("simulations", simulations),
            median_return_pct=mc.get("medianReturnPct", 0),
            prob_profit=mc.get("probProfit", 0),
            var_95=mc.get("var95", 0),
            best_case_pct=mc.get("bestCasePct", 0),
            worst_case_pct=mc.get("worstCasePct", 0),
        )

        # Parse walk-forward
        wf = data.get("walkForward", {})
        walk_forward = WalkForwardSummary(
            assessment=wf.get("assessment", "marginal"),
            in_sample_return=wf.get("inSampleReturn", 0),
            out_of_sample_return=wf.get("outOfSampleReturn", 0),
            degradation_ratio=wf.get("degradationRatio", 0),
        )

        # Step 2: Optional slippage estimate
        slippage = None
        if include_slippage:
            slip_raw = await self._hub_request(
                "/backtest/slippage-estimate",
                "GET",
                params={
                    "coin": coin,
                    "amountUsd": str(capital),
                    "side": "buy",
                    "venue": venue,
                },
            )
            try:
                slip_data = json.loads(slip_raw)
                if "error" not in slip_data:
                    slippage = SlippageEstimate(
                        slippage_pct=slip_data.get("slippagePct", 0),
                        effective_price=slip_data.get("effectivePrice", 0),
                        price_impact=slip_data.get("priceImpact", 0),
                        fees_usd=slip_data.get("feesUsd", 0),
                    )
            except (json.JSONDecodeError, TypeError):
                pass  # Slippage is optional — don't fail the whole pipeline

        # Step 3: Generate recommendation
        recommendation = _generate_recommendation(
            walk_forward, monte_carlo, slippage
        )

        result = StrategyAnalysisResult(
            coin=coin,
            strategy=strategy,
            period_days=period_days,
            backtest=backtest,
            monte_carlo=monte_carlo,
            walk_forward=walk_forward,
            slippage=slippage,
            recommendation=recommendation,
        )

        return result.model_dump_json(indent=2)


def _generate_recommendation(
    wf: WalkForwardSummary,
    mc: MonteCarloSummary,
    slippage: Optional[SlippageEstimate],
) -> str:
    """Rule-based recommendation from analysis results."""
    score = 0

    # Walk-forward assessment (most important)
    if wf.assessment == "robust":
        score += 3
    elif wf.assessment == "marginal":
        score += 1
    else:  # overfit
        score -= 2

    # Monte Carlo probability of profit
    if mc.prob_profit >= 0.65:
        score += 2
    elif mc.prob_profit >= 0.50:
        score += 1
    else:
        score -= 1

    # Slippage impact
    if slippage:
        if slippage.slippage_pct > 2.0:
            score -= 1  # High slippage eats into returns

    if score >= 4:
        return "DEPLOY — Strategy is robust with high probability of profit."
    elif score >= 2:
        return "PAPER-TRADE — Strategy shows promise but needs more validation."
    else:
        return "AVOID — Strategy appears overfit or has low probability of profit."
