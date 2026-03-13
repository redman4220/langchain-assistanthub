"""
Hybrid Execution Tool — Paper and live trade execution for LangChain agents.

Server APIs consumed:
  POST /api/paper/order      — Paper trades
  POST /api/agents/:id/execute — Live via durable executor (human approval >$100)
  GET  /api/agents/workflows/:id — Check approval status

Usage:
    toolkit = AssistantHubToolkit(api_key="...")
    tools = toolkit.get_tools()
    # AssistantHubExecuteTrade + AssistantHubCheckApproval included in premium
"""

from __future__ import annotations

import json
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from langchain_assistanthub.tools import AssistantHubBaseTool


# ── Input / Output Models ─────────────────────────────────────


class ExecutionInput(BaseModel):
    """Input for trade execution."""

    coin: str = Field(description="Cryptocurrency symbol (e.g., BTC, ETH, SOL)")
    action: str = Field(
        description="Trade action: 'buy' or 'sell'",
    )
    amount_usd: float = Field(
        description="Trade amount in USD",
    )
    mode: str = Field(
        default="paper",
        description="Execution mode: 'paper' (simulated) or 'live' (real, requires agent_id)",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in USD",
    )
    take_profit: Optional[float] = Field(
        default=None,
        description="Optional take-profit price in USD",
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Agent ID for live execution (required when mode='live')",
    )


class ApprovalCheckInput(BaseModel):
    """Input for checking workflow approval status."""

    workflow_id: str = Field(
        description="Workflow ID returned by a live execution that requires approval",
    )


# ── Execute Trade Tool ────────────────────────────────────────


class AssistantHubExecuteTrade(AssistantHubBaseTool):
    """
    Execute a trade in paper mode (simulated) or live mode (real, via durable executor).

    Paper mode: Instant execution, no real funds. Great for strategy testing.
    Live mode: Routes through durable executor with human approval gate for
    trades >$100. Returns workflow_id for approval tracking.

    PREMIUM: Requires Pro/Premium tier or x402 USDC payment.
    """

    name: str = "assistant_hub_execute_trade"
    description: str = (
        "Execute a cryptocurrency trade. Supports paper (simulated) or live mode. "
        "Paper trades execute instantly with no real funds — ideal for strategy testing. "
        "Live trades >$100 require human approval via the durable executor. "
        "Input: coin, action (buy/sell), amount_usd, mode (paper/live). PREMIUM."
    )
    hub_tool_id: str = "execute_trade"
    hub_endpoint: str = "/api/paper/order"
    hub_method: str = "POST"
    hub_premium: bool = True
    args_schema: Type[BaseModel] = ExecutionInput

    async def _arun(
        self,
        coin: str = "BTC",
        action: str = "buy",
        amount_usd: float = 100,
        mode: str = "paper",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        agent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        coin = coin.upper()
        action = action.lower()

        if action not in ("buy", "sell"):
            return json.dumps({
                "error": "invalid_action",
                "message": "Action must be 'buy' or 'sell'.",
            })

        if mode == "live":
            return await self._execute_live(
                coin, action, amount_usd, stop_loss, take_profit, agent_id
            )
        else:
            return await self._execute_paper(
                coin, action, amount_usd, stop_loss, take_profit
            )

    async def _execute_paper(
        self,
        coin: str,
        action: str,
        amount_usd: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> str:
        payload: dict[str, Any] = {
            "coin": coin,
            "side": action,
            "amountUsd": amount_usd,
            "source": "agent",
        }
        if stop_loss is not None:
            payload["stopLoss"] = stop_loss
        if take_profit is not None:
            payload["takeProfit"] = take_profit

        raw = await self._hub_request("/api/paper/order", "POST", body=payload)

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "parse_error", "raw": str(raw)[:500]})

        # Wrap in consistent output format
        if "error" in data:
            return json.dumps(data)

        return json.dumps({
            "success": True,
            "mode": "paper",
            "position_id": data.get("positionId", data.get("id")),
            "coin": coin,
            "action": action,
            "amount_usd": amount_usd,
            "entry_price": data.get("entryPrice"),
            "wallet_balance": data.get("walletBalance"),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }, indent=2)

    async def _execute_live(
        self,
        coin: str,
        action: str,
        amount_usd: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        agent_id: Optional[str],
    ) -> str:
        if not agent_id:
            return json.dumps({
                "error": "missing_agent_id",
                "message": "agent_id is required for live execution mode.",
            })

        payload: dict[str, Any] = {
            "action": "trade",
            "coin": coin,
            "side": action,
            "amountUsd": amount_usd,
        }
        if stop_loss is not None:
            payload["stopLoss"] = stop_loss
        if take_profit is not None:
            payload["takeProfit"] = take_profit

        raw = await self._hub_request(
            f"/api/agents/{agent_id}/execute", "POST", body=payload
        )

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "parse_error", "raw": str(raw)[:500]})

        if "error" in data:
            return json.dumps(data)

        result: dict[str, Any] = {
            "success": True,
            "mode": "live",
            "coin": coin,
            "action": action,
            "amount_usd": amount_usd,
        }

        # Check if human approval is required
        if data.get("approvalRequired") or data.get("status") == "pending_approval":
            result["approval_required"] = True
            result["workflow_id"] = data.get("workflowId", data.get("executionId"))
            result["message"] = (
                "Trade requires human approval (amount >$100). "
                "Use assistant_hub_check_approval to poll for approval status."
            )
        else:
            result["approval_required"] = False
            result["tx_hash"] = data.get("txHash")
            result["execution_id"] = data.get("executionId")
            result["pnl"] = data.get("pnl")

        return json.dumps(result, indent=2)


# ── Check Approval Tool ──────────────────────────────────────


class AssistantHubCheckApproval(AssistantHubBaseTool):
    """
    Check the approval status of a live trade workflow.

    After AssistantHubExecuteTrade returns approval_required=True,
    use this tool to poll for approval status.
    """

    name: str = "assistant_hub_check_approval"
    description: str = (
        "Check the approval status of a pending live trade. "
        "Input: workflow_id (from execute_trade result). "
        "Returns: status (pending/approved/rejected/expired), details. PREMIUM."
    )
    hub_tool_id: str = "check_approval"
    hub_endpoint: str = "/api/agents/workflows"
    hub_method: str = "GET"
    hub_premium: bool = True
    args_schema: Type[BaseModel] = ApprovalCheckInput

    async def _arun(
        self,
        workflow_id: str = "",
        **kwargs: Any,
    ) -> str:
        if not workflow_id:
            return json.dumps({
                "error": "missing_workflow_id",
                "message": "workflow_id is required.",
            })

        raw = await self._hub_request(
            f"/api/agents/workflows/{workflow_id}", "GET"
        )

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "parse_error", "raw": str(raw)[:500]})

        if "error" in data:
            return json.dumps(data)

        return json.dumps({
            "workflow_id": workflow_id,
            "status": data.get("status", "unknown"),
            "approved_by": data.get("approvedBy"),
            "approved_at": data.get("approvedAt"),
            "rejected_reason": data.get("rejectedReason"),
            "tx_hash": data.get("txHash"),
            "execution_result": data.get("result"),
        }, indent=2)
