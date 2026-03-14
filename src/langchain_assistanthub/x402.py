"""
x402 Auto-Payment Module — USDC micropayments for premium tools.

When a tool call returns HTTP 402, this module can automatically
pay via USDC on Base and retry the request.

Two payment modes:
  1. BANKR API — uses BANKR agent wallet (server-side, no private key needed)
  2. Custom signer — you provide a callable that returns a tx hash

Example (BANKR):
    from langchain_assistanthub import AssistantHubToolkit
    from langchain_assistanthub.x402 import X402Config

    toolkit = AssistantHubToolkit(
        api_key="ahk_abc123",
        x402=X402Config(bankr_api_key="your-bankr-key"),
    )
    # Premium tools now auto-pay — no 402 errors!

Example (custom signer):
    toolkit = AssistantHubToolkit(
        api_key="ahk_abc123",
        x402=X402Config(
            signer=lambda payment: send_usdc(payment.recipient_address, payment.amount_usdc),
        ),
    )
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Union


# ── Types ─────────────────────────────────────────────────────────


@dataclass
class X402PaymentRequest:
    """Payment details extracted from a 402 response."""

    amount_usdc: float = 0.01
    recipient_address: str = "0xb21bed8c8338b943912f3c2fc2a84c9b883a3776"
    chain: str = "base"
    tool_id: str = "unknown"
    description: str = ""


@dataclass
class X402PaymentReceipt:
    """Receipt after a successful payment."""

    tx_hash: str = ""
    amount_usdc: float = 0.0
    chain: str = "base"


# Signer type: sync or async callable that takes X402PaymentRequest and returns tx hash
X402Signer = Callable[[X402PaymentRequest], Union[str, Awaitable[str]]]


@dataclass
class X402Config:
    """x402 auto-payment configuration.

    Args:
        bankr_api_key: BANKR API key for agent-managed wallet payments.
        signer: Custom callable that accepts X402PaymentRequest and returns a tx hash.
                Takes priority over bankr_api_key.
        max_per_call_usdc: Maximum USDC to auto-pay per call (default: 0.10). Safety cap.
        max_per_session_usdc: Maximum USDC to auto-pay per session (default: 1.00). Safety cap.
        verbose: If True, log payment details to console (default: False).
    """

    bankr_api_key: Optional[str] = None
    signer: Optional[X402Signer] = None
    max_per_call_usdc: float = 0.10
    max_per_session_usdc: float = 1.00
    verbose: bool = False


# ── Constants ─────────────────────────────────────────────────────

PAYMENT_WALLET = "0xb21bed8c8338b943912f3c2fc2a84c9b883a3776"
BANKR_API_URL = "https://api.bankr.bot"
POLL_INTERVAL = 2.0  # seconds
MAX_POLLS = 30


# ── Payment Handler ──────────────────────────────────────────────


class X402PaymentHandler:
    """Handles x402 auto-payments for premium tool calls.

    Supports two modes:
    - BANKR agent wallet (just provide an API key)
    - Custom signer (provide a callable)

    Includes safety caps and session spend tracking.
    """

    def __init__(self, config: X402Config) -> None:
        self._config = config
        self._session_spent: float = 0.0

    @property
    def is_configured(self) -> bool:
        """True if either a signer or BANKR API key is set."""
        return bool(self._config.signer or self._config.bankr_api_key)

    @property
    def spent(self) -> float:
        """Total USDC spent this session."""
        return self._session_spent

    def reset_session(self) -> None:
        """Reset the session spending counter."""
        self._session_spent = 0.0

    def parse_payment_request(
        self,
        headers: Dict[str, str],
        body: Dict[str, Any],
        tool_id: str = "unknown",
    ) -> X402PaymentRequest:
        """Extract payment details from a 402 response."""
        return X402PaymentRequest(
            amount_usdc=float(
                headers.get("X-Payment-Amount")
                or body.get("x402_amount")
                or body.get("price_usdc")
                or 0.01
            ),
            recipient_address=str(
                headers.get("X-Payment-Address")
                or body.get("x402_address")
                or PAYMENT_WALLET
            ),
            chain=str(
                headers.get("X-Payment-Chain")
                or body.get("x402_chain")
                or "base"
            ),
            tool_id=tool_id,
            description=f"x402 payment for {tool_id}",
        )

    async def pay(self, payment: X402PaymentRequest) -> X402PaymentReceipt:
        """Execute payment and return a receipt.

        Uses custom signer if provided, otherwise falls back to BANKR.

        Raises:
            ValueError: If payment exceeds safety caps.
            RuntimeError: If payment fails.
        """
        # Safety checks
        if payment.amount_usdc > self._config.max_per_call_usdc:
            raise ValueError(
                f"x402: Payment ${payment.amount_usdc} exceeds max_per_call_usdc "
                f"(${self._config.max_per_call_usdc}). "
                f"Increase limit or pay manually."
            )

        if self._session_spent + payment.amount_usdc > self._config.max_per_session_usdc:
            raise ValueError(
                f"x402: Session spending would reach "
                f"${self._session_spent + payment.amount_usdc:.4f} "
                f"(limit: ${self._config.max_per_session_usdc}). "
                f"Increase max_per_session_usdc or start a new session."
            )

        if self._config.verbose:
            print(
                f"[x402] Paying ${payment.amount_usdc} USDC "
                f"for {payment.tool_id} → {payment.recipient_address}"
            )

        # Execute payment
        if self._config.signer:
            result = self._config.signer(payment)
            # Support both sync and async signers
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                tx_hash = await result
            else:
                tx_hash = result  # type: ignore[assignment]
        elif self._config.bankr_api_key:
            tx_hash = await self._pay_via_bankr(payment)
        else:
            raise RuntimeError("x402: No signer or bankr_api_key configured.")

        self._session_spent += payment.amount_usdc

        if self._config.verbose:
            print(
                f"[x402] Payment confirmed: {tx_hash} "
                f"(${self._session_spent:.4f} spent this session)"
            )

        return X402PaymentReceipt(
            tx_hash=str(tx_hash),
            amount_usdc=payment.amount_usdc,
            chain=payment.chain,
        )

    # ── BANKR payment flow ────────────────────────────────────────

    async def _pay_via_bankr(self, payment: X402PaymentRequest) -> str:
        """Pay via BANKR agent API (prompt → poll → get tx hash)."""
        import aiohttp

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self._config.bankr_api_key or "",
        }

        prompt = (
            f"Send {payment.amount_usdc} USDC to {payment.recipient_address} "
            f"on Base. This is an x402 micropayment for tool access."
        )

        # Step 1: Submit prompt
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BANKR_API_URL}/agent/prompt",
                headers=headers,
                json={"prompt": prompt},
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"x402 BANKR prompt failed ({resp.status}): {text}"
                    )
                data = await resp.json()
                job_id = data["jobId"]

            # Step 2: Poll for completion
            for _ in range(MAX_POLLS):
                async with session.get(
                    f"{BANKR_API_URL}/agent/job/{job_id}",
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(
                            f"x402 BANKR poll failed ({resp.status})"
                        )
                    job = await resp.json()

                    if job["status"] == "completed":
                        txs = job.get("transactions", [])
                        if txs:
                            tx_hash = txs[0].get("metadata", {}).get("hash")
                            if tx_hash:
                                return str(tx_hash)
                        raise RuntimeError(
                            "x402 BANKR: Payment completed but no tx hash returned."
                        )

                    if job["status"] in ("failed", "cancelled"):
                        raise RuntimeError(
                            f"x402 BANKR payment {job['status']}: "
                            f"{job.get('error', 'unknown error')}"
                        )

                await asyncio.sleep(POLL_INTERVAL)

            raise RuntimeError("x402 BANKR: Payment timed out (60s).")
