"""
Exception hierarchy for Assistant Hub toolkit.

Provides actionable error messages with upgrade CTAs so agents and
notebooks surface clear paths to resolution.
"""

from __future__ import annotations


class AssistantHubError(Exception):
    """Base error for all Assistant Hub toolkit exceptions."""

    def __init__(self, message: str = "An error occurred with Assistant Hub."):
        super().__init__(message)
        self.detail = message


class AssistantHubRateLimitError(AssistantHubError):
    """Raised when the Hub API returns 429 (rate limit exceeded)."""

    def __init__(self, detail: str = "Free tier: 10 calls/day limit reached."):
        super().__init__(
            f"{detail} "
            "Upgrade to Pro ($1/mo) or stake HUB for 50% off: "
            "https://rmassistanthub.io/#payments"
        )
        self.detail = detail


class AssistantHubPaymentRequiredError(AssistantHubError):
    """Raised when the Hub API returns 402 (payment required for premium tool)."""

    def __init__(self, detail: str = "Payment required for this premium tool."):
        super().__init__(
            f"{detail} Use x402 USDC on Base or upgrade tier: https://rmassistanthub.io/docs#x402"
        )
        self.detail = detail


class AssistantHubForbiddenError(AssistantHubError):
    """Raised when the Hub API returns 403 (access forbidden)."""

    def __init__(self, detail: str = "Access forbidden — check tier or auth."):
        super().__init__(f"{detail} Login or upgrade: https://rmassistanthub.io/#payments")
        self.detail = detail


class AssistantHubServerError(AssistantHubError):
    """Raised when the Hub API returns 5xx (server error)."""

    def __init__(self, detail: str = "Server error — try again later."):
        super().__init__(
            f"{detail} "
            "If this persists, report at: "
            "https://github.com/redman4220/langchain-assistanthub/issues"
        )
        self.detail = detail
