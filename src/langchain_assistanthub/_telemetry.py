"""
Privacy-first anonymous telemetry for Assistant Hub toolkit.

Fires a single non-blocking POST on toolkit init to help gauge adoption.
No PII is collected — only a one-time random ID, package version, and
whether the user is authenticated.

Opt out by setting ASSISTANT_HUB_TELEMETRY_OPT_OUT=1 in your environment.
"""

from __future__ import annotations

import os
import threading

from langchain_assistanthub._version import __version__


def _send_telemetry(
    base_url: str,
    has_api_key: bool,
    event: str = "init",
) -> None:
    """Fire-and-forget telemetry ping. Never raises."""
    if os.environ.get("ASSISTANT_HUB_TELEMETRY_OPT_OUT"):
        return

    def _post() -> None:
        try:
            import json
            import urllib.request

            # One-time random ID — not tied to any user identity
            anon_id = os.urandom(8).hex()

            payload = json.dumps({
                "event": event,
                "version": __version__,
                "anon_id": anon_id,
                "auth_type": "api_key" if has_api_key else "anonymous",
            }).encode()

            url = f"{base_url}/api/telemetry/toolkit"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass  # Never fail the user's workflow

    threading.Thread(target=_post, daemon=True).start()
