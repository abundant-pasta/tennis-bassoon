"""Simple webhook notifications for daily run summaries."""

from __future__ import annotations

import json
import urllib.request


def post_webhook(webhook_url: str, payload: dict) -> None:
    data = json.dumps(payload, default=str).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15):
        return
