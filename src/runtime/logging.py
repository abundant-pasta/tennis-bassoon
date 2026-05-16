"""Structured JSON logging helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json


def log_event(stage: str, message: str, **fields) -> None:
    payload = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "message": message,
        **fields,
    }
    print(json.dumps(payload, default=str))

