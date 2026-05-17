#!/usr/bin/env python3
"""Run the Railway tennis shadow job and close snapshot on the same volume."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, time as dtime, timezone
from pathlib import Path


def _parse_close_time(raw: str) -> dtime:
    try:
        hour_text, minute_text = raw.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError as exc:
        raise ValueError(f"TENNIS_CLOSE_SNAPSHOT_UTC must be HH:MM, got {raw!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"TENNIS_CLOSE_SNAPSHOT_UTC must be HH:MM, got {raw!r}")
    return dtime(hour=hour, minute=minute, tzinfo=timezone.utc)


def _sleep_until_close(run_date: str, close_time: dtime) -> None:
    close_dt = datetime.fromisoformat(run_date).replace(
        hour=close_time.hour,
        minute=close_time.minute,
        second=0,
        microsecond=0,
        tzinfo=timezone.utc,
    )
    now = datetime.now(timezone.utc)
    sleep_seconds = max(0, int((close_dt - now).total_seconds()))
    if sleep_seconds <= 0:
        print(f"Close snapshot time {close_dt.isoformat()} has passed; running now.", flush=True)
        return
    print(
        f"Shadow run complete. Sleeping {sleep_seconds}s until close snapshot at "
        f"{close_dt.isoformat()}.",
        flush=True,
    )
    time.sleep(sleep_seconds)


def _run(args: list[str]) -> None:
    print(f"+ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    run_date = os.getenv("TENNIS_RUN_DATE") or datetime.now(timezone.utc).date().isoformat()
    close_time = _parse_close_time(os.getenv("TENNIS_CLOSE_SNAPSHOT_UTC", "03:00"))

    _run([sys.executable, str(repo_root / "scripts" / "prepare_runtime_data.py")])
    _run([sys.executable, str(repo_root / "scripts" / "run_shadow.py"), "--date", run_date])
    _sleep_until_close(run_date, close_time)
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_shadow.py"),
            "--date",
            run_date,
            "--mode",
            "close_snapshot",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
