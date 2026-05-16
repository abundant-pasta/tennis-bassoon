#!/usr/bin/env python3
"""
Daily shadow-run orchestrator for the tennis pipeline.

Bootstraps the schedule from The Odds API (since there is no separate schedule
feed), then drives tennis_daily_run in shadow or close_snapshot mode.

Usage:
    python3 scripts/run_shadow.py                      # morning shadow picks
    python3 scripts/run_shadow.py --mode close_snapshot # evening CLV snapshot
    python3 scripts/run_shadow.py --date 2026-05-01    # override date

The script:
  1. Reads ODDS_API_KEY from env or laughing-bassoon/.env
  2. Queries The Odds API for all known ATP sport keys (no pre-existing
     schedule required — override_sport_keys bypasses the schedule lookup)
  3. Writes today's matches to a temp schedule CSV
  4. Sets TENNIS_SCHEDULE_SOURCE_URI and other env vars
  5. Calls run_daily() directly
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config_tennis as cfg

# ── repo root on sys.path so imports work whether installed or not ─────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.tennis_odds_api import TOURNAMENTS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_ALL_SPORT_KEYS = ",".join(t.sport_key for t in TOURNAMENTS)
_SURFACE_BY_KEY = {t.sport_key: t.surface for t in TOURNAMENTS}
_TITLE_BY_KEY = {t.sport_key: t.title for t in TOURNAMENTS}

_LAUGHING_ENV = _ROOT.parent / "laughing-bassoon" / ".env"
_REMOTE_CONFIG = _ROOT.parent / "laughing-bassoon" / "remote_config.env"

_TA_RANKINGS_URL = "https://tennisabstract.com/reports/atpRankings.html"


def _read_env_file_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        left, right = line.split("=", 1)
        if left.strip() != key:
            continue
        return right.split("#", 1)[0].strip().strip("'\"")
    return None


def _resolve_api_key() -> str:
    for env_name in ("TENNIS_ODDS_API_KEY", "ODDS_API_KEY"):
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    for path in (_LAUGHING_ENV, _REMOTE_CONFIG):
        val = _read_env_file_value(path, "ODDS_API_KEY")
        if val:
            return val
    raise SystemExit("ERROR: No ODDS_API_KEY found. Set ODDS_API_KEY in env or laughing-bassoon/.env")


# ---------------------------------------------------------------------------
# ATP rankings from Tennis Abstract (free, updated weekly)
# ---------------------------------------------------------------------------

def _fetch_atp_rankings(run_date: str) -> dict[str, int]:
    """Scrape current ATP singles rankings from tennisabstract.com.

    Returns {lowercase_full_name: rank} for the top ~500 players.
    Cached per calendar day under data/tennis/ledger/.
    """
    import re as _re

    cache_path = Path(cfg.LEDGER_DIR) / f"atp_rankings_{run_date[:10]}.json"
    if cache_path.exists():
        import json as _json
        return _json.loads(cache_path.read_text())

    try:
        req = urllib.request.Request(
            _TA_RANKINGS_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; tennis-shadow-runner)"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"  WARNING: could not fetch ATP rankings from Tennis Abstract: {exc}", flush=True)
        return {}

    rows = _re.findall(r"<tr[^>]*>.*?</tr>", html, _re.DOTALL)
    ranks: dict[str, int] = {}
    for row in rows:
        cells = _re.findall(r"<td[^>]*>(.*?)</td>", row, _re.DOTALL)
        clean = [_re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip() for c in cells]
        if len(clean) >= 2 and clean[0].isdigit():
            ranks[clean[1].lower()] = int(clean[0])

    import json as _json
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(_json.dumps(ranks))
    print(f"  ATP rankings fetched: {len(ranks)} players → cached {cache_path.name}", flush=True)
    return ranks


def _resolve_rank(name: str, rankings: dict[str, int]) -> int | None:
    """Look up a player's rank by name, trying exact then surname match."""
    rank = rankings.get(name.lower())
    if rank is not None:
        return rank
    surname = name.split()[-1].lower() if name else ""
    matches = [v for k, v in rankings.items() if k.split()[-1] == surname]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# Schedule bootstrap from Odds API
# ---------------------------------------------------------------------------

def _fetch_today_schedule(api_key: str, run_date: str) -> pd.DataFrame:
    """
    Query all known ATP sport keys and return today's matches as a schedule
    DataFrame ready for the daily runner.
    """
    today = pd.Timestamp(run_date).normalize()
    tomorrow = today + pd.Timedelta(days=1)

    rows = []
    for meta in TOURNAMENTS:
        params = {
            "apiKey": api_key,
            "markets": "h2h",
            "oddsFormat": "decimal",
            "regions": "us",
        }
        url = f"{_ODDS_API_BASE}/sports/{meta.sport_key}/odds?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tennis-shadow-runner/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"  [{meta.sport_key}] fetch failed: {exc}", flush=True)
            continue

        for event in data:
            commence = event.get("commence_time", "")
            try:
                event_dt = pd.Timestamp(commence, tz="UTC").tz_convert("US/Eastern").normalize().tz_localize(None)
            except Exception:
                event_dt = pd.Timestamp(commence)
            # include matches today or tomorrow (covers late US time zones)
            if event_dt not in (today, tomorrow):
                continue

            player_name = event.get("home_team", "")
            opp_name = event.get("away_team", "")
            if not player_name or not opp_name:
                continue

            match_date = pd.Timestamp(commence, tz="UTC").isoformat()
            rows.append({
                "match_id": f"oddsapi_{event.get('id', '')}",
                "match_date": match_date,
                "tourney_name": meta.title,
                "surface": meta.surface,
                "tourney_level": "A",
                "round": "",
                "best_of": 3,
                "draw_size": 32,
                "player_name": player_name,
                "opp_name": opp_name,
                "player_rank": None,
                "opp_rank": None,
            })

    return pd.DataFrame(rows).drop_duplicates(subset=["match_id"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # config_tennis uses relative paths (data/tennis, runs/tennis); must run from project root
    os.chdir(_ROOT)

    parser = argparse.ArgumentParser(description="Tennis daily shadow run orchestrator.")
    parser.add_argument("--date", default=None, help="Run date YYYY-MM-DD (default: today)")
    parser.add_argument("--mode", default="shadow", choices=["shadow", "close_snapshot"])
    args = parser.parse_args()

    run_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Tennis shadow run: date={run_date}  mode={args.mode}", flush=True)

    api_key = _resolve_api_key()

    print("Fetching today's schedule from The Odds API...", flush=True)
    schedule_df = _fetch_today_schedule(api_key, run_date)

    if schedule_df.empty:
        print(f"No ATP matches found for {run_date}. Exiting cleanly.")
        sys.exit(0)

    print(f"Found {len(schedule_df)} matches: {', '.join(schedule_df['tourney_name'].unique())}", flush=True)

    # Populate player ranks from Tennis Abstract so governance and features work
    print("Fetching current ATP rankings from Tennis Abstract...", flush=True)
    rankings = _fetch_atp_rankings(run_date)
    if rankings:
        def _fill_rank(row: pd.Series, col: str) -> int | None:
            if pd.notna(row.get(col)):
                return row[col]
            return _resolve_rank(row["player_name"] if col == "player_rank" else row["opp_name"], rankings)
        schedule_df["player_rank"] = schedule_df.apply(lambda r: _fill_rank(r, "player_rank"), axis=1)
        schedule_df["opp_rank"] = schedule_df.apply(lambda r: _fill_rank(r, "opp_rank"), axis=1)
        filled = schedule_df[["player_name", "player_rank", "opp_name", "opp_rank"]].to_string(index=False)
        print(f"  Ranks populated:\n{filled}", flush=True)

    # Write schedule to a stable temp path (not /tmp — survives the process)
    sched_dir = Path(cfg.LEDGER_DIR) / "schedules"
    sched_dir.mkdir(parents=True, exist_ok=True)
    sched_path = sched_dir / f"schedule_{run_date}.csv"
    schedule_df.to_csv(sched_path, index=False)
    print(f"Schedule written → {sched_path}", flush=True)

    # Extend the historical match DB with recent 2025/2026 completed matches.
    # Prefer TennisMyLife because it carries the serve/return stat columns the
    # rolling features need; fall back to the older approximation if needed.
    # Cache is re-used within the same calendar day; rebuilt otherwise.
    ext_path = Path(cfg.LEDGER_DIR) / "extended_matches.parquet"
    ext_stamp = Path(cfg.LEDGER_DIR) / "extended_matches.date"
    today_stamp = run_date[:10]
    if ext_path.exists() and ext_stamp.exists() and ext_stamp.read_text().strip() == today_stamp:
        print(f"Extended match DB up to date (built today). Reusing {ext_path}", flush=True)
    else:
        print("Building extended match DB (recent 2025/2026 history)...", flush=True)
        from src.model.tennis_backtest_2026_ytd import build_extended_matches
        build_extended_matches(output_path=ext_path)
        ext_stamp.write_text(today_stamp)

    # Set env vars for the daily runner
    os.environ["TENNIS_SCHEDULE_SOURCE_URI"] = f"file://{sched_path}"
    os.environ["TENNIS_SOURCE_ADAPTER"] = "odds_api_tennis"
    os.environ["TENNIS_ODDS_API_SPORT_KEYS"] = _ALL_SPORT_KEYS
    os.environ["ODDS_API_KEY"] = api_key
    os.environ["TENNIS_MATCH_DB_PATH"] = str(ext_path)
    os.environ.setdefault("TENNIS_ENV", "shadow")
    # Relax freshness check — the schedule file we just wrote is fresh by definition
    os.environ.setdefault("TENNIS_SCHEDULE_MAX_AGE_HOURS", "48")
    os.environ.setdefault("TENNIS_ODDS_MAX_AGE_HOURS", "4")
    # Keep a modest coverage threshold because the recent feed can still lag a few days,
    # even though it is much richer than the older tennis-data approximation.
    os.environ.setdefault("TENNIS_MIN_FEATURE_COVERAGE", "0.40")

    # Import here so env vars are set before RuntimeConfig reads them
    from src.pipeline.tennis_daily_run import run_daily

    try:
        result = run_daily(run_date=run_date, mode=args.mode)
        print(json.dumps(result, indent=2, default=str))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
