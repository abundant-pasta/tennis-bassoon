from __future__ import annotations

import json
import importlib.util
from datetime import time as dtime, timezone
from pathlib import Path

from src.data.tennis_odds_api import TournamentMeta

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("run_shadow", _ROOT / "scripts" / "run_shadow.py")
assert _SPEC is not None
assert _SPEC.loader is not None
run_shadow = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_shadow)

_COMBINED_SPEC = importlib.util.spec_from_file_location(
    "run_shadow_and_close", _ROOT / "scripts" / "run_shadow_and_close.py"
)
assert _COMBINED_SPEC is not None
assert _COMBINED_SPEC.loader is not None
run_shadow_and_close = importlib.util.module_from_spec(_COMBINED_SPEC)
_COMBINED_SPEC.loader.exec_module(run_shadow_and_close)


class _FakeResponse:
    def __init__(self, payload: list[dict]):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_fetch_today_schedule_uses_utc_run_date_window(monkeypatch):
    monkeypatch.setattr(
        run_shadow,
        "TOURNAMENTS",
        (TournamentMeta("tennis_atp_test", "Test Open", "Hard", ("test open",)),),
    )

    payload = [
        {
            "id": "early-utc",
            "commence_time": "2026-05-17T00:30:00Z",
            "home_team": "Alpha One",
            "away_team": "Beta Two",
        },
        {
            "id": "late-utc",
            "commence_time": "2026-05-17T23:30:00Z",
            "home_team": "Gamma Three",
            "away_team": "Delta Four",
        },
        {
            "id": "next-day",
            "commence_time": "2026-05-18T00:15:00Z",
            "home_team": "Epsilon Five",
            "away_team": "Zeta Six",
        },
    ]

    def fake_urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(payload)

    monkeypatch.setattr(run_shadow.urllib.request, "urlopen", fake_urlopen)

    schedule = run_shadow._fetch_today_schedule("test-key", "2026-05-17")

    assert schedule["match_id"].tolist() == ["oddsapi_early-utc", "oddsapi_late-utc"]
    assert schedule["match_date"].tolist() == [
        "2026-05-17T00:30:00+00:00",
        "2026-05-17T23:30:00+00:00",
    ]


def test_extended_match_cache_accepts_recent_stamp(tmp_path, monkeypatch):
    cache = tmp_path / "extended_matches.parquet"
    stamp = tmp_path / "extended_matches.date"
    cache.write_bytes(b"parquet-ish")
    stamp.write_text("2026-05-15")
    monkeypatch.setenv("TENNIS_EXTENDED_MATCH_DB_MAX_AGE_DAYS", "7")

    assert run_shadow._extended_match_cache_is_usable(cache, stamp, "2026-05-17")


def test_extended_match_cache_rejects_stale_stamp(tmp_path, monkeypatch):
    cache = tmp_path / "extended_matches.parquet"
    stamp = tmp_path / "extended_matches.date"
    cache.write_bytes(b"parquet-ish")
    stamp.write_text("2026-05-01")
    monkeypatch.setenv("TENNIS_EXTENDED_MATCH_DB_MAX_AGE_DAYS", "7")

    assert not run_shadow._extended_match_cache_is_usable(cache, stamp, "2026-05-17")


def test_combined_runner_parses_close_snapshot_time():
    assert run_shadow_and_close._parse_close_time("03:00") == dtime(
        hour=3, minute=0, tzinfo=timezone.utc
    )
