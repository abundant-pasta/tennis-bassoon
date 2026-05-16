"""Load recent TennisMyLife yearly CSVs and normalize them to Sackmann-style rows."""

from __future__ import annotations

import io
import re
import urllib.request

import numpy as np
import pandas as pd

import config_tennis as cfg
from src.data.tennis_sackmann import _STAT_COLS, load_matches, load_players, to_player_perspective

RAW_DIR = cfg.RAW_DIR
RAW_DIR.mkdir(parents=True, exist_ok=True)

_TML_BASE = "https://stats.tennismylife.org/data"


def _canon(text: str) -> str:
    return re.sub(r"[^a-z]", "", str(text).lower())


def _player_lookup() -> tuple[dict[str, list[dict]], dict[int, int]]:
    players = load_players(verbose=False).copy()
    hist_counts = load_matches()["player_id"].value_counts().to_dict()
    players["full_key"] = (
        players["name_first"].fillna("").astype(str).map(_canon)
        + players["name_last"].fillna("").astype(str).map(_canon)
    )
    players["surname_key"] = players["name_last"].fillna("").astype(str).map(_canon)
    players["first_key"] = players["name_first"].fillna("").astype(str).map(_canon)
    lookup: dict[str, list[dict]] = {}
    for rec in players.to_dict("records"):
        lookup.setdefault(rec["full_key"], []).append(rec)
        lookup.setdefault(rec["surname_key"], []).append(rec)
    return lookup, hist_counts


def _resolve_player_ids(df: pd.DataFrame) -> pd.DataFrame:
    lookup, hist_counts = _player_lookup()
    synthetic_ids: dict[str, int] = {}

    def resolve(name: str) -> int:
        key = _canon(name)
        candidates = lookup.get(key, [])
        if not candidates:
            parts = str(name).split()
            if parts:
                candidates = lookup.get(_canon(parts[-1]), [])
        if candidates:
            candidates = sorted(
                candidates,
                key=lambda rec: hist_counts.get(int(rec["player_id"]), 0),
                reverse=True,
            )
            return int(candidates[0]["player_id"])
        if key not in synthetic_ids:
            synthetic_ids[key] = 900_000_000 + len(synthetic_ids) + 1
        return synthetic_ids[key]

    out = df.copy()
    out["winner_id"] = out["winner_name"].map(resolve)
    out["loser_id"] = out["loser_name"].map(resolve)
    return out


def _fetch_bytes(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except Exception:
        return None


def _cached_csv(url: str, local_name: str) -> pd.DataFrame | None:
    local_path = RAW_DIR / local_name
    if local_path.exists():
        try:
            return pd.read_csv(local_path, low_memory=False)
        except Exception:
            pass

    data = _fetch_bytes(url)
    if data is None:
        return None

    local_path.write_bytes(data)
    try:
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    except Exception:
        return None


def _normalize_matches(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    df = raw.copy()
    if df.empty:
        return pd.DataFrame()

    if "tourney_id" in df.columns and "match_num" in df.columns:
        df["match_id"] = df["tourney_id"].astype(str) + "_" + df["match_num"].astype(str)

    for col in [
        "winner_hand",
        "loser_hand",
        "winner_name",
        "loser_name",
        "surface",
        "tourney_name",
        "tourney_level",
        "round",
    ]:
        if col not in df.columns:
            df[col] = ""

    for col in _STAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "draw_size",
        "best_of",
        "minutes",
        "winner_ht",
        "winner_age",
        "winner_rank",
        "winner_rank_points",
        "loser_ht",
        "loser_age",
        "loser_rank",
        "loser_rank_points",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")

    df["score"] = df.get("score", "").fillna("").astype(str)
    df["is_walkover"] = df["score"].str.contains(r"W/O", case=False, na=False)
    df["is_retirement"] = df["score"].str.contains(r"RET", case=False, na=False)
    df["data_source"] = source
    return df


def load_tml_year(year: int, include_challenger: bool = True, verbose: bool = True) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    tour = _cached_csv(f"{_TML_BASE}/{year}.csv", f"tennis_tml_{year}.csv")
    if tour is not None and not tour.empty:
        frames.append(_normalize_matches(tour, source=f"tml_tour_{year}"))
        if verbose:
            print(f"  TML {year} tour: {len(tour)} matches")

    if include_challenger:
        chall = _cached_csv(
            f"{_TML_BASE}/{year}_challenger.csv",
            f"tennis_tml_{year}_challenger.csv",
        )
        if chall is not None and not chall.empty:
            frames.append(_normalize_matches(chall, source=f"tml_challenger_{year}"))
            if verbose:
                print(f"  TML {year} challenger: {len(chall)} matches")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df[df["tourney_date"].notna()].copy()
    df = df.drop_duplicates(subset=["tourney_id", "match_num"], keep="last")
    return df.reset_index(drop=True)


def load_tml_matches(
    years: list[int],
    include_challenger: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    frames = [
        load_tml_year(year, include_challenger=include_challenger, verbose=verbose)
        for year in years
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    return _resolve_player_ids(combined)


def load_tml_player_rows(
    years: list[int],
    include_challenger: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    matches = load_tml_matches(years, include_challenger=include_challenger, verbose=verbose)
    if matches.empty:
        return pd.DataFrame()
    return to_player_perspective(matches)
