"""
Download and normalise Jeff Sackmann's ATP CSV files into a clean match DB.

Produces two DataFrames:
  matches_df  — one row per match (canonical schema from the plan)
  rankings_df — weekly ATP rankings indexed by (player_id, ranking_date)
"""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config_tennis as cfg

RAW_DIR = cfg.RAW_DIR
RAW_DIR.mkdir(parents=True, exist_ok=True)

_ROUND_DAY_OFFSET = {
    "Q1": -2,
    "Q2": -1,
    "Q3": 0,
    "R128": 0,
    "R64": 1,
    "R32": 2,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
    "RR": 3,
    "BR": 1,
}
_ROUND_SORT_ORDER = {
    "Q1": -3,
    "Q2": -2,
    "Q3": -1,
    "R128": 0,
    "R64": 1,
    "R32": 2,
    "R16": 3,
    "QF": 4,
    "SF": 5,
    "F": 6,
    "RR": 2,
    "BR": 1,
}

_ATP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
_MATCH_FNAME = "atp_matches_{year}.csv"
_CHALL_FNAME = "atp_matches_qual_chall_{year}.csv"
_RANKINGS_FNAME = "atp_rankings_{suffix}.csv"
_RANKINGS_CURRENT = "atp_rankings_current.csv"
_PLAYERS_FNAME = "atp_players.csv"

_MATCH_YEARS_TOUR = list(range(1991, 2025))
_MATCH_YEARS_CHALL = list(range(2008, 2025))
_RANKING_SUFFIXES = [f"{d}s" for d in range(1973, 2020, 10)] + ["20s"]

_STAT_COLS = [
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]

_MATCH_COLS = [
    "match_id", "tourney_date", "tourney_name", "surface", "tourney_level",
    "draw_size", "round", "best_of",
    "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_age",
    "winner_rank", "winner_rank_points",
    "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_age",
    "loser_rank", "loser_rank_points",
    "score", "minutes",
    *_STAT_COLS,
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _fetch_url(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception:
        return None


def _cached_csv(url: str, local_path: Path) -> pd.DataFrame | None:
    if not local_path.exists():
        data = _fetch_url(url)
        if data is None:
            return None
        local_path.write_bytes(data)
    try:
        return pd.read_csv(local_path, low_memory=False)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Match ingestion
# ---------------------------------------------------------------------------

def _load_match_year(year: int, kind: str = "tour") -> pd.DataFrame:
    if kind == "tour":
        fname = _MATCH_FNAME.format(year=year)
    else:
        fname = _CHALL_FNAME.format(year=year)
    url = f"{_ATP_BASE}/{fname}"
    local = RAW_DIR / fname
    df = _cached_csv(url, local)
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def _normalize_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply schema, types, and derived columns to a raw Sackmann match CSV."""
    df = raw.copy()

    # match_id
    if "tourney_id" in df.columns and "match_num" in df.columns:
        df["match_id"] = df["tourney_id"].astype(str) + "_" + df["match_num"].astype(str)
    else:
        df["match_id"] = range(len(df))

    # tourney_date → date
    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")

    for col in _STAT_COLS:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
                "winner_age", "loser_age", "winner_ht", "loser_ht",
                "draw_size", "best_of", "minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out = pd.DataFrame()
    for col in _MATCH_COLS:
        out[col] = df.get(col, np.nan)

    return out


def load_all_matches(
    tour_years: list[int] | None = None,
    chall_years: list[int] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download and concatenate ATP tour + Challenger matches."""
    tour_years = tour_years or _MATCH_YEARS_TOUR
    chall_years = chall_years or _MATCH_YEARS_CHALL

    frames: list[pd.DataFrame] = []

    for year in tour_years:
        raw = _load_match_year(year, kind="tour")
        if not raw.empty:
            norm = _normalize_matches(raw)
            norm["data_source"] = "tour"
            frames.append(norm)
            if verbose:
                print(f"  tour {year}: {len(norm)} matches")

    for year in chall_years:
        raw = _load_match_year(year, kind="chall")
        if not raw.empty:
            norm = _normalize_matches(raw)
            norm["data_source"] = "chall"
            frames.append(norm)
            if verbose:
                print(f"  chall {year}: {len(norm)} matches")

    if not frames:
        raise RuntimeError("No match data downloaded — check network access.")

    df = pd.concat(frames, ignore_index=True)

    # Drop walkovers and retirements for training (but keep the flag for live filtering)
    df["score"] = df["score"].fillna("").astype(str)
    df["is_walkover"] = df["score"].str.contains(r"W/O", case=False, na=False)
    df["is_retirement"] = df["score"].str.contains(r"RET", case=False, na=False)

    df = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Player-perspective normalisation
# ---------------------------------------------------------------------------

def to_player_perspective(matches: pd.DataFrame) -> pd.DataFrame:
    """Expand one-row-per-match into two rows: winner perspective + loser perspective.

    Returns a DataFrame with columns:
      player_id, player_name, player_hand, player_ht, player_age,
      player_rank, player_rank_points,
      opp_id, opp_name, opp_hand, opp_ht, opp_age,
      opp_rank, opp_rank_points,
      won (1/0),
      serve stats from player perspective (ace, df, svpt, 1stIn, 1stWon, 2ndWon, ...),
      all match context cols
    """
    shared = ["match_id", "tourney_date", "tourney_name", "surface", "tourney_level",
              "draw_size", "round", "best_of", "score", "minutes", "is_walkover",
              "is_retirement", "data_source"]

    winner_rows = matches[shared + [
        "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_age",
        "winner_rank", "winner_rank_points",
        "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_age",
        "loser_rank", "loser_rank_points",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
    ]].copy()
    winner_rows["won"] = 1
    winner_rows = winner_rows.rename(columns={
        "winner_id": "player_id", "winner_name": "player_name",
        "winner_hand": "player_hand", "winner_ht": "player_ht",
        "winner_age": "player_age", "winner_rank": "player_rank",
        "winner_rank_points": "player_rank_points",
        "loser_id": "opp_id", "loser_name": "opp_name",
        "loser_hand": "opp_hand", "loser_ht": "opp_ht",
        "loser_age": "opp_age", "loser_rank": "opp_rank",
        "loser_rank_points": "opp_rank_points",
        "w_ace": "ace", "w_df": "df", "w_svpt": "svpt",
        "w_1stIn": "first_in", "w_1stWon": "first_won",
        "w_2ndWon": "second_won", "w_SvGms": "sv_gms",
        "w_bpSaved": "bp_saved", "w_bpFaced": "bp_faced",
        "l_ace": "opp_ace", "l_df": "opp_df", "l_svpt": "opp_svpt",
        "l_1stIn": "opp_first_in", "l_1stWon": "opp_first_won",
        "l_2ndWon": "opp_second_won", "l_SvGms": "opp_sv_gms",
        "l_bpSaved": "opp_bp_saved", "l_bpFaced": "opp_bp_faced",
    })

    loser_rows = matches[shared + [
        "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_age",
        "loser_rank", "loser_rank_points",
        "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_age",
        "winner_rank", "winner_rank_points",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
    ]].copy()
    loser_rows["won"] = 0
    loser_rows = loser_rows.rename(columns={
        "loser_id": "player_id", "loser_name": "player_name",
        "loser_hand": "player_hand", "loser_ht": "player_ht",
        "loser_age": "player_age", "loser_rank": "player_rank",
        "loser_rank_points": "player_rank_points",
        "winner_id": "opp_id", "winner_name": "opp_name",
        "winner_hand": "opp_hand", "winner_ht": "opp_ht",
        "winner_age": "opp_age", "winner_rank": "opp_rank",
        "winner_rank_points": "opp_rank_points",
        "l_ace": "ace", "l_df": "df", "l_svpt": "svpt",
        "l_1stIn": "first_in", "l_1stWon": "first_won",
        "l_2ndWon": "second_won", "l_SvGms": "sv_gms",
        "l_bpSaved": "bp_saved", "l_bpFaced": "bp_faced",
        "w_ace": "opp_ace", "w_df": "opp_df", "w_svpt": "opp_svpt",
        "w_1stIn": "opp_first_in", "w_1stWon": "opp_first_won",
        "w_2ndWon": "opp_second_won", "w_SvGms": "opp_sv_gms",
        "w_bpSaved": "opp_bp_saved", "w_bpFaced": "opp_bp_faced",
    })

    combined = pd.concat([winner_rows, loser_rows], ignore_index=True)
    combined["sim_date"] = (
        pd.to_datetime(combined["tourney_date"])
        + pd.to_timedelta(combined["round"].map(lambda r: _ROUND_DAY_OFFSET.get(str(r), 2)), unit="D")
    )
    combined["_round_order"] = combined["round"].map(lambda r: _ROUND_SORT_ORDER.get(str(r), 2))
    combined = combined.sort_values(["sim_date", "_round_order", "match_id", "won"],
                                   ascending=[True, True, True, False]).reset_index(drop=True)
    combined.drop(columns=["_round_order"], inplace=True)
    return combined


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def load_rankings(verbose: bool = True) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for suffix in _RANKING_SUFFIXES:
        fname = _RANKINGS_FNAME.format(suffix=suffix)
        url = f"{_ATP_BASE}/{fname}"
        local = RAW_DIR / fname
        df = _cached_csv(url, local)
        if df is not None and not df.empty:
            frames.append(df)
            if verbose:
                print(f"  rankings {suffix}: {len(df)} rows")

    # Current rankings file
    url = f"{_ATP_BASE}/{_RANKINGS_CURRENT}"
    local = RAW_DIR / _RANKINGS_CURRENT
    df = _cached_csv(url, local)
    if df is not None and not df.empty:
        frames.append(df)

    if not frames:
        raise RuntimeError("No ranking data downloaded.")

    ranks = pd.concat(frames, ignore_index=True)
    ranks.columns = [c.strip().lower() for c in ranks.columns]

    # Sackmann uses 'ranking_date', 'rank', 'player', 'points'
    col_map = {}
    if "player" in ranks.columns:
        col_map["player"] = "player_id"
    if "ranking_date" not in ranks.columns and "date" in ranks.columns:
        col_map["date"] = "ranking_date"
    ranks = ranks.rename(columns=col_map)

    ranks["ranking_date"] = pd.to_datetime(ranks["ranking_date"].astype(str), format="%Y%m%d", errors="coerce")
    ranks["rank"] = pd.to_numeric(ranks["rank"], errors="coerce")
    ranks["player_id"] = pd.to_numeric(ranks["player_id"], errors="coerce")
    ranks["points"] = pd.to_numeric(ranks.get("points", np.nan), errors="coerce")

    ranks = ranks.dropna(subset=["player_id", "ranking_date", "rank"])
    ranks = ranks.sort_values(["player_id", "ranking_date"]).reset_index(drop=True)
    return ranks


# ---------------------------------------------------------------------------
# Players metadata
# ---------------------------------------------------------------------------

def load_players(verbose: bool = True) -> pd.DataFrame:
    url = f"{_ATP_BASE}/{_PLAYERS_FNAME}"
    local = RAW_DIR / _PLAYERS_FNAME
    df = _cached_csv(url, local)
    if df is None:
        return pd.DataFrame()
    df.columns = [c.strip().lower() for c in df.columns]
    if verbose:
        print(f"  players: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_matches(matches: pd.DataFrame) -> None:
    out = cfg.FEATURES_DIR / "matches.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(out, index=False)
    print(f"Saved {len(matches)} player-perspective rows → {out}")


def load_matches() -> pd.DataFrame:
    path = cfg.FEATURES_DIR / "matches.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Match DB not found at {path}. Run tennis_sackmann.py first.")
    return pd.read_parquet(path)


def save_rankings(rankings: pd.DataFrame) -> None:
    out = cfg.FEATURES_DIR / "rankings.parquet"
    rankings.to_parquet(out, index=False)
    print(f"Saved {len(rankings)} ranking rows → {out}")


def load_rankings_cached() -> pd.DataFrame:
    path = cfg.FEATURES_DIR / "rankings.parquet"
    if not path.exists():
        raise FileNotFoundError("Rankings parquet not found. Run tennis_sackmann.py first.")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print("Downloading Sackmann ATP tour matches...")
    matches_raw = load_all_matches(verbose=True)
    print(f"\nTotal raw matches: {len(matches_raw)}")

    # Filter: skip Futures, walkovers, and retirements from training set
    matches_train = matches_raw[
        matches_raw["tourney_level"].isin(cfg.INCLUDE_TOURNEY_LEVELS)
    ].copy()
    print(f"After tourney level filter: {len(matches_train)}")

    print("\nConverting to player-perspective rows...")
    player_df = to_player_perspective(matches_train)
    print(f"Player-perspective rows: {len(player_df)}")

    save_matches(player_df)

    print("\nDownloading rankings...")
    rankings = load_rankings(verbose=True)
    save_rankings(rankings)
    print("Done.")


if __name__ == "__main__":
    main()
