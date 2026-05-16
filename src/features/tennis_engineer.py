"""
Feature engineering for the tennis predictor.

All features use only data available *before* the match being predicted.
Rolling stats use shift(1) before each rolling window to prevent leakage.

Feature set (all computed as player_value - opponent_value differentials):
  - Surface-specific ELO (elo_hard, elo_clay, elo_grass, elo_overall)
  - Serve/return rolling stats (20-match all-surface, 10-match same-surface)
  - Recent form (win pct last 5/10, sets win pct, deciding set win pct)
  - Head-to-head (Bayesian shrinkage applied)
  - Tournament/context features (rank, age, hand matchup, rest, congestion)
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

import config_tennis as cfg

FEATURES_DIR = cfg.FEATURES_DIR
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

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


def _estimate_match_date(tourney_date: pd.Series, round_: pd.Series) -> pd.Series:
    base = pd.to_datetime(tourney_date)
    offsets = round_.map(lambda r: _ROUND_DAY_OFFSET.get(str(r), 2))
    return base + pd.to_timedelta(offsets, unit="D")


def _ensure_sim_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sim_date" not in out.columns:
        out["sim_date"] = _estimate_match_date(out["tourney_date"], out["round"])
    else:
        out["sim_date"] = pd.to_datetime(out["sim_date"], utc=True).dt.tz_localize(None)
    return out


def _sort_match_time(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    out["_round_order"] = out["round"].map(lambda r: _ROUND_SORT_ORDER.get(str(r), 2))
    cols = ["sim_date", "_round_order", "match_id"]
    if extra:
        cols = extra + cols
    out = out.sort_values(cols).reset_index(drop=True)
    out.drop(columns=["_round_order"], inplace=True)
    return out


def _known_won(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------

def compute_elo_ratings(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute surface-specific ELO ratings for every player.

    Processes matches in chronological order. Pre-match ELO is recorded,
    then updated after the result is applied.

    Returns matches_df with added columns:
      player_elo_overall, player_elo_hard, player_elo_clay, player_elo_grass,
      opp_elo_overall, opp_elo_hard, opp_elo_clay, opp_elo_grass,
      elo_diff_overall, elo_diff_surface
    """
    df = _ensure_sim_date(matches_df)
    df = _sort_match_time(df)

    surfaces = ["overall", "hard", "clay", "grass"]

    elo_store: dict[int, dict[str, float]] = {}
    match_count: dict[int, int] = {}

    def _get_elo(pid: int) -> dict[str, float]:
        if pid not in elo_store:
            elo_store[pid] = {s: cfg.ELO_START for s in surfaces}
            match_count[pid] = 0
        return elo_store[pid]

    def _k(pid: int) -> float:
        n = match_count.get(pid, 0)
        return cfg.ELO_K_STABLE if n >= cfg.ELO_K_THRESHOLD else cfg.ELO_K_INITIAL

    def _surface_key(surface: str) -> str:
        s = str(surface).lower()
        if s in ("hard", "hardcourt"):
            return "hard"
        if s == "clay":
            return "clay"
        if s == "grass":
            return "grass"
        return "overall"

    player_elo_cols = {s: [] for s in surfaces}
    opp_elo_cols = {s: [] for s in surfaces}

    for _, row in df.iterrows():
        pid = int(row["player_id"]) if pd.notna(row["player_id"]) else -1
        oid = int(row["opp_id"]) if pd.notna(row["opp_id"]) else -2
        surf = _surface_key(row.get("surface", ""))
        won = _known_won(row["won"])

        p_elo = _get_elo(pid)
        o_elo = _get_elo(oid)

        for s in surfaces:
            player_elo_cols[s].append(p_elo[s])
            opp_elo_cols[s].append(o_elo[s])

        # Update after recording pre-match ELO. Future schedule rows can flow
        # through here with won missing; we record the state but do not update it.
        if won is None:
            continue

        # Update after recording pre-match ELO
        # surface-specific update
        exp_p = 1.0 / (1.0 + 10.0 ** ((o_elo[surf] - p_elo[surf]) / 400.0))
        exp_o = 1.0 - exp_p
        kp, ko = _k(pid), _k(oid)
        p_elo[surf] += kp * (won - exp_p)
        o_elo[surf] += ko * ((1 - won) - exp_o)

        # overall update
        exp_p_ov = 1.0 / (1.0 + 10.0 ** ((o_elo["overall"] - p_elo["overall"]) / 400.0))
        p_elo["overall"] += kp * (won - exp_p_ov)
        o_elo["overall"] += ko * ((1 - won) - (1 - exp_p_ov))

        match_count[pid] = match_count.get(pid, 0) + 1
        match_count[oid] = match_count.get(oid, 0) + 1

    for s in surfaces:
        df[f"player_elo_{s}"] = player_elo_cols[s]
        df[f"opp_elo_{s}"] = opp_elo_cols[s]
        df[f"elo_diff_{s}"] = df[f"player_elo_{s}"] - df[f"opp_elo_{s}"]

    # Surface-conditional ELO diff
    def _elo_diff_surface(row):
        sk = _surface_key(row.get("surface", ""))
        return row[f"player_elo_{sk}"] - row[f"opp_elo_{sk}"]

    df["elo_diff_surface"] = df.apply(_elo_diff_surface, axis=1)
    return df


# ---------------------------------------------------------------------------
# Serve / return rolling stats
# ---------------------------------------------------------------------------

def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num.where(denom > 0, np.nan) / denom.where(denom > 0, np.nan)


def _compute_serve_stats(grp: pd.DataFrame) -> pd.DataFrame:
    """Add match-level serve/return stat columns needed for rolling."""
    g = grp.copy()
    g["spw"] = _safe_div(g["first_won"] + g["second_won"], g["svpt"])
    g["rpw"] = 1.0 - _safe_div(g["opp_first_won"] + g["opp_second_won"], g["opp_svpt"])
    g["ace_rate"] = _safe_div(g["ace"], g["svpt"])
    g["df_rate"] = _safe_div(g["df"], g["svpt"])
    g["first_serve_pct"] = _safe_div(g["first_in"], g["svpt"])
    g["first_serve_win_pct"] = _safe_div(g["first_won"], g["first_in"])
    second_in = (g["svpt"] - g["first_in"]).clip(lower=0)
    g["second_serve_win_pct"] = _safe_div(g["second_won"], second_in)
    g["bp_save_pct"] = _safe_div(g["bp_saved"], g["bp_faced"])
    opp_bp_conv = g["opp_bp_faced"] - g["opp_bp_saved"]
    g["bp_conversion_pct"] = _safe_div(opp_bp_conv, g["opp_bp_faced"])
    return g


_SERVE_STAT_COLS = [
    "spw", "rpw", "ace_rate", "df_rate", "first_serve_pct",
    "first_serve_win_pct", "second_serve_win_pct", "bp_save_pct", "bp_conversion_pct",
]


def _rolling_serve_stats(grp: pd.DataFrame, window: int, suffix: str) -> pd.DataFrame:
    """Compute rolling means for serve stats with shift(1) to prevent leakage."""
    g = _sort_match_time(_ensure_sim_date(grp))
    for col in _SERVE_STAT_COLS:
        if col in g.columns:
            rolled = g[col].shift(1).rolling(window, min_periods=max(1, window // 2)).mean()
            g[f"{col}_{suffix}"] = rolled
    return g


def _rolling_surface_stats(grp: pd.DataFrame, window: int, suffix: str) -> pd.DataFrame:
    """Compute rolling means restricted to the current row's surface."""
    g = _sort_match_time(_ensure_sim_date(grp))
    surface_key = g["surface"].fillna("").astype(str).str.lower()

    for col in _SERVE_STAT_COLS:
        if col not in g.columns:
            continue
        rolled = (
            g.groupby(surface_key, dropna=False)[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
        )
        g[f"{col}_{suffix}"] = rolled

    return g


# ---------------------------------------------------------------------------
# Recent form
# ---------------------------------------------------------------------------

def _rolling_form(grp: pd.DataFrame) -> pd.DataFrame:
    g = _sort_match_time(_ensure_sim_date(grp))
    won_s = g["won"].astype(float).shift(1)

    g["win_pct_last_10"] = won_s.rolling(10, min_periods=1).mean()
    g["win_pct_last_5"] = won_s.rolling(5, min_periods=1).mean()

    # sets win pct (approximate from score)
    def _sets_win_pct(sub_df: pd.DataFrame) -> pd.Series:
        scores = sub_df["score"].fillna("").astype(str)
        set_wins, set_total = [], []
        for sc in scores:
            w, t = 0, 0
            for s in sc.split():
                parts = re.split(r"[-(\[\(]", s)
                if len(parts) >= 2:
                    try:
                        pw = int(parts[0])
                        pl = int(parts[1][:1])
                        t += 1
                        if pw > pl:
                            w += 1
                    except (ValueError, IndexError):
                        pass
            set_wins.append(w)
            set_total.append(t)
        return pd.Series(np.where(np.array(set_total) > 0,
                                  np.array(set_wins) / np.array(set_total), np.nan),
                         index=sub_df.index)

    with np.errstate(invalid="ignore"):
        g["set_win_pct_match"] = _sets_win_pct(g)
    set_s = g["set_win_pct_match"].shift(1)
    g["sets_win_pct_last_10"] = set_s.rolling(10, min_periods=3).mean()

    # deciding set win pct: matches that went to the deciding set
    def _is_deciding(score: str, best_of: int) -> tuple[bool, bool]:
        sets = [s for s in str(score).split() if "-" in s and not s.startswith("(")]
        target = int(best_of) // 2 + 1 if pd.notna(best_of) else 2
        return len(sets) >= (target * 2 - 1), len(sets) >= (target * 2 - 1)

    g["went_deciding"] = g.apply(
        lambda r: str(r.get("score", "")).count("-") >= 2 and
        ((int(r.get("best_of", 3)) == 3 and str(r.get("score", "")).count(" ") >= 2) or
         (int(r.get("best_of", 5)) == 5 and str(r.get("score", "")).count(" ") >= 4)),
        axis=1
    )
    deciding_won = (g["won"] * g["went_deciding"]).astype(float).shift(1)
    deciding_total = g["went_deciding"].astype(float).shift(1)
    raw_dw = deciding_won.rolling(20, min_periods=3).sum()
    raw_dt = deciding_total.rolling(20, min_periods=3).sum()
    g["deciding_set_win_pct"] = raw_dw / raw_dt.where(raw_dt > 0, np.nan)

    return g

# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------

def compute_h2h(player_df: pd.DataFrame) -> pd.DataFrame:
    """Add H2H features for each match using Bayesian shrinkage."""
    df = _sort_match_time(_ensure_sim_date(player_df))
    shrink_k = cfg.H2H_SHRINK_K

    h2h_wins, h2h_losses, h2h_n = [], [], []
    h2h_win_pct, h2h_win_pct_adj = [], []
    h2h_wins_surface, h2h_n_surface, h2h_win_pct_surface = [], [], []
    h2h_wins_recent, h2h_n_recent, h2h_win_pct_recent = [], [], []

    history: dict[tuple[int, int], list[tuple]] = {}

    for _, row in df.iterrows():
        pid = int(row["player_id"]) if pd.notna(row["player_id"]) else -1
        oid = int(row["opp_id"]) if pd.notna(row["opp_id"]) else -2
        surf = str(row.get("surface", "")).lower()
        date = row["tourney_date"]
        won = _known_won(row["won"])

        key = (min(pid, oid), max(pid, oid))
        past = history.get(key, [])

        wins = sum(1 for p, o, w, s, d in past if p == pid and w == 1)
        losses = sum(1 for p, o, w, s, d in past if p == pid and w == 0)
        n = wins + losses

        w_surf = sum(1 for p, o, w, s, d in past if p == pid and w == 1 and s == surf)
        n_surf = sum(1 for p, o, w, s, d in past if p == pid and s == surf)

        _date_naive = pd.Timestamp(date)
        if _date_naive.tzinfo is not None:
            _date_naive = _date_naive.tz_localize(None)
        three_yrs_ago = _date_naive - pd.DateOffset(years=3)
        w_rec = sum(1 for p, o, w, s, d in past if p == pid and w == 1 and d >= three_yrs_ago)
        n_rec = sum(1 for p, o, w, s, d in past if p == pid and d >= three_yrs_ago)

        h2h_wins.append(wins)
        h2h_losses.append(losses)
        h2h_n.append(n)
        h2h_win_pct.append(wins / n if n > 0 else 0.5)
        adj = (wins + shrink_k / 2) / (n + shrink_k)
        h2h_win_pct_adj.append(adj)

        h2h_wins_surface.append(w_surf)
        h2h_n_surface.append(n_surf)
        h2h_win_pct_surface.append(w_surf / n_surf if n_surf > 0 else 0.5)

        h2h_wins_recent.append(w_rec)
        h2h_n_recent.append(n_rec)
        h2h_win_pct_recent.append(w_rec / n_rec if n_rec > 0 else 0.5)

        if won is not None:
            history.setdefault(key, []).append((pid, oid, won, surf, pd.Timestamp(date)))

    df["h2h_wins"] = h2h_wins
    df["h2h_losses"] = h2h_losses
    df["h2h_n"] = h2h_n
    df["h2h_win_pct"] = h2h_win_pct
    df["h2h_win_pct_adj"] = h2h_win_pct_adj
    df["h2h_n_surface"] = h2h_n_surface
    df["h2h_win_pct_surface"] = h2h_win_pct_surface
    df["h2h_n_recent"] = h2h_n_recent
    df["h2h_win_pct_recent"] = h2h_win_pct_recent
    return df


# ---------------------------------------------------------------------------
# Context features
# ---------------------------------------------------------------------------

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_sim_date(df)
    df = df.sort_values(["player_id", "sim_date", "round", "match_id"])

    # Days since last match / matches in last 14 days (per player)
    days_since, matches_14 = [], []
    last_match_date: dict[int, pd.Timestamp] = {}
    recent_matches: dict[int, list[pd.Timestamp]] = {}

    for _, row in df.iterrows():
        pid = int(row["player_id"]) if pd.notna(row["player_id"]) else -1
        date = pd.Timestamp(row["sim_date"])
        last = last_match_date.get(pid)
        days_since.append((date - last).days if last is not None else np.nan)

        past = [d for d in recent_matches.get(pid, []) if (date - d).days <= 14]
        matches_14.append(len(past))

        last_match_date[pid] = date
        recent_matches.setdefault(pid, []).append(date)

    df["days_since_last_match"] = days_since
    df["matches_last_14_days"] = matches_14

    # Encoded features
    df["tourney_level_enc"] = df["tourney_level"].map(cfg.TOURNEY_LEVEL_ENCODE).fillna(1)
    df["round_enc"] = df["round"].map(cfg.ROUND_ENCODE).fillna(2)
    df["surface_enc"] = df["surface"].map(cfg.SURFACE_ENCODE).fillna(0)

    df["player_rank_log"] = np.log1p(df["player_rank"].fillna(500))
    df["opp_rank_log"] = np.log1p(df["opp_rank"].fillna(500))
    df["rank_diff"] = df["player_rank"].fillna(500) - df["opp_rank"].fillna(500)
    df["rank_diff_log"] = df["player_rank_log"] - df["opp_rank_log"]

    df["age_diff"] = df["player_age"].fillna(25) - df["opp_age"].fillna(25)

    # Hand matchup
    def _hand_matchup(ph, oh):
        ph = str(ph)[:1].upper() if pd.notna(ph) else "U"
        oh = str(oh)[:1].upper() if pd.notna(oh) else "U"
        if ph not in ("R", "L"):
            ph = "U"
        if oh not in ("R", "L"):
            oh = "U"
        key = f"{ph}v{oh}"
        return cfg.HAND_MATCHUP_ENCODE.get(key, 0)

    df["hand_matchup_enc"] = df.apply(
        lambda r: _hand_matchup(r["player_hand"], r["opp_hand"]), axis=1
    )
    df["best_of_enc"] = df["best_of"].fillna(3).astype(int)

    return df


# ---------------------------------------------------------------------------
# Win pct on same surface (rolling)
# ---------------------------------------------------------------------------

def _rolling_surface_win_pct(grp: pd.DataFrame, surface: str, window: int = 10) -> pd.Series:
    g = _sort_match_time(_ensure_sim_date(grp))
    mask = g["surface"].str.lower() == surface.lower()
    vals = g["won"].astype(float).copy()
    vals[~mask] = np.nan
    return vals.shift(1).rolling(window, min_periods=3).mean()


# ---------------------------------------------------------------------------
# Master build
# ---------------------------------------------------------------------------

def get_feature_columns() -> list[str]:
    return [
        # ELO
        "elo_diff_overall", "elo_diff_surface",
        "player_elo_overall", "opp_elo_overall",
        # Serve/return (all surface, 20-match window)
        "spw_20", "rpw_20", "ace_rate_20", "df_rate_20",
        "first_serve_pct_20", "first_serve_win_pct_20", "second_serve_win_pct_20",
        "bp_save_pct_20", "bp_conversion_pct_20",
        # Same-surface serve/return (10-match window)
        "spw_surf10", "rpw_surf10",
        "first_serve_pct_surf10", "first_serve_win_pct_surf10",
        "bp_save_pct_surf10",
        # Differentials
        "spw_diff", "rpw_diff", "ace_rate_diff", "bp_save_pct_diff",
        # Form
        "win_pct_last_10", "win_pct_last_5",
        "sets_win_pct_last_10", "deciding_set_win_pct",
        "win_pct_last_10_surface",
        # H2H
        "h2h_n", "h2h_win_pct_adj", "h2h_win_pct_surface", "h2h_win_pct_recent",
        # Context
        "tourney_level_enc", "round_enc", "surface_enc",
        "rank_diff_log", "age_diff", "hand_matchup_enc", "best_of_enc",
        "days_since_last_match", "matches_last_14_days",
    ]


def build_features(player_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Full feature build pipeline. Input is player-perspective match rows."""
    player_df = _ensure_sim_date(player_df)
    if verbose:
        print(f"Computing ELO ratings ({len(player_df)} player-match rows)...")
    df = compute_elo_ratings(player_df)

    if verbose:
        print("Computing serve/return rolling stats...")
    df = _compute_serve_stats(df)

    out_frames = []
    for pid, grp in df.groupby("player_id"):
        g = _sort_match_time(_ensure_sim_date(grp))
        g = _rolling_serve_stats(g, window=cfg.ROLLING_ALL_SURFACE, suffix="20")

        g = _rolling_surface_stats(g, window=cfg.ROLLING_SAME_SURFACE, suffix="surf10")

        g = _rolling_form(g)

        # Surface win pct
        g["win_pct_last_10_surface"] = np.nan
        for surf in ["hard", "clay", "grass"]:
            mask = g["surface"].str.lower() == surf
            pct = _rolling_surface_win_pct(g, surf, window=10)
            g.loc[mask, "win_pct_last_10_surface"] = pct[mask]

        out_frames.append(g)

    if verbose:
        print("Computing H2H features...")
    df_rolled = _sort_match_time(pd.concat(out_frames, ignore_index=True))
    df_h2h = compute_h2h(df_rolled)

    if verbose:
        print("Adding context features...")
    df_ctx = add_context_features(df_h2h)

    # Differential serve stats (player - opponent for same-window)
    for base in ["spw", "rpw", "ace_rate", "bp_save_pct"]:
        if f"{base}_20" in df_ctx.columns:
            opp_col = f"opp_{base}_20"
            if opp_col not in df_ctx.columns:
                df_ctx[f"{base}_diff"] = np.nan
            else:
                df_ctx[f"{base}_diff"] = df_ctx[f"{base}_20"] - df_ctx[opp_col]

    if verbose:
        print(f"Feature build complete. {len(df_ctx)} rows, {len(df_ctx.columns)} columns.")
    return df_ctx


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_features(features: pd.DataFrame) -> None:
    out = FEATURES_DIR / "features.parquet"
    features.to_parquet(out, index=False)
    print(f"Saved features → {out}")


def load_features() -> pd.DataFrame:
    path = FEATURES_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features parquet not found at {path}.")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from src.data.tennis_sackmann import load_matches

    print("Loading match DB...")
    player_df = load_matches()
    print(f"  {len(player_df)} player-perspective rows")

    # Filter: only include years in training window for feature build
    player_df = player_df[player_df["tourney_date"].dt.year >= 2000].copy()

    features = build_features(player_df, verbose=True)
    save_features(features)
    print("Done.")


if __name__ == "__main__":
    main()
