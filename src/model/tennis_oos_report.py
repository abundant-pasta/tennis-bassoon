"""
Out-of-sample Kelly simulation for the tennis model.

For each match in the OOS window (default 2022–2024):
  1. Generate model win probability
  2. Compute no-vig market implied probability from closing odds
  3. Compute edge
  4. Apply governance rules
  5. Simulate 0.25 Kelly bet if edge > threshold and not governed
  6. Track P&L, ROI, CLV, surface/level breakdowns

Mirrors src/model/oos_report.py from the MLB system.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config_tennis as cfg
from src.data.tennis_odds import devig_multiplicative, load_odds, join_odds_to_matches
from src.features.tennis_engineer import get_feature_columns, load_features
from src.model.tennis_calibration import TennisCalibratedModel  # noqa: F401
from src.pipeline.tennis_results import governance_reason, kelly_adjustment

OOS_DIR = cfg.OOS_DIR
OOS_DIR.mkdir(parents=True, exist_ok=True)

_EDGE_BUCKETS = [0.06, 0.08, 0.10, 0.12]
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_bucket(edge: float | None) -> str | None:
    if edge is None or np.isnan(edge):
        return None
    if edge < 0.08:
        return "06-08%"
    if edge < 0.10:
        return "08-10%"
    if edge < 0.12:
        return "10-12%"
    return "12%+"


def _decimal_to_implied(decimal: float) -> float:
    if decimal <= 1.0:
        return np.nan
    return 1.0 / decimal


def _kelly(prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (b * prob - (1 - prob)) / b
    return max(0.0, f)


def _profit(stake: float, decimal_odds: float, won: bool) -> float:
    if won:
        return stake * (decimal_odds - 1.0)
    return -stake


def _preferred_odds(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    return primary.where(primary.notna() & (primary > 1.0), fallback)


def _normalize_tourney_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _estimate_match_date(tourney_date: pd.Series, round_: pd.Series) -> pd.Series:
    base = pd.to_datetime(tourney_date)
    offsets = round_.map(lambda r: _ROUND_DAY_OFFSET.get(str(r), 2))
    return base + pd.to_timedelta(offsets, unit="D")


def _round_sort_value(round_: str) -> int:
    return _ROUND_SORT_ORDER.get(str(round_), 2)


def _oos_auc(pred_df: pd.DataFrame) -> float | None:
    try:
        y_true = pred_df["won"].astype(int)
        y_prob = pred_df["player_win_prob"].astype(float)
        if y_true.nunique() < 2:
            return None
        return round(float(roc_auc_score(y_true, y_prob)), 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _load_model(
    holdout: bool = True,
    model_path: Path | None = None,
    scaler_path: Path | None = None,
    medians_path: Path | None = None,
):
    path = model_path or (cfg.HOLDOUT_MODEL_PATH if holdout else cfg.MODEL_PATH)
    scaler_path = scaler_path or (cfg.HOLDOUT_SCALER_PATH if holdout else cfg.SCALER_PATH)
    medians_path = medians_path or (cfg.HOLDOUT_MEDIANS_PATH if holdout else cfg.MEDIANS_PATH)
    if not path.exists():
        path = cfg.MODEL_PATH
        scaler_path = cfg.SCALER_PATH
        medians_path = cfg.MEDIANS_PATH
    if not path.exists():
        raise FileNotFoundError(f"No model artifact found at {path}.")
    medians = joblib.load(medians_path) if medians_path.exists() else None
    return joblib.load(path), joblib.load(scaler_path), medians


def _deduplicate_to_one_row_per_match(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the higher-ranked player's row per match (stable, no leakage)."""
    df = df.copy()
    df["_rank_p"] = df["player_rank"].fillna(9999)
    df["_rank_o"] = df["opp_rank"].fillna(9999)
    df["_pid"] = pd.to_numeric(df["player_id"], errors="coerce").fillna(9999999)
    df["_oid"] = pd.to_numeric(df["opp_id"], errors="coerce").fillna(9999999)
    df["_keep"] = (df["_rank_p"] < df["_rank_o"]) | (
        (df["_rank_p"] == df["_rank_o"]) & (df["_pid"] <= df["_oid"])
    )
    df = df.sort_values(["match_id", "_keep"], ascending=[True, False])
    kept = df[df["_keep"]].drop_duplicates(subset=["match_id"]).copy()
    kept.drop(columns=["_rank_p", "_rank_o", "_pid", "_oid", "_keep"], inplace=True)
    return kept.reset_index(drop=True)


def _predict_oos(features: pd.DataFrame, model, scaler, oos_years: list[int], feature_medians: dict | None = None) -> pd.DataFrame:
    feat_cols = get_feature_columns()
    df = features.copy()
    df["year"] = pd.to_datetime(df["tourney_date"]).dt.year
    df = df[df["year"].isin(oos_years)].copy()
    if df.empty:
        raise RuntimeError(f"No feature rows for OOS years {oos_years}.")

    # Deduplicate to one row per match before prediction
    df = _deduplicate_to_one_row_per_match(df)

    fill_values = pd.Series(feature_medians) if feature_medians is not None else df[feat_cols].median(numeric_only=True)
    df[feat_cols] = df[feat_cols].fillna(fill_values)
    X = scaler.transform(df[feat_cols].values)
    raw_prob = model.predict_proba(X)[:, 1]
    # Store the model probability directly. Kelly sizing uses a separate, mild
    # clamp later so edge selection is not distorted by clipping.
    df["player_win_prob_raw"] = raw_prob
    df["player_win_prob"] = raw_prob
    return df


def _attach_odds(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Join odds to all match predictions using symmetric pair matching.

    After deduplication, each match appears once from the higher-ranked player's
    perspective. We match using sorted (player_name, opp_name) pair + year_month
    so we can join regardless of whether our 'player' was the actual winner or loser.
    """
    from src.data.tennis_odds import add_novig_probs, _normalize_name

    try:
        odds = load_odds()
    except FileNotFoundError:
        print("  No odds file found — run tennis_odds.py first. Skipping odds join.")
        for col in ["novig_winner_prob", "novig_loser_prob",
                    "pinnacle_winner_odds", "pinnacle_loser_odds",
                    "winner_name", "loser_name",
                    "winner_model_prob", "loser_model_prob",
                    "winner_rank", "loser_rank"]:
            pred_df[col] = np.nan
        return pred_df

    odds = add_novig_probs(odds)
    # Symmetric key: sorted pair of normalized last names + year_month
    odds["_n1"] = odds["winner_name_odds"].apply(_normalize_name)
    odds["_n2"] = odds["loser_name_odds"].apply(_normalize_name)
    odds["_pair"] = odds.apply(lambda r: tuple(sorted([r["_n1"], r["_n2"]])), axis=1)
    odds["_ym"] = odds["match_date_odds"].dt.to_period("M").astype(str)
    odds["_surface_norm"] = odds["surface_odds"].fillna("").astype(str).str.lower()
    odds["_round_norm"] = odds["round_odds"].fillna("").astype(str)
    odds["_tourney_norm"] = odds["tourney_name_odds"].apply(_normalize_tourney_name)

    df = pred_df.copy()
    df["_np"] = df["player_name"].apply(_normalize_name)
    df["_no"] = df["opp_name"].apply(_normalize_name)
    df["_pair"] = df.apply(lambda r: tuple(sorted([r["_np"], r["_no"]])), axis=1)
    df["_tourney_dt"] = pd.to_datetime(df["tourney_date"])
    df["_estimated_match_dt"] = _estimate_match_date(df["tourney_date"], df["round"])
    df["_ym"] = df["_estimated_match_dt"].dt.to_period("M").astype(str)
    df["_ym_next"] = (df["_estimated_match_dt"] + pd.DateOffset(months=1)).dt.to_period("M").astype(str)
    df["_surface_norm"] = df["surface"].fillna("").astype(str).str.lower()
    df["_round_norm"] = df["round"].fillna("").astype(str)
    df["_tourney_norm"] = df["tourney_name"].apply(_normalize_tourney_name)
    df["_row_id"] = np.arange(len(df))

    keep_odds = ["_ym", "_pair", "_n1", "_n2",
                 "winner_name_odds", "loser_name_odds",
                 "match_date_odds", "tourney_name_odds", "surface_odds", "round_odds",
                 "pinnacle_winner_odds", "pinnacle_loser_odds",
                 "b365_winner_odds", "b365_loser_odds",
                 "novig_winner_prob", "novig_loser_prob",
                 "_surface_norm", "_round_norm", "_tourney_norm"]

    same_month = df.merge(
        odds[keep_odds],
        on=["_ym", "_pair"],
        how="left",
        suffixes=("", "_odds"),
    )

    next_month_base = df.drop(columns=["_ym"]).rename(columns={"_ym_next": "_ym"})
    next_month = next_month_base.merge(
        odds[keep_odds],
        on=["_ym", "_pair"],
        how="left",
        suffixes=("", "_odds"),
    )

    candidates = pd.concat([same_month, next_month], ignore_index=True)
    candidates = candidates[candidates["novig_winner_prob"].notna()].copy()

    if candidates.empty:
        for col in ["novig_winner_prob", "novig_loser_prob",
                    "pinnacle_winner_odds", "pinnacle_loser_odds",
                    "winner_name", "loser_name",
                    "winner_model_prob", "loser_model_prob",
                    "winner_rank", "loser_rank"]:
            df[col] = np.nan
        return df

    candidates["_date_distance"] = (
        pd.to_datetime(candidates["match_date_odds"]) - pd.to_datetime(candidates["_estimated_match_dt"])
    ).abs().dt.days.fillna(999)
    candidates["_surface_penalty"] = (candidates["_surface_norm"] != candidates["_surface_norm_odds"]).astype(int) * 30
    candidates["_round_penalty"] = (candidates["_round_norm"] != candidates["_round_norm_odds"]).astype(int) * 7
    candidates["_tourney_penalty"] = (candidates["_tourney_norm"] != candidates["_tourney_norm_odds"]).astype(int) * 14
    candidates["_score"] = (
        candidates["_date_distance"]
        + candidates["_surface_penalty"]
        + candidates["_round_penalty"]
        + candidates["_tourney_penalty"]
    )
    candidates = candidates[
        (candidates["_surface_norm"] == candidates["_surface_norm_odds"])
        & (candidates["_date_distance"] <= cfg.MAX_ODDS_DATE_DISTANCE_DAYS)
    ].copy()
    if candidates.empty:
        for col in ["novig_winner_prob", "novig_loser_prob",
                    "pinnacle_winner_odds", "pinnacle_loser_odds",
                    "b365_winner_odds", "b365_loser_odds",
                    "winner_name", "loser_name",
                    "winner_model_prob", "loser_model_prob",
                    "winner_rank", "loser_rank"]:
            df[col] = np.nan
        return df
    candidates = candidates.sort_values(["_row_id", "_score", "_date_distance", "match_date_odds"])
    best = candidates.drop_duplicates(subset=["_row_id"], keep="first").copy()
    best = best.sort_values("_row_id").reset_index(drop=True)

    fill_cols = [
        "_n1", "_n2",
        "winner_name_odds", "loser_name_odds",
        "match_date_odds", "tourney_name_odds", "surface_odds", "round_odds",
        "pinnacle_winner_odds", "pinnacle_loser_odds",
        "b365_winner_odds", "b365_loser_odds",
        "novig_winner_prob", "novig_loser_prob",
        "_surface_norm_odds", "_round_norm_odds", "_tourney_norm_odds",
        "_score", "_date_distance", "_surface_penalty", "_round_penalty", "_tourney_penalty",
    ]
    merged = df.merge(best[["_row_id", *fill_cols]], on="_row_id", how="left")

    # Now determine player/opp odds: check if player_name matches winner_name_odds
    # If player is the odds-winner: player gets winner odds, opp gets loser odds
    # If player is the odds-loser: player gets loser odds, opp gets winner odds
    merged["_np_match_winner"] = merged["_np"] == merged["_n1"]

    raw_col = "player_win_prob_raw" if "player_win_prob_raw" in merged.columns else "player_win_prob"
    merged["player_model_prob_raw"] = merged[raw_col]
    merged["opp_model_prob_raw"] = 1.0 - merged[raw_col]
    merged["player_model_prob"] = merged["player_win_prob"]
    merged["opp_model_prob"] = 1.0 - merged["player_win_prob"]

    # Market prob and decimal odds from player's perspective
    merged["player_market_prob"] = np.where(
        merged["_np_match_winner"],
        merged["novig_winner_prob"],
        merged["novig_loser_prob"],
    )
    merged["opp_market_prob"] = np.where(
        merged["_np_match_winner"],
        merged["novig_loser_prob"],
        merged["novig_winner_prob"],
    )
    merged["player_decimal_odds"] = np.where(
        merged["_np_match_winner"],
        _preferred_odds(merged["pinnacle_winner_odds"], merged["b365_winner_odds"]),
        _preferred_odds(merged["pinnacle_loser_odds"], merged["b365_loser_odds"]),
    )
    merged["opp_decimal_odds"] = np.where(
        merged["_np_match_winner"],
        _preferred_odds(merged["pinnacle_loser_odds"], merged["b365_loser_odds"]),
        _preferred_odds(merged["pinnacle_winner_odds"], merged["b365_winner_odds"]),
    )

    # Canonical winner/loser columns for reporting
    merged["winner_name"] = np.where(
        merged["_np_match_winner"], merged["player_name"], merged["opp_name"]
    )
    merged["loser_name"] = np.where(
        merged["_np_match_winner"], merged["opp_name"], merged["player_name"]
    )
    merged["winner_model_prob"] = np.where(
        merged["_np_match_winner"], merged["player_model_prob"], merged["opp_model_prob"]
    )
    merged["loser_model_prob"] = 1.0 - merged["winner_model_prob"]
    merged["winner_rank"] = np.where(
        merged["_np_match_winner"], merged["player_rank"], merged["opp_rank"]
    )
    merged["loser_rank"] = np.where(
        merged["_np_match_winner"], merged["opp_rank"], merged["player_rank"]
    )

    merged["sim_date"] = merged["match_date_odds"].fillna(merged["_estimated_match_dt"])

    merged.drop(columns=["_np", "_no", "_pair", "_ym", "_ym_next", "_tourney_dt",
                          "_estimated_match_dt", "_surface_norm", "_surface_norm_odds",
                          "_round_norm", "_round_norm_odds", "_tourney_norm", "_tourney_norm_odds",
                          "_row_id", "_score", "_date_distance", "_surface_penalty",
                          "_round_penalty", "_tourney_penalty",
                          "_n1", "_n2", "_np_match_winner"], inplace=True, errors="ignore")
    return merged


def _score_picks(df: pd.DataFrame, edge_threshold: float) -> pd.DataFrame:
    """Score matches with model edge vs. market, select the better side to bet."""
    df = df.copy()
    df = df[df["player_market_prob"].notna()].copy()

    # Use the same clipped probabilities for side selection, edge gating, and Kelly sizing.
    df["edge_player"] = df["player_model_prob"] - df["player_market_prob"]
    df["edge_opp"] = df["opp_model_prob"] - df["opp_market_prob"]

    df["edge"] = df[["edge_player", "edge_opp"]].max(axis=1)
    df["side"] = np.where(
        df["edge_player"] >= df["edge_opp"], "player", "opp"
    )
    df["model_prob"] = np.where(
        df["side"] == "player", df["player_model_prob"], df["opp_model_prob"]
    )
    df["market_prob"] = np.where(
        df["side"] == "player", df["player_market_prob"], df["opp_market_prob"]
    )
    # Keep edge selection based on the raw model probability, but compress larger
    # edges for Kelly sizing so recent-era calibration drift does not over-stake them.
    df["kelly_prob_raw"] = (
        np.clip(df["model_prob"], cfg.KELLY_PROB_FLOOR, cfg.KELLY_PROB_CEIL).astype(float)
    )
    df["kelly_prob"] = df["kelly_prob_raw"].astype(float)
    compress_mask = df["edge"] > cfg.KELLY_EDGE_COMPRESSION_THRESHOLD
    if compress_mask.any():
        compressed = (
            df.loc[compress_mask, "market_prob"]
            + cfg.KELLY_EDGE_COMPRESSION_FACTOR
            * (df.loc[compress_mask, "kelly_prob_raw"] - df.loc[compress_mask, "market_prob"])
        )
        df.loc[compress_mask, "kelly_prob"] = compressed.clip(cfg.KELLY_PROB_FLOOR, cfg.KELLY_PROB_CEIL)
    df["decimal_odds"] = np.where(
        df["side"] == "player", df["player_decimal_odds"], df["opp_decimal_odds"]
    )
    df["full_kelly"] = df.apply(
        lambda r: _kelly(float(r["kelly_prob_raw"]), float(r["decimal_odds"]))
        if pd.notna(r["decimal_odds"]) and float(r["decimal_odds"]) > 1.0 else 0.0,
        axis=1,
    )
    # won_bet: bet is on player → won if player won (won=1); on opp → won if player lost (won=0)
    df["won_bet"] = np.where(
        df["side"] == "player", df["won"] == 1, df["won"] == 0
    ).astype(int)

    df["above_threshold"] = df["edge"] > edge_threshold
    df["bettable_odds"] = df["decimal_odds"].notna() & (df["decimal_odds"] > 1.0)
    df["governance_block"] = df.apply(
        lambda r: governance_reason(r, r.get("edge", 0.0), r.get("side", "player")),
        axis=1,
    )
    df["eligible"] = (
        df["above_threshold"]
        & df["bettable_odds"]
        & (df["full_kelly"] > 0)
        & df["governance_block"].isna()
    )
    df["edge_bucket"] = df["edge"].apply(_edge_bucket)
    return df


def _simulate_flat_bet(
    picks: pd.DataFrame,
    unit: float = 1.0,
) -> dict:
    """Simulate flat-bet (1 unit per bet) on eligible picks. Returns summary dict."""
    eligible = picks[picks["eligible"]].copy()
    if eligible.empty:
        return {"bets": 0, "wins": 0, "win_rate": None, "units_profit": None, "flat_roi": None}

    eligible["_round_order"] = eligible["round"].map(_round_sort_value)
    eligible = eligible.sort_values(["sim_date", "_round_order", "match_id"])
    total_profit_units = 0.0
    wins = 0
    bets = 0
    for _, row in eligible.iterrows():
        dec_odds = float(row["decimal_odds"])
        if np.isnan(dec_odds) or dec_odds <= 1.0:
            continue
        won = bool(row["won_bet"])
        bets += 1
        if won:
            profit = (dec_odds - 1.0) * unit
        else:
            profit = -unit
        total_profit_units += profit
        if won:
            wins += 1

    return {
        "bets": bets,
        "wins": wins,
        "win_rate": round(wins / bets, 4) if bets > 0 else None,
        "units_profit": round(total_profit_units, 4),
        "flat_roi": round(total_profit_units / bets, 4) if bets > 0 else None,
    }


def _simulate_kelly(
    picks: pd.DataFrame,
    starting_bankroll: float = cfg.STARTING_BANKROLL,
    kelly_fraction: float = cfg.KELLY_FRACTION,
    max_stake_fraction: float = cfg.MAX_STAKE_FRACTION,
    max_daily_exposure: float = cfg.MAX_DAILY_EXPOSURE,
) -> pd.DataFrame:
    bankroll = float(starting_bankroll)
    records = []
    eligible = picks[picks["eligible"]].copy()
    eligible["_round_order"] = eligible["round"].map(_round_sort_value)
    eligible = eligible.sort_values(["sim_date", "_round_order", "match_id"])
    for sim_date, day_grp in eligible.groupby("sim_date", sort=True):
        bankroll_before_day = bankroll
        day_records = []
        day_profit = 0.0
        for _, row in day_grp.iterrows():
            dec_odds = float(row["decimal_odds"])
            if np.isnan(dec_odds) or dec_odds <= 1.0:
                continue

            prob = float(row["model_prob"])
            sizing_prob = float(row.get("kelly_prob", prob))
            full_k = _kelly(sizing_prob, dec_odds)
            adj = kelly_adjustment(row, float(row["edge"]))
            used = min(full_k * kelly_fraction * adj, max_stake_fraction)
            if used <= 0:
                continue

            stake = bankroll_before_day * used
            won = bool(row["won_bet"])
            profit = _profit(stake, dec_odds, won)
            day_profit += profit

            player_name = row.get("player_name", row.get("winner_name", ""))
            opp_name = row.get("opp_name", row.get("loser_name", ""))
            day_records.append({
                "date": str(pd.to_datetime(sim_date).date()),
                "match": f"{player_name} vs {opp_name}",
                "bet_on": "player" if row["side"] == "player" else "opp",
                "surface": row.get("surface", ""),
                "tourney_level": row.get("tourney_level", ""),
                "decimal_odds": round(dec_odds, 3),
                "model_prob": round(prob, 4),
                "kelly_prob": round(sizing_prob, 4),
                "market_prob": round(float(row["market_prob"]), 4),
                "edge": round(float(row["edge"]), 4),
                "edge_bucket": _edge_bucket(float(row["edge"])),
                "stake": round(stake, 2),
                "won": won,
                "profit": round(profit, 2),
                "bankroll_before": round(bankroll_before_day, 2),
                "full_kelly": round(full_k, 4),
                "kelly_adjustment": round(adj, 4),
                "kelly_used": round(used, 4),
            })

        total_used = sum(rec["kelly_used"] for rec in day_records)
        if total_used > max_daily_exposure and total_used > 0:
            scale = max_daily_exposure / total_used
            day_profit = 0.0
            for rec in day_records:
                rec["kelly_used"] = round(rec["kelly_used"] * scale, 4)
                rec["stake"] = round(bankroll_before_day * rec["kelly_used"], 2)
                won = bool(rec["won"])
                profit = _profit(rec["stake"], float(rec["decimal_odds"]), won)
                rec["profit"] = round(profit, 2)
                day_profit += profit

        bankroll += day_profit
        for rec in day_records:
            rec["bankroll"] = round(bankroll, 2)
            records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_oos_report(
    oos_years: list[int] | None = None,
    edge_threshold: float = cfg.EDGE_THRESHOLD,
    kelly_fraction: float = cfg.KELLY_FRACTION,
    starting_bankroll: float = cfg.STARTING_BANKROLL,
    use_holdout: bool = True,
    output_tag: str | None = None,
    model_path: Path | None = None,
    scaler_path: Path | None = None,
    medians_path: Path | None = None,
    features_df: pd.DataFrame | None = None,
) -> dict:
    oos_years = oos_years or cfg.OOS_YEARS

    print("Loading model...")
    model, scaler, feature_medians = _load_model(
        holdout=use_holdout,
        model_path=model_path,
        scaler_path=scaler_path,
        medians_path=medians_path,
    )

    print("Loading features...")
    features = features_df if features_df is not None else load_features()

    print(f"Predicting OOS years {oos_years}...")
    pred_df = _predict_oos(features, model, scaler, oos_years, feature_medians=feature_medians)
    print(f"  {len(pred_df)} player-match rows in OOS window")

    print("Joining odds...")
    match_df = _attach_odds(pred_df)
    print(f"  {len(match_df)} match rows after odds join")

    scored = _score_picks(match_df, edge_threshold)
    picks = scored[scored["eligible"]].copy()
    flat_stats = _simulate_flat_bet(picks)
    kelly_df = _simulate_kelly(picks, starting_bankroll, kelly_fraction)

    suffix = f"_{output_tag}" if output_tag else ""
    tag = "_".join(str(y) for y in oos_years)
    kelly_path = OOS_DIR / f"kelly_oos_{tag}{suffix}.csv"
    scored_path = OOS_DIR / f"scored_oos_{tag}{suffix}.csv"
    kelly_df.to_csv(kelly_path, index=False)
    scored.to_csv(scored_path, index=False)

    total_bets = len(kelly_df)
    wins = kelly_df["won"].sum() if total_bets > 0 else 0
    total_profit = kelly_df["profit"].sum() if total_bets > 0 else 0
    total_staked = kelly_df["stake"].sum() if total_bets > 0 else 1

    surface_breakdown = {}
    level_breakdown = {}
    bucket_breakdown = {}
    if total_bets > 0:
        for surf, grp in kelly_df.groupby("surface"):
            n = len(grp)
            surface_breakdown[surf] = {
                "bets": n,
                "wins": int(grp["won"].sum()),
                "profit": round(grp["profit"].sum(), 2),
                "roi": round(grp["profit"].sum() / grp["stake"].sum(), 4) if grp["stake"].sum() > 0 else None,
            }
        for lvl, grp in kelly_df.groupby("tourney_level"):
            n = len(grp)
            level_breakdown[lvl] = {
                "bets": n,
                "wins": int(grp["won"].sum()),
                "profit": round(grp["profit"].sum(), 2),
                "roi": round(grp["profit"].sum() / grp["stake"].sum(), 4) if grp["stake"].sum() > 0 else None,
            }
        for bkt, grp in kelly_df.groupby("edge_bucket"):
            if bkt:
                n = len(grp)
                bucket_breakdown[bkt] = {
                    "bets": n,
                    "wins": int(grp["won"].sum()),
                    "roi": round(grp["profit"].sum() / grp["stake"].sum(), 4) if grp["stake"].sum() > 0 else None,
                }

    # CLV approximation: not available without time-series odds data
    summary = {
        "oos_years": oos_years,
        "edge_threshold": edge_threshold,
        "kelly_fraction": kelly_fraction,
        "starting_bankroll": starting_bankroll,
        "auc_oos": _oos_auc(pred_df),
        "total_matches_with_odds": int(len(scored)),
        "governance_blocked": int(scored.get("governance_block", pd.Series()).notna().sum()),
        "flat_bet": flat_stats,
        "total_bets": total_bets,
        "win_rate": round(wins / total_bets, 4) if total_bets > 0 else None,
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / total_staked, 4) if total_staked > 0 else None,
        "final_bankroll": round(kelly_df["bankroll"].iloc[-1], 2) if total_bets > 0 else starting_bankroll,
        "surface_breakdown": surface_breakdown,
        "level_breakdown": level_breakdown,
        "edge_bucket_breakdown": bucket_breakdown,
        "kelly_path": str(kelly_path),
        "scored_path": str(scored_path),
    }

    report_path = OOS_DIR / f"oos_report_{tag}{suffix}.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"\nOOS report saved → {report_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Tennis OOS Kelly simulation.")
    parser.add_argument("--years", nargs="+", type=int, default=cfg.OOS_YEARS)
    parser.add_argument("--edge-threshold", type=float, default=cfg.EDGE_THRESHOLD)
    parser.add_argument("--kelly-fraction", type=float, default=cfg.KELLY_FRACTION)
    parser.add_argument("--starting-bankroll", type=float, default=cfg.STARTING_BANKROLL)
    parser.add_argument("--production", action="store_true",
                        help="Use production model instead of holdout")
    parser.add_argument("--output-tag")
    args, _ = parser.parse_known_args()

    summary = build_oos_report(
        oos_years=args.years,
        edge_threshold=args.edge_threshold,
        kelly_fraction=args.kelly_fraction,
        starting_bankroll=args.starting_bankroll,
        use_holdout=not args.production,
        output_tag=args.output_tag,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
