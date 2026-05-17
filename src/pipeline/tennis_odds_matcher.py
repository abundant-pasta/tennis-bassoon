"""Confidence-scored matching between schedule snapshots and odds snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

import config_tennis as cfg
from src.data.tennis_odds import devig_multiplicative


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def normalize_match_frame(df: pd.DataFrame, player_col: str, opp_col: str, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", utc=True)
    out["_player_norm"] = out[player_col].map(_norm_text)
    out["_opp_norm"] = out[opp_col].map(_norm_text)
    out["_pair"] = out.apply(
        lambda r: tuple(sorted([r["_player_norm"], r["_opp_norm"]])),
        axis=1,
    )
    out["_surface_norm"] = out["surface"].fillna("").astype(str).str.lower() if "surface" in out.columns else ""
    out["_round_norm"] = out["round"].fillna("").astype(str) if "round" in out.columns else ""
    out["_tourney_norm"] = out["tourney_name"].map(_norm_text) if "tourney_name" in out.columns else ""
    return out


@dataclass(frozen=True)
class MatchResult:
    matched: pd.DataFrame
    rejected: pd.DataFrame


def match_schedule_to_odds(
    schedule_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    max_date_distance_days: int = cfg.MAX_ODDS_DATE_DISTANCE_DAYS,
) -> MatchResult:
    schedule = normalize_match_frame(schedule_df, "player_name", "opp_name", "match_date").copy()
    odds = normalize_match_frame(odds_df, "player_name", "opp_name", "match_date").copy()
    schedule["_row_id"] = np.arange(len(schedule))
    odds["_candidate_id"] = np.arange(len(odds))

    merged = schedule.merge(
        odds,
        on="_pair",
        how="left",
        suffixes=("", "_odds"),
    )
    merged["date_distance_days"] = (
        pd.to_datetime(merged["match_date"]) - pd.to_datetime(merged["match_date_odds"])
    ).abs().dt.days
    merged["surface_match"] = merged["_surface_norm"] == merged["_surface_norm_odds"]
    merged["round_match"] = merged["_round_norm"] == merged["_round_norm_odds"]
    merged["tourney_match"] = merged["_tourney_norm"] == merged["_tourney_norm_odds"]
    merged["surface_penalty"] = np.where(merged["surface_match"], 0, 0.40)
    merged["round_penalty"] = np.where(merged["round_match"], 0, 0.10)
    merged["tourney_penalty"] = np.where(merged["tourney_match"], 0, 0.20)
    merged["date_penalty"] = merged["date_distance_days"].fillna(max_date_distance_days + 1) / max_date_distance_days
    merged["raw_penalty"] = (
        merged["surface_penalty"] + merged["round_penalty"] + merged["tourney_penalty"] + merged["date_penalty"]
    )
    merged["odds_match_confidence"] = (1.0 - merged["raw_penalty"]).clip(lower=0.0, upper=1.0)

    same_orientation = merged["_player_norm"] == merged["_player_norm_odds"]
    reversed_orientation = merged["_player_norm"] == merged["_opp_norm_odds"]
    valid = merged[
        merged["player_decimal_odds"].notna()
        & merged["opp_decimal_odds"].notna()
        & merged["surface_match"]
        & (merged["date_distance_days"] <= max_date_distance_days)
        & (same_orientation | reversed_orientation)
    ].copy()

    if valid.empty:
        rejected = schedule.copy()
        rejected["rejection_reason"] = "no_valid_odds_candidate"
        rejected["odds_match_confidence"] = 0.0
        return MatchResult(matched=schedule.iloc[0:0].copy(), rejected=rejected)

    same_orientation = valid["_player_norm"] == valid["_player_norm_odds"]
    raw_player_odds = valid["player_decimal_odds"].copy()
    raw_opp_odds = valid["opp_decimal_odds"].copy()
    valid["player_decimal_odds"] = np.where(same_orientation, raw_player_odds, raw_opp_odds)
    valid["opp_decimal_odds"] = np.where(same_orientation, raw_opp_odds, raw_player_odds)

    valid = valid.sort_values(
        ["_row_id", "raw_penalty", "date_distance_days", "_candidate_id"],
        ascending=[True, True, True, True],
    )
    best = valid.drop_duplicates(subset=["_row_id"], keep="first").copy()
    ambiguity = (
        valid.groupby("_row_id")["raw_penalty"]
        .apply(lambda s: len(s) > 1 and abs(float(sorted(s.tolist())[0]) - float(sorted(s.tolist())[1])) < 0.05)
        .rename("ambiguous_odds_match")
        .reset_index()
    )
    best = best.merge(ambiguity, on="_row_id", how="left")
    best["ambiguous_odds_match"] = best["ambiguous_odds_match"].fillna(False)

    best["novig_player_prob"] = best.apply(
        lambda r: devig_multiplicative(float(r["player_decimal_odds"]), float(r["opp_decimal_odds"]))[0],
        axis=1,
    )
    best["novig_opp_prob"] = 1.0 - best["novig_player_prob"]

    ambiguous_ids = set(best.loc[best["ambiguous_odds_match"], "_row_id"].tolist())
    matched_ids = set(best.loc[~best["ambiguous_odds_match"], "_row_id"].tolist())
    no_match_ids = set(schedule["_row_id"].tolist()) - matched_ids - ambiguous_ids

    rejected_frames: list[pd.DataFrame] = []
    if ambiguous_ids:
        rejected_amb = schedule[schedule["_row_id"].isin(ambiguous_ids)].copy()
        rejected_amb["rejection_reason"] = "ambiguous_odds_match"
        rejected_amb["odds_match_confidence"] = 0.0
        rejected_frames.append(rejected_amb)
    if no_match_ids:
        rejected_nomatch = schedule[schedule["_row_id"].isin(no_match_ids)].copy()
        rejected_nomatch["rejection_reason"] = "no_valid_odds_candidate"
        rejected_nomatch["odds_match_confidence"] = 0.0
        rejected_frames.append(rejected_nomatch)
    rejected = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else schedule.iloc[0:0].copy()

    matched = best[~best["ambiguous_odds_match"]].copy()
    provider_series = matched["provider"] if "provider" in matched.columns else pd.Series(index=matched.index, dtype="object")
    matched["odds_provider"] = np.where(provider_series.notna(), provider_series, "generic_csv")
    matched["rejection_reason"] = None
    keep_cols = [
        "match_id",
        "match_date",
        "player_name",
        "opp_name",
        "surface",
        "round",
        "tourney_name",
        "player_rank",
        "opp_rank",
        "best_of",
        "draw_size",
        "player_decimal_odds",
        "opp_decimal_odds",
        "novig_player_prob",
        "novig_opp_prob",
        "odds_match_confidence",
        "odds_provider",
        "rejection_reason",
        "date_distance_days",
    ]
    for col in keep_cols:
        if col not in matched.columns:
            matched[col] = np.nan
    for col in ["match_id", "match_date", "player_name", "opp_name", "surface", "round", "tourney_name"]:
        if col not in rejected.columns:
            rejected[col] = None
    return MatchResult(matched=matched[keep_cols].reset_index(drop=True), rejected=rejected.reset_index(drop=True))
