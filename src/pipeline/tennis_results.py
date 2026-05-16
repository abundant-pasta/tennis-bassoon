"""
Governance rules and Kelly sizing for the tennis pipeline.

Mirrors src/pipeline/results.py from the MLB system.

governance_reason() returns a string describing why a bet is blocked,
or None if the bet passes all governance checks.

kelly_adjustment() returns a multiplier (0.0–1.0) to apply to the
fractional Kelly stake for soft governance adjustments (not full blocks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config_tennis as cfg


def governance_reason(row: pd.Series | dict, edge: float, side: str) -> str | None:
    """Return a governance block reason string, or None if the bet is allowed."""
    reasons: list[str] = []

    tourney_level = str(row.get("tourney_level", ""))
    surface = str(row.get("surface", "")).lower()
    round_ = str(row.get("round", ""))
    side = str(side).lower()

    if side in {"player", "opp"}:
        selected_rank = _safe_float(row.get("player_rank") if side == "player" else row.get("opp_rank"))
        opp_rank = _safe_float(row.get("opp_rank") if side == "player" else row.get("player_rank"))
    else:
        selected_rank = _safe_float(row.get("winner_rank") if side == "winner" else row.get("loser_rank"))
        opp_rank = _safe_float(row.get("loser_rank") if side == "winner" else row.get("winner_rank"))

    score = str(row.get("score", ""))
    is_wo = "W/O" in score.upper()
    is_ret = "RET" in score.upper()

    # Hard gate: walkover or retirement
    if is_wo:
        reasons.append("walkover — not predictable pre-match")
    if is_ret:
        reasons.append("retirement — outcome not pre-match predictable")

    # Qualifying rounds: model not trained on qual data
    if round_ in ("Q1", "Q2", "Q3", "Q4", "ER"):
        reasons.append(f"qualifying round ({round_}) — model not trained on qual pool")

    # Missing ranks
    if selected_rank is None or np.isnan(selected_rank):
        reasons.append("selected side rank missing — feature reliability low")
    if opp_rank is None or np.isnan(opp_rank):
        reasons.append("opponent rank missing — feature reliability low")

    # Low-rank opponent
    if opp_rank is not None and not np.isnan(opp_rank) and opp_rank > cfg.MIN_RANK_BOTH_PLAYERS:
        reasons.append(
            f"opponent ranked {int(opp_rank)} > {cfg.MIN_RANK_BOTH_PLAYERS} — feature sparse"
        )

    # Extreme edge on Challengers
    if tourney_level == "C" and edge >= cfg.CHALLENGER_EXTREME_EDGE:
        reasons.append(
            f"Challenger extreme edge {edge:.1%} >= {cfg.CHALLENGER_EXTREME_EDGE:.1%} — thin market"
        )

    # Recent 2025 audit showed oversized Masters edges were miscalibrated.
    if tourney_level == "M" and edge >= cfg.MASTERS_EXTREME_EDGE:
        reasons.append(
            f"Masters extreme edge {edge:.1%} >= {cfg.MASTERS_EXTREME_EDGE:.1%} — recent calibration unstable"
        )

    # Grass early-round block
    if surface == "grass" and round_ in cfg.GRASS_EARLY_ROUND_BLOCK:
        reasons.append(f"grass early round {round_} — historically noisy surface adjustment period")

    return "; ".join(reasons) if reasons else None


def kelly_adjustment(row: pd.Series | dict, edge: float) -> float:
    """Return Kelly multiplier for soft governance adjustments (0.0–1.0)."""
    adj = 1.0

    # H2H thin sample
    h2h_n = _safe_float(row.get("h2h_n"))
    if h2h_n is not None and h2h_n < cfg.H2H_MIN_SAMPLE_KELLY_REDUCTION:
        adj *= cfg.KELLY_H2H_REDUCTION

    # Missing rank
    player_rank = _safe_float(row.get("winner_rank", row.get("player_rank")))
    if player_rank is None or np.isnan(player_rank):
        adj *= cfg.KELLY_HALF_FRACTION

    # Surface-sparse rolling window
    surface_matches = _safe_float(row.get("surface_matches_count"))
    if surface_matches is not None and surface_matches < cfg.MIN_SURFACE_MATCHES:
        adj *= cfg.KELLY_HALF_FRACTION

    return adj


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def build_recommendation(
    match_row: dict,
    model_prob_winner: float,
    market_prob_winner: float,
    decimal_odds_winner: float,
    decimal_odds_loser: float,
    edge_threshold: float = cfg.EDGE_THRESHOLD,
    kelly_fraction: float = cfg.KELLY_FRACTION,
    max_stake_fraction: float = cfg.MAX_STAKE_FRACTION,
    bankroll: float = cfg.STARTING_BANKROLL,
) -> dict:
    """Generate a recommendation dict for a single match.

    Returns a dict with side, edge, stake, governance details.
    """
    model_prob_loser = 1.0 - model_prob_winner
    edge_winner = model_prob_winner - market_prob_winner
    edge_loser = model_prob_loser - (1.0 - market_prob_winner)

    if edge_winner >= edge_loser:
        side = "winner"
        edge = edge_winner
        model_prob = model_prob_winner
        decimal_odds = decimal_odds_winner
    else:
        side = "loser"
        edge = edge_loser
        model_prob = model_prob_loser
        decimal_odds = decimal_odds_loser

    gov_block = governance_reason(match_row, edge, side)
    adj = kelly_adjustment(match_row, edge)

    if edge < edge_threshold or gov_block is not None:
        stake = 0.0
        full_k = 0.0
    else:
        b = decimal_odds - 1.0
        full_k = max(0.0, (b * model_prob - (1 - model_prob)) / b) if b > 0 else 0.0
        used = min(full_k * kelly_fraction * adj, max_stake_fraction)
        stake = bankroll * used

    return {
        "side": side,
        "edge": round(edge, 4),
        "model_prob": round(model_prob, 4),
        "market_prob": round(market_prob_winner if side == "winner" else 1.0 - market_prob_winner, 4),
        "decimal_odds": round(decimal_odds, 3),
        "full_kelly": round(full_k, 4),
        "kelly_adjustment": round(adj, 4),
        "stake": round(stake, 2),
        "governance_block": gov_block,
        "eligible": edge >= edge_threshold and gov_block is None,
    }
