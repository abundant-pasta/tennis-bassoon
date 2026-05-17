"""Shadow ledger and odds snapshot helpers for tennis daily runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config_tennis as cfg
from src.data.tennis_odds import devig_multiplicative


def _date_dir(root: Path, run_date: str) -> Path:
    return root / run_date


def _compute_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    novig = out.apply(
        lambda r: pd.Series(
            devig_multiplicative(float(r["player_decimal_odds"]), float(r["opp_decimal_odds"]))
        ),
        axis=1,
    )
    out["player_market_prob"] = novig.iloc[:, 0].values
    out["opp_market_prob"] = novig.iloc[:, 1].values
    return out


def _snapshot_payload(matched_odds_df: pd.DataFrame) -> pd.DataFrame:
    return _compute_probs(
        matched_odds_df[
            [
                "match_id",
                "match_date",
                "tourney_name",
                "surface",
                "round",
                "player_name",
                "opp_name",
                "player_decimal_odds",
                "opp_decimal_odds",
                "odds_match_confidence",
                "odds_provider",
            ]
        ].copy()
    )


def _write_snapshot(out_path: Path, matched_odds_df: pd.DataFrame) -> Path:
    snapshot = _snapshot_payload(matched_odds_df)
    snapshot["captured_utc"] = datetime.now(timezone.utc).isoformat()
    snapshot.to_csv(out_path, index=False)
    return out_path


def save_current_odds_snapshot(run_date: str, run_id: str, matched_odds_df: pd.DataFrame) -> Path:
    out_dir = _date_dir(cfg.ODDS_SNAPSHOT_DIR, run_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"
    return _write_snapshot(out_path, matched_odds_df)


def ensure_opening_odds_snapshot(run_date: str, matched_odds_df: pd.DataFrame) -> Path:
    out_dir = _date_dir(cfg.OPENING_ODDS_DIR, run_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "opening.csv"
    if out_path.exists():
        return out_path
    return _write_snapshot(out_path, matched_odds_df)


def save_close_odds_snapshot(run_date: str, run_id: str, matched_odds_df: pd.DataFrame) -> Path:
    out_dir = _date_dir(cfg.CLOSE_ODDS_DIR, run_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"
    return _write_snapshot(out_path, matched_odds_df)


def _attach_pick_side_fields(
    base_df: pd.DataFrame,
    player_market_col: str,
    opp_market_col: str,
    player_odds_col: str,
    opp_odds_col: str,
    prob_out_col: str,
    odds_out_col: str,
) -> pd.DataFrame:
    out = base_df.copy()
    out[prob_out_col] = np.where(out["side"] == "player", out[player_market_col], out[opp_market_col])
    out[odds_out_col] = np.where(out["side"] == "player", out[player_odds_col], out[opp_odds_col])
    return out


def attach_line_movement(scored_df: pd.DataFrame, opening_snapshot_path: Path) -> pd.DataFrame:
    if not opening_snapshot_path.exists():
        return scored_df.copy()
    opening = pd.read_csv(opening_snapshot_path)
    merged = scored_df.merge(
        opening[
            [
                "match_id",
                "player_decimal_odds",
                "opp_decimal_odds",
                "player_market_prob",
                "opp_market_prob",
                "captured_utc",
            ]
        ].rename(
            columns={
                "player_decimal_odds": "opening_player_decimal_odds",
                "opp_decimal_odds": "opening_opp_decimal_odds",
                "player_market_prob": "opening_player_market_prob",
                "opp_market_prob": "opening_opp_market_prob",
                "captured_utc": "opening_snapshot_utc",
            }
        ),
        on="match_id",
        how="left",
    )
    merged = _attach_pick_side_fields(
        merged,
        "player_market_prob",
        "opp_market_prob",
        "player_decimal_odds",
        "opp_decimal_odds",
        "current_pick_market_prob",
        "current_pick_decimal_odds",
    )
    merged = _attach_pick_side_fields(
        merged,
        "opening_player_market_prob",
        "opening_opp_market_prob",
        "opening_player_decimal_odds",
        "opening_opp_decimal_odds",
        "opening_pick_market_prob",
        "opening_pick_decimal_odds",
    )
    merged["line_move_for_pick"] = merged["current_pick_market_prob"] - merged["opening_pick_market_prob"]
    merged["beat_open"] = merged["line_move_for_pick"] > 0
    return merged


def finalize_close_snapshot(run_date: str, close_snapshot_path: Path) -> dict[str, int]:
    ledger_path = cfg.SHADOW_LEDGER_PATH
    if not ledger_path.exists():
        raise FileNotFoundError(f"Shadow ledger not found at {ledger_path}")
    if not close_snapshot_path.exists():
        raise FileNotFoundError(f"Close snapshot not found at {close_snapshot_path}")

    ledger = pd.read_csv(ledger_path)
    if ledger.empty:
        raise ValueError("Shadow ledger exists but is empty.")
    if "run_date" not in ledger.columns:
        raise ValueError("Shadow ledger is missing run_date.")

    target_mask = ledger["run_date"].astype(str) == run_date
    target_count = int(target_mask.sum())
    if target_count == 0:
        raise ValueError(f"No shadow ledger rows found for run_date={run_date}.")

    close_snapshot = pd.read_csv(close_snapshot_path)
    close_work_cols = [
        "_close_snapshot_utc_new",
        "closing_player_decimal_odds",
        "closing_opp_decimal_odds",
        "closing_player_market_prob",
        "closing_opp_market_prob",
    ]
    target_ledger = ledger.loc[target_mask].drop(
        columns=close_work_cols,
        errors="ignore",
    )
    merged = target_ledger.merge(
        close_snapshot[
            [
                "match_id",
                "player_decimal_odds",
                "opp_decimal_odds",
                "player_market_prob",
                "opp_market_prob",
                "captured_utc",
            ]
        ].rename(
            columns={
                "player_decimal_odds": "closing_player_decimal_odds",
                "opp_decimal_odds": "closing_opp_decimal_odds",
                "player_market_prob": "closing_player_market_prob",
                "opp_market_prob": "closing_opp_market_prob",
                "captured_utc": "_close_snapshot_utc_new",
            }
        ),
        on="match_id",
        how="left",
    )
    merged["close_snapshot_utc"] = merged["_close_snapshot_utc_new"]
    merged = _attach_pick_side_fields(
        merged,
        "closing_player_market_prob",
        "closing_opp_market_prob",
        "closing_player_decimal_odds",
        "closing_opp_decimal_odds",
        "closing_pick_market_prob",
        "closing_pick_decimal_odds",
    )
    merged["line_move_open_to_close"] = (
        merged["closing_pick_market_prob"] - merged["opening_pick_market_prob"]
    )
    merged["line_move_pick_to_close"] = (
        merged["closing_pick_market_prob"] - merged["current_pick_market_prob"]
    )
    merged["beat_close"] = merged["line_move_pick_to_close"] > 0
    merged = merged.drop(columns=close_work_cols, errors="ignore")

    for col in merged.columns:
        if col not in ledger.columns:
            ledger[col] = pd.Series([np.nan] * len(ledger), dtype="object")
        elif merged[col].dtype == "object" or str(merged[col].dtype).startswith("bool"):
            ledger[col] = ledger[col].astype("object")
        ledger.loc[target_mask, col] = merged[col].tolist()
    ledger.to_csv(ledger_path, index=False)

    matched_close_rows = int(merged["closing_pick_market_prob"].notna().sum())
    beat_close_rows = int((merged["beat_close"] == True).sum())  # noqa: E712
    return {
        "target_rows": target_count,
        "matched_close_rows": matched_close_rows,
        "beat_close_rows": beat_close_rows,
    }


def record_shadow_ledger(run_date: str, run_id: str, scored_df: pd.DataFrame) -> int:
    cfg.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = cfg.SHADOW_LEDGER_PATH
    rows = scored_df.copy()
    rows["run_id"] = run_id
    rows["run_date"] = run_date
    keep_cols = [
        "run_id",
        "run_date",
        "match_id",
        "player_name",
        "opp_name",
        "tourney_name",
        "surface",
        "round",
        "side",
        "eligible",
        "governance_block",
        "odds_provider",
        "odds_match_confidence",
        "edge",
        "model_prob",
        "market_prob",
        "decimal_odds",
        "opening_pick_market_prob",
        "current_pick_market_prob",
        "opening_pick_decimal_odds",
        "current_pick_decimal_odds",
        "line_move_for_pick",
        "beat_open",
        "close_snapshot_utc",
        "closing_pick_market_prob",
        "closing_pick_decimal_odds",
        "line_move_open_to_close",
        "line_move_pick_to_close",
        "beat_close",
    ]
    for col in keep_cols:
        if col not in rows.columns:
            rows[col] = np.nan
    append_df = rows[keep_cols].copy()
    if ledger_path.exists():
        existing = pd.read_csv(ledger_path)
        if not existing.empty and {"run_date", "match_id", "side"}.issubset(existing.columns):
            keys = append_df[["run_date", "match_id", "side"]].astype(str)
            existing_keys = existing[["run_date", "match_id", "side"]].astype(str)
            duplicate_mask = existing_keys.apply(tuple, axis=1).isin(keys.apply(tuple, axis=1))
            existing = existing.loc[~duplicate_mask].copy()
        append_df = pd.concat([existing, append_df], ignore_index=True)
    append_df.to_csv(ledger_path, index=False)
    return len(rows)
