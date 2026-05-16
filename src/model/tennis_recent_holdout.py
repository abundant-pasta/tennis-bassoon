"""Recent-year holdout evaluation with TennisMyLife-enriched 2025 coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config_tennis as cfg
from src.data.tennis_sackmann import load_matches
from src.data.tennis_tml import load_tml_player_rows
from src.data.tennis_odds import load_all_odds, save_odds
from src.features.tennis_engineer import build_features
from src.model.tennis_oos_report import build_oos_report
from src.model.tennis_train import train


def _artifact_dir(label: str) -> Path:
    return cfg.HOLDOUTS_DIR / label


def build_recent_features(
    recent_years: list[int] | None = None,
    include_challenger: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    recent_years = recent_years or [2025]

    if verbose:
        print("Loading historical player-perspective match DB through 2024...")
    hist = load_matches()
    hist = hist[hist["tourney_date"].dt.year <= 2024].copy()

    if verbose:
        print(f"Loading TennisMyLife rows for {recent_years}...")
    recent = load_tml_player_rows(
        recent_years,
        include_challenger=include_challenger,
        verbose=verbose,
    )
    if recent.empty:
        raise RuntimeError(f"No TennisMyLife rows found for years {recent_years}.")

    combined = pd.concat([hist, recent], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["match_id", "player_id", "opp_id"], keep="last")

    if verbose:
        print(f"Building features over {len(combined)} player-match rows...")
    return build_features(combined, verbose=verbose)


def recent_coverage_summary(features: pd.DataFrame, year: int = 2025) -> dict:
    subset = features[pd.to_datetime(features["tourney_date"]).dt.year == year].copy()
    if subset.empty:
        raise RuntimeError(f"No feature rows found for {year}.")

    cols = [
        "spw_20",
        "rpw_20",
        "ace_rate_20",
        "bp_save_pct_20",
        "spw_surf10",
        "rpw_surf10",
    ]
    coverage = {
        col: round(float(subset[col].notna().mean()), 4)
        for col in cols
        if col in subset.columns
    }
    raw_stat_cols = [
        "ace",
        "df",
        "svpt",
        "first_in",
        "first_won",
        "second_won",
        "bp_saved",
        "bp_faced",
    ]
    raw_stats = {
        col: round(float(subset[col].notna().mean()), 4)
        for col in raw_stat_cols
        if col in subset.columns
    }
    return {
        "year": year,
        "player_rows": int(len(subset)),
        "feature_coverage": coverage,
        "raw_stat_coverage": raw_stats,
    }


def run_recent_holdout(
    train_years: list[int] | None = None,
    val_year: int = 2024,
    test_year: int = 2025,
    n_trials: int = cfg.OPTUNA_TRIALS,
    include_challenger: bool = True,
    refresh_odds: bool = True,
) -> dict:
    train_years = train_years or list(range(2008, 2024))
    features = build_recent_features(
        recent_years=[test_year],
        include_challenger=include_challenger,
        verbose=True,
    )

    coverage = recent_coverage_summary(features, year=test_year)

    if refresh_odds:
        print("Refreshing tennis-data odds cache through recent years...")
        odds = load_all_odds(verbose=True)
        if not odds.empty:
            save_odds(odds)

    label = f"recent_holdout_{test_year}"
    artifact_dir = _artifact_dir(label)
    _, _, metrics = train(
        features,
        n_trials=n_trials,
        train_years=train_years,
        val_year=val_year,
        oos_years=[test_year],
        artifact_dir=artifact_dir,
        skip_production_refit=True,
    )
    report = build_oos_report(
        oos_years=[test_year],
        use_holdout=False,
        output_tag=label,
        model_path=artifact_dir / "model.pkl",
        scaler_path=artifact_dir / "scaler.pkl",
        medians_path=artifact_dir / "medians.pkl",
        features_df=features,
    )

    summary = {
        "train_years": train_years,
        "val_year": val_year,
        "test_year": test_year,
        "n_trials": n_trials,
        "include_challenger": include_challenger,
        "coverage": coverage,
        "train_metrics": metrics,
        "oos_report": report,
        "artifact_dir": str(artifact_dir),
    }
    out_path = cfg.OOS_DIR / f"recent_holdout_summary_{test_year}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Recent holdout summary saved → {out_path}")
    return summary

