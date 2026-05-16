"""Independent holdout-year training and evaluation suite for tennis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config_tennis as cfg
from src.features.tennis_engineer import load_features
from src.model.tennis_oos_report import build_oos_report
from src.model.tennis_train import train


def _artifact_dir(year: int) -> Path:
    return cfg.HOLDOUTS_DIR / str(year)


def _artifact_paths(year: int) -> dict[str, Path]:
    base = _artifact_dir(year)
    return {
        "dir": base,
        "model": base / "model.pkl",
        "scaler": base / "scaler.pkl",
        "medians": base / "medians.pkl",
        "metadata": base / "metadata.json",
    }


def run_holdout_suite(
    holdout_years: list[int],
    n_trials: int = cfg.OPTUNA_TRIALS,
    earliest_train_year: int = 2008,
) -> dict:
    features = load_features()
    rows: list[dict] = []

    for year in holdout_years:
        val_year = year - 1
        train_years = list(range(earliest_train_year, val_year))
        if len(train_years) < 3:
            raise RuntimeError(f"Not enough training history available for holdout year {year}.")

        print("=" * 72)
        print(f"Training independent holdout artifact for {year} "
              f"(train={train_years[0]}-{train_years[-1]}, val={val_year}, test={year})")
        print("=" * 72)
        paths = _artifact_paths(year)
        _, _, metrics = train(
            features,
            n_trials=n_trials,
            train_years=train_years,
            val_year=val_year,
            oos_years=[year],
            artifact_dir=paths["dir"],
            skip_production_refit=True,
        )
        report = build_oos_report(
            oos_years=[year],
            use_holdout=False,
            output_tag=f"holdout_{year}",
            model_path=paths["model"],
            scaler_path=paths["scaler"],
            medians_path=paths["medians"],
        )
        rows.append(
            {
                "holdout_year": year,
                "train_years": train_years,
                "val_year": val_year,
                "val_auc_roc": metrics.get("val_auc_roc"),
                "val_log_loss_selected": metrics.get("val_log_loss_selected"),
                "oos_auc": report.get("auc_oos"),
                "flat_roi": report.get("flat_bet", {}).get("flat_roi"),
                "flat_units": report.get("flat_bet", {}).get("units_profit"),
                "total_bets": report.get("total_bets"),
                "win_rate": report.get("win_rate"),
                "artifact_dir": str(paths["dir"]),
                "report_path": str(cfg.OOS_DIR / f"oos_report_{year}_holdout_{year}.json"),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary = {
        "holdout_years": holdout_years,
        "n_trials": n_trials,
        "rows": rows,
        "avg_oos_auc": round(float(summary_df["oos_auc"].mean()), 4) if not summary_df.empty else None,
        "avg_flat_roi": round(float(summary_df["flat_roi"].mean()), 4) if not summary_df.empty else None,
    }
    out_json = cfg.OOS_DIR / "holdout_suite_summary.json"
    out_csv = cfg.OOS_DIR / "holdout_suite_summary.csv"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    summary_df.to_csv(out_csv, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent per-year tennis holdout evaluations.")
    parser.add_argument("--years", nargs="+", type=int, default=cfg.OOS_YEARS)
    parser.add_argument("--n-trials", type=int, default=cfg.OPTUNA_TRIALS)
    parser.add_argument("--earliest-train-year", type=int, default=2008)
    args = parser.parse_args()

    summary = run_holdout_suite(
        holdout_years=args.years,
        n_trials=args.n_trials,
        earliest_train_year=args.earliest_train_year,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
