"""Supported CLI entrypoints for the tennis pipeline."""

from __future__ import annotations

import argparse
import json

from src.features.tennis_engineer import get_feature_columns, load_features
from src.model.tennis_backtest_2026_ytd import build_2026_ytd_backtest
from src.model.tennis_clv import build_clv_panel
from src.model.tennis_bundle import (
    create_candidate_bundle,
    evaluate_promotion_gates,
    promote_bundle,
)
from src.model.tennis_holdout_suite import run_holdout_suite
from src.model.tennis_oos_report import build_oos_report
from src.model.tennis_recent_holdout import run_recent_holdout
from src.model.tennis_train import get_feature_importance, train
from src.pipeline.tennis_daily_run import run_daily


def tennis_train_main() -> None:
    parser = argparse.ArgumentParser(description="Train and optionally promote the tennis model bundle.")
    parser.add_argument("--skip-promotion", action="store_true", help="Do not promote the candidate bundle.")
    args = parser.parse_args()

    print("Loading features...")
    features = load_features()
    model, scaler, metrics = train(features)
    get_feature_importance(model, get_feature_columns())

    bundle_dir = create_candidate_bundle()
    oos_summary = build_oos_report(use_holdout=True, output_tag=bundle_dir.name)
    ytd_summary = build_2026_ytd_backtest(output_tag=bundle_dir.name)
    gate_report = evaluate_promotion_gates(metrics, oos_summary, ytd_summary)

    summary = {
        "bundle_dir": str(bundle_dir),
        "gate_report": gate_report,
        "oos_summary": oos_summary,
        "ytd_summary": ytd_summary,
    }
    if not args.skip_promotion and gate_report["passed"]:
        promoted_dir = promote_bundle(bundle_dir, gate_report, oos_summary, ytd_summary)
        summary["promoted_dir"] = str(promoted_dir)
    elif not gate_report["passed"]:
        print("Promotion gates failed; candidate bundle was not promoted.")

    print(json.dumps(summary, indent=2, default=str))


def tennis_backtest_main() -> None:
    parser = argparse.ArgumentParser(description="Run historical OOS backtests.")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--production", action="store_true", help="Use production model instead of holdout.")
    parser.add_argument("--output-tag")
    args = parser.parse_args()

    summary = build_oos_report(
        oos_years=args.years,
        use_holdout=not args.production,
        output_tag=args.output_tag,
    )
    print(json.dumps(summary, indent=2, default=str))


def tennis_backtest_2026_ytd_main() -> None:
    summary = build_2026_ytd_backtest()
    print(json.dumps(summary, indent=2, default=str))


def tennis_daily_run_main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily shadow-mode tennis job.")
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--mode", default="shadow", choices=["shadow", "close_snapshot"])
    args = parser.parse_args()
    summary = run_daily(run_date=args.run_date, mode=args.mode)
    print(json.dumps(summary, indent=2, default=str))


def tennis_clv_main() -> None:
    parser = argparse.ArgumentParser(description="Build the Pinnacle CLV panel for holdout years.")
    parser.add_argument("--years", nargs="+", type=int, default=None)
    args, _ = parser.parse_known_args()
    panel = build_clv_panel(years=args.years)
    print(json.dumps(panel, indent=2, default=str))


def tennis_holdout_suite_main() -> None:
    parser = argparse.ArgumentParser(description="Run independent per-year tennis holdout evaluations.")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--earliest-train-year", type=int, default=2008)
    args = parser.parse_args()
    summary = run_holdout_suite(
        holdout_years=args.years,
        n_trials=args.n_trials,
        earliest_train_year=args.earliest_train_year,
    )
    print(json.dumps(summary, indent=2, default=str))


def tennis_recent_holdout_main() -> None:
    parser = argparse.ArgumentParser(description="Run a TennisMyLife-enriched 2025 holdout evaluation.")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--val-year", type=int, default=2024)
    parser.add_argument("--no-challenger", action="store_true")
    parser.add_argument("--skip-odds-refresh", action="store_true")
    args = parser.parse_args()
    summary = run_recent_holdout(
        train_years=list(range(2008, args.val_year)),
        val_year=args.val_year,
        test_year=args.test_year,
        n_trials=args.n_trials,
        include_challenger=not args.no_challenger,
        refresh_odds=not args.skip_odds_refresh,
    )
    print(json.dumps(summary, indent=2, default=str))
