"""
Tennis pipeline runner — executes all phases in the recommended build order.

Usage:
  python run_pipeline.py [--phase PHASE]

Phases:
  1  data       Download Sackmann match + ranking CSVs
  2  odds       Download tennis-data.co.uk odds files
  3  features   Build ELO + rolling feature matrix
  4  train      Train XGBoost + calibration
  5  oos        Run OOS Kelly simulation
  all           Run phases 1 → 5 in sequence (skips phases already cached)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def phase_data() -> None:
    print("=" * 60)
    print("Phase 1: Downloading Sackmann data")
    print("=" * 60)
    from src.data.tennis_sackmann import main
    main()


def phase_odds() -> None:
    print("=" * 60)
    print("Phase 2: Downloading tennis-data.co.uk odds")
    print("=" * 60)
    from src.data.tennis_odds import main
    main()


def phase_features() -> None:
    print("=" * 60)
    print("Phase 3: Building features")
    print("=" * 60)
    from src.features.tennis_engineer import main
    main()


def phase_train() -> None:
    print("=" * 60)
    print("Phase 4: Training model")
    print("=" * 60)
    from src.model.tennis_train import main
    main()


def phase_oos() -> None:
    print("=" * 60)
    print("Phase 5: OOS simulation")
    print("=" * 60)
    from src.model.tennis_oos_report import main
    main()


_PHASES = {
    "1": phase_data,
    "data": phase_data,
    "2": phase_odds,
    "odds": phase_odds,
    "3": phase_features,
    "features": phase_features,
    "4": phase_train,
    "train": phase_train,
    "5": phase_oos,
    "oos": phase_oos,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tennis ML pipeline runner.")
    parser.add_argument(
        "--phase",
        default="all",
        help="Phase to run: 1/data, 2/odds, 3/features, 4/train, 5/oos, or 'all'",
    )
    args = parser.parse_args()

    if args.phase == "all":
        for fn in [phase_data, phase_odds, phase_features, phase_train, phase_oos]:
            fn()
    elif args.phase in _PHASES:
        _PHASES[args.phase]()
    else:
        print(f"Unknown phase: {args.phase}")
        sys.exit(1)


if __name__ == "__main__":
    main()
