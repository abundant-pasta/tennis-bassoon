"""Tennis pipeline configuration — kept separate from any MLB config."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TENNIS_DATA_DIR = Path(os.getenv("TENNIS_DATA_DIR", "data/tennis")).expanduser()
RAW_DIR = TENNIS_DATA_DIR / "raw"
FEATURES_DIR = TENNIS_DATA_DIR / "features"
MODEL_DIR = TENNIS_DATA_DIR / "model"
OOS_DIR = TENNIS_DATA_DIR / "oos"
LEDGER_DIR = TENNIS_DATA_DIR / "ledger"
RUNS_DIR = Path(os.getenv("TENNIS_RUNS_DIR", "runs/tennis")).expanduser()
HOLDOUTS_DIR = MODEL_DIR / "holdouts"
ODDS_SNAPSHOT_DIR = LEDGER_DIR / "odds_snapshots"
OPENING_ODDS_DIR = LEDGER_DIR / "opening_odds"
CLOSE_ODDS_DIR = LEDGER_DIR / "close_odds"

MODEL_PATH = MODEL_DIR / "tennis_xgb.pkl"
SCALER_PATH = MODEL_DIR / "tennis_scaler.pkl"
HOLDOUT_MODEL_PATH = MODEL_DIR / "tennis_xgb_holdout.pkl"
HOLDOUT_SCALER_PATH = MODEL_DIR / "tennis_scaler_holdout.pkl"
MEDIANS_PATH = MODEL_DIR / "tennis_feature_medians.pkl"
HOLDOUT_MEDIANS_PATH = MODEL_DIR / "tennis_feature_medians_holdout.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "tennis_model_metadata.json"
PRODUCTION_METADATA_PATH = MODEL_DIR / "tennis_production_metadata.json"
MODEL_BUNDLES_DIR = MODEL_DIR / "bundles"
PROMOTED_MODEL_DIR = MODEL_DIR / "promoted"
PROMOTED_CURRENT_DIR = PROMOTED_MODEL_DIR / "current"
PROMOTION_MANIFEST_PATH = PROMOTED_MODEL_DIR / "promotion_manifest.json"
MODEL_FEATURE_CONTRACT_PATH = MODEL_DIR / "feature_contract.json"

# ---------------------------------------------------------------------------
# Data scope
# ---------------------------------------------------------------------------
# ATP tour-level (A) + Challengers (C) only; skip Futures (F) and Qualifying (Q)
INCLUDE_TOURNEY_LEVELS = {"A", "C", "G", "M", "F"}  # G=Slam, M=Masters, A=250/500, C=Challenger
EXCLUDE_TOURNEY_LEVELS: set[str] = set()  # Futures handled by min-rank filter

# Only include matches where the player is ranked inside this threshold.
# Drops unranked Futures players with no feature history.
MAX_PLAYER_RANK = 500

# Minimum completed matches before a player's rolling stats are trusted.
MIN_MATCHES_PLAYED = 20

# Years to use for training and OOS.
TRAIN_YEARS = list(range(2008, 2021))   # 2008–2020 inclusive
VAL_YEAR = 2021
OOS_YEARS = [2022, 2023, 2024]

# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------
ELO_START = 1500.0
ELO_K_INITIAL = 32.0   # first 30 matches
ELO_K_STABLE = 16.0    # after 30 matches
ELO_K_THRESHOLD = 30   # matches before K decays

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
ROLLING_ALL_SURFACE = 20    # rolling window: all surfaces
ROLLING_SAME_SURFACE = 10   # rolling window: same surface only
H2H_SHRINK_K = 4.0          # Bayesian shrinkage for H2H

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
OPTUNA_TRIALS = 50

# ---------------------------------------------------------------------------
# OOS / Kelly
# ---------------------------------------------------------------------------
EDGE_THRESHOLD = 0.06       # minimum edge to consider a bet
KELLY_FRACTION = 0.25       # fractional Kelly multiplier
MAX_STAKE_FRACTION = 0.02   # cap: 2% of bankroll per bet (Kelly with realistic sizing)
MAX_DAILY_EXPOSURE = 0.10   # cap total same-day stake to 10% of bankroll
STARTING_BANKROLL = 1000.0

# Kelly sizing probability bounds. These are only used for stake sizing, not
# edge selection, so we do not fabricate underdog/favorite edges by clipping.
KELLY_PROB_FLOOR = 0.05
KELLY_PROB_CEIL = 0.95
KELLY_EDGE_COMPRESSION_THRESHOLD = 0.10
KELLY_EDGE_COMPRESSION_FACTOR = 0.35
MAX_ODDS_DATE_DISTANCE_DAYS = 14
MAX_SNAPSHOT_AGE_HOURS = 30
MIN_FEATURE_COVERAGE = 0.65
MIN_ODDS_MATCH_CONFIDENCE = 0.75

# ---------------------------------------------------------------------------
# Governance thresholds
# ---------------------------------------------------------------------------
CHALLENGER_EXTREME_EDGE = 0.15      # block or halve Kelly on Challengers at this edge
MASTERS_EXTREME_EDGE = 0.12         # block very large Masters edges until recent calibration is cleaner
MIN_RANK_BOTH_PLAYERS = 200         # block if either player ranked beyond this
MIN_SURFACE_MATCHES = 5             # halve Kelly if < this on current surface
H2H_MIN_SAMPLE_KELLY_REDUCTION = 3  # shrink Kelly 20% if H2H < this many matches
KELLY_HALF_FRACTION = 0.5           # multiplier when halving Kelly
KELLY_H2H_REDUCTION = 0.8           # multiplier when H2H is thin

# Tournament level encoding
TOURNEY_LEVEL_ENCODE = {"G": 4, "M": 3, "A": 2, "C": 1, "F": 0}
ROUND_ENCODE = {"F": 6, "SF": 5, "QF": 4, "R16": 3, "R32": 2, "R64": 1, "R128": 0, "RR": 2, "BR": 1}
SURFACE_ENCODE = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
HAND_ENCODE = {"R": 0, "L": 1, "U": 2}
HAND_MATCHUP_ENCODE = {"RvR": 0, "RvL": 1, "LvR": 2, "LvL": 3}

# Grass early-tournament governance: block rounds R64 and R128 at grass events.
GRASS_EARLY_ROUND_BLOCK = {"R64", "R128"}

SHADOW_LEDGER_PATH = LEDGER_DIR / "shadow_ledger.csv"

# ---------------------------------------------------------------------------
# Production / promotion
# ---------------------------------------------------------------------------
PRODUCTION_TIMEZONE = "America/Denver"
PROMOTION_MIN_VAL_AUC = 0.70
PROMOTION_MAX_VAL_LOG_LOSS = 0.65
PROMOTION_MIN_OOS_FLAT_ROI = 0.0
PROMOTION_MIN_YTD_FLAT_ROI = 0.0
