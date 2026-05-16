# Tennis Betting Pipeline — Build Plan

## Context

This plan describes building a tennis ML betting pipeline modeled after an existing MLB system.
The MLB system uses XGBoost + walk-forward calibration, Kelly criterion sizing, governance rules
to suppress unreliable predictions, and an OOS simulation to validate before shadow betting.

The tennis pipeline should follow the same architecture. The immediate goal is to validate that
a model trained on Sackmann's historical data produces a believable edge distribution before
committing to any live data costs.

---

## Data Sources

### Historical (free, covers model training and OOS validation)

- **Jeff Sackmann ATP repo**: `https://github.com/JeffSackmann/tennis_atp`
  - `atp_matches_YYYY.csv` — tour-level matches 1991–present
  - `atp_matches_qual_chall_YYYY.csv` — Challengers 2008–present
  - `atp_matches_futures_YYYY.csv` — ITF Futures 2011–present
  - `atp_rankings_current.csv` + `atp_rankings_YYYYY.csv` — weekly ATP rankings
  - `atp_players.csv` — player metadata (DOB, hand, country)

- **Jeff Sackmann WTA repo**: `https://github.com/JeffSackmann/tennis_wta`
  - Same structure as ATP

- **tennis-data.co.uk**: Historical match results + closing odds (ATP 2000–present)
  - Used to construct the market implied probability column (equivalent to MLB soft-book odds)
  - Also gives Pinnacle odds for CLV analysis

Start with **ATP tour-level + Challengers only**. ITF Futures have thin odds data and noisy stats.

### Live data (deferred — do not solve until model validates)

The Sackmann repo is manually updated (days to weeks lag). For a shadow run you will eventually
need a live feed. Do not spend time or money on this until the OOS simulation looks compelling.
At that point, evaluate API-Tennis (RapidAPI, ~$10-20/month) or Goalserve (~$100/month) based
on Challenger coverage depth.

---

## Project Structure

Mirror the existing MLB layout:

```
src/
  data/
    tennis_sackmann.py     # Download/parse Sackmann CSVs, build match DB
    tennis_odds.py         # Parse tennis-data.co.uk odds files
  features/
    tennis_engineer.py     # Rolling stats, ELO, surface splits, H2H
  model/
    tennis_train.py        # XGBoost training + walk-forward calibration
    tennis_oos_report.py   # OOS simulation + Kelly backtest
    tennis_evaluate.py     # Calibration curves, edge buckets, CLV
  pipeline/
    tennis_results.py      # _build_recommendation, governance, Kelly sizing
    tennis_daily_run.py    # Morning run orchestration (stub until live data exists)
config_tennis.py           # All thresholds (keep separate from MLB config)
data/
  tennis/
    raw/                   # Sackmann CSV downloads (gitignored if large)
    features/              # Parquet feature cache per year
    model/                 # Saved model pkl + calibrator
    oos/                   # OOS simulation outputs
    ledger/                # Shadow bet ledger (when live)
```

---

## Phase 1 — Data Ingestion (`src/data/tennis_sackmann.py`)

### Match table schema

Each row is one match (not one player). Canonical columns:

| Column | Source |
|---|---|
| `match_id` | `tourney_id + match_num` |
| `tourney_date` | `tourney_date` (YYYYMMDD → date) |
| `tourney_name` | `tourney_name` |
| `surface` | `surface` (Hard / Clay / Grass / Carpet) |
| `tourney_level` | `tourney_level` (G=Slam, M=Masters, A=250/500, C=Challenger, F=Futures) |
| `draw_size` | `draw_size` |
| `round` | `round` (R128, R64, R32, R16, QF, SF, F) |
| `best_of` | `best_of` |
| `winner_id` | `winner_id` |
| `winner_name` | `winner_name` |
| `winner_hand` | `winner_hand` |
| `winner_ht` | `winner_ht` |
| `winner_age` | `winner_age` |
| `winner_rank` | `winner_rank` |
| `winner_rank_points` | `winner_rank_points` |
| `loser_id` | `loser_id` |
| `loser_name` | (same pattern) |
| `score` | `score` |
| `minutes` | `minutes` |
| `w_ace` | aces by winner |
| `w_df` | double faults by winner |
| `w_svpt` | serve points played by winner |
| `w_1stIn` | first serves in |
| `w_1stWon` | first serve points won |
| `w_2ndWon` | second serve points won |
| `w_SvGms` | serve games |
| `w_bpSaved` | break points saved |
| `w_bpFaced` | break points faced |
| `l_ace` … `l_bpFaced` | loser equivalents |

Normalize to **player-perspective rows** (one row per player per match, with `won` = 0/1)
for feature engineering. This simplifies rolling lookups.

### Rankings table

Parse all `atp_rankings_*.csv` files into a single table:
- `ranking_date`, `rank`, `player_id`, `points`
- Index by `(player_id, ranking_date)` for fast lookups

### Odds join

Parse tennis-data.co.uk files and join to match table on `(tourney_name, date, winner_name, loser_name)`.
Name matching will require fuzzy matching (same problem as MLB team name reconciliation).
Output columns: `winner_closing_odds`, `loser_closing_odds`, `pinnacle_winner_odds`, `pinnacle_loser_odds`.

---

## Phase 2 — Feature Engineering (`src/features/tennis_engineer.py`)

Features are computed as rolling lookups **as of match date** (no lookahead). Always use
a minimum games threshold (e.g., 20 matches) before trusting a rolling stat.

### ELO ratings (most important feature)

Implement surface-specific ELO:
- Separate K-factors per surface (grass converges more slowly — fewer matches)
- Separate ELO per surface: `elo_hard`, `elo_clay`, `elo_grass`, `elo_overall`
- Starting ELO: 1500 for all new players
- K-factor: start at 32, decay to 16 after 30 matches
- Use **pre-match ELO** only (update after match resolves)

ELO difference `elo_diff = player_elo - opponent_elo` on the relevant surface is likely
the single strongest feature.

### Serve/return rolling stats (last N matches on surface)

Compute rolling over last 20 matches (all surfaces) and last 10 matches (same surface):

- `spw` = service points won % = `(w_1stWon + w_2ndWon) / w_svpt`
- `rpw` = return points won % = `1 - opponent_spw`
- `ace_rate` = `w_ace / w_svpt`
- `df_rate` = `w_df / w_svpt`
- `first_serve_pct` = `w_1stIn / w_svpt`
- `first_serve_win_pct` = `w_1stWon / w_1stIn`
- `second_serve_win_pct` = `w_2ndWon / (w_svpt - w_1stIn)`
- `bp_conversion_pct` = break points converted / break points faced (opponent perspective)
- `bp_save_pct` = `w_bpSaved / w_bpFaced`

Compute differential features: `spw_diff`, `rpw_diff`, `ace_rate_diff`, etc.

### Recent form

- `win_pct_last_10` (all surfaces)
- `win_pct_last_10_surface` (same surface)
- `win_pct_last_5` (recency-weighted)
- `sets_win_pct_last_10` (tighter signal than match win %)
- `deciding_set_win_pct` (clutch signal)

### Head-to-head

- `h2h_wins`, `h2h_losses`, `h2h_win_pct` (all time)
- `h2h_win_pct_surface` (same surface only)
- `h2h_recent_win_pct` (last 3 years)
- `h2h_n` (sample size — low n → shrink toward 0.5)

Apply Bayesian shrinkage: `h2h_adj = (h2h_wins + 0.5 * shrink_k) / (h2h_n + shrink_k)`
with `shrink_k = 4`. H2H with < 3 matches should contribute very little.

### Tournament / context features

- `days_since_last_match` (rest/fatigue)
- `matches_last_14_days` (congestion)
- `tourney_level_encoded` (Slam=4, Masters=3, 500=2, 250=1, Challenger=0)
- `round_encoded` (F=6, SF=5, QF=4, R16=3, R32=2, R64=1, R128=0)
- `player_rank` (raw ATP rank, log-transformed)
- `rank_diff` = `player_rank - opponent_rank`
- `surface_encoded` (Hard=0, Clay=1, Grass=2)
- `player_age`, `age_diff`
- `hand_matchup` (RvR, RvL, LvL — left-handers have documented surface/matchup edges)

### Features NOT worth building initially

- Shot-type distributions (not in Sackmann main data)
- In-match momentum (requires point-by-point)
- Weather (tennis-data.co.uk doesn't have it)
- Prize money / ranking points on offer (collinear with tourney_level)

---

## Phase 3 — Model Training (`src/model/tennis_train.py`)

### Data split strategy

Use **walk-forward validation** matching the MLB approach:
- Training window: all data before season N
- Validation: season N (for hyperparameter tuning)
- OOS test: seasons N+1 onward
- Recommended: train on 2008–2020, validate 2021, OOS test 2022–2024

### Target variable

`won` (1 = player won the match). This is binary classification.
Predict `P(player wins)` — the model outputs a probability, not a side.

Frame each match as two rows (player A perspective, player B perspective) with `won` flipped.
Features are computed from player A's perspective (differentials handle directionality).
At inference time, generate one prediction per match and reconcile.

Alternatively: frame as one row per match, predict `P(home wins)` where "home" is assigned
as the higher-ranked player. This matches the MLB framing more closely and halves training data
size. Either works — one-row-per-match is cleaner.

### Model

XGBoost classifier, same hyperparameter search as MLB:
- Objective: `binary:logistic`
- Eval metric: `logloss`
- Key params to tune via Optuna: `max_depth`, `learning_rate`, `subsample`,
  `colsample_bytree`, `min_child_weight`, `n_estimators`

### Calibration

Apply isotonic regression calibration (same as MLB walk-forward calibration).
Tennis win probabilities tend to be well-calibrated from ELO alone; the model should
improve on ELO but calibration is still necessary for Kelly sizing.

### Expected baseline

Published ML models on ATP data achieve ~66-68% accuracy (vs. ~63% for ATP ranking baseline).
If the model can't beat ATP ranking accuracy on the OOS set, stop and investigate features
before proceeding.

---

## Phase 4 — OOS Simulation (`src/model/tennis_oos_report.py`)

Mirror `oos_report.py` from the MLB system. For each match in the OOS window:

1. Generate model win probability for each player
2. Compute market implied probability from closing odds (de-vig: multiplicative method)
3. Compute edge: `model_prob - market_implied_prob`
4. Apply governance rules (see Phase 5)
5. If edge > threshold and not governed: simulate Kelly bet
6. Track P&L, ROI, CLV

### Kelly sizing

- Full Kelly: `f = (p * b - q) / b` where `b = decimal_odds - 1`
- Use fractional Kelly: start at 0.25 (same as MLB)
- Cap individual bet at 5% of bankroll
- Starting bankroll: $1,000 (notional)

### Metrics to report

- Total bets, win rate, ROI, profit
- Edge bucket breakdown (6-8%, 8-10%, 10-12%, 12%+)
- Surface breakdown (do Clay edges close? Grass?)
- Tournament level breakdown (Challengers vs. tour-level)
- Calibration curve (model prob vs. actual win rate)
- CLV: did closing odds move toward model predictions?

---

## Phase 5 — Governance Rules (`config_tennis.py` + `tennis_results.py`)

These must be validated against OOS data before treating as final. Start with these
candidates and remove/adjust based on what the backtest shows:

### Hard gates (always pass)

- **No odds available**: can't size a bet without a price
- **Walkover / retirement mid-match**: not predictable pre-match; skip any match
  with `score` containing `W/O` or `RET` in the training filter AND flag in live pipeline
- **Qualifying rounds**: model trained on main draw + Challengers; qualifying stats
  are sparse and the player pool is different

### Governance blocks (equivalent to MLB extreme-edge rules)

- **Extreme edge on Challengers** (edge ≥ 15%): Challenger markets are thin;
  extreme model edge often means missing data, not real signal. Block or halve Kelly.
- **Low-rank opponent** (opponent rank > 200): model features become unreliable for
  players with very few tracked matches. Block if either player rank is missing.
- **Surface mismatch governance**: if a player has < 5 matches on the current surface
  in the rolling window, flag as data-sparse. Halve Kelly.
- **High retirement risk** (deferred): when live data available, flag players with
  recent injury news. Not possible from Sackmann alone.
- **Grass early-tournament block**: first 2 rounds at Wimbledon / grass tune-up events
  in June are historically noisy (players adjusting to surface). Consider blocking
  analogous to MLB early-season governance.
- **Best-of-5 vs best-of-3**: Slams are best-of-5; all other events are best-of-3.
  The model should either (a) train separate models or (b) include `best_of` as a feature.
  Do not mix without a feature — best-of-5 significantly changes upset probability.

### Kelly adjustments (not full blocks)

- **H2H sample < 3**: shrink Kelly by 20%
- **Player rank missing**: halve Kelly
- **Surface-sparse rolling window**: halve Kelly (< 5 same-surface matches)

---

## Phase 6 — Validation Checklist (before shadow run)

Do not proceed to a live shadow run until all of these pass:

- [ ] OOS ROI is positive across at least 2 full seasons
- [ ] Edge buckets show monotonic improvement (higher edge → higher ROI)
- [ ] CLV is positive (model is finding real market inefficiency, not data artifacts)
- [ ] Calibration curve is within ±3% of diagonal
- [ ] Challenger-only subset shows positive ROI (or you exclude Challengers)
- [ ] Clay/Hard/Grass subsets are individually reviewed — if one surface is dragging
      results, understand why before betting into it
- [ ] Governance rules improve OOS ROI vs. unfiltered (verify each rule pulls its weight)
- [ ] No single player or matchup is driving disproportionate P&L (data artifact check)

---

## Phase 7 — Shadow Run (future, deferred)

When validation passes and a live data feed is acquired:

1. Morning script pulls today's ATP/WTA schedule
2. For each match, fetch current rankings and rolling stats (via live API)
3. Run feature pipeline and model
4. Apply governance rules
5. Output picks with Kelly sizes to a ledger (no real money)
6. At match end, record result and CLV
7. Weekly review of shadow ledger vs. OOS expectations

The ledger schema should mirror the MLB `model_bet_ledger.csv` so the same
tracker UI can display both sports.

---

## Key Decisions / Open Questions

1. **ATP only vs. ATP + WTA**: Start ATP-only. WTA data has more gaps and separate
   market dynamics. Add WTA after ATP validates.

2. **Challenger inclusion**: Include in training data for volume, but consider separate
   governance thresholds. Challenger market efficiency is lower — could be good or bad
   (thinner markets mean both more edge and more noise).

3. **Separate models per surface vs. one model with surface features**: Start with one
   model + surface features. If calibration is poor on a specific surface, split.

4. **ELO implementation**: Build your own (simple, fully controlled) rather than importing
   a library. You need surface-specific ELO and the ability to replay history from scratch.

5. **Odds source**: tennis-data.co.uk covers ATP 2000–present with Pinnacle odds.
   This is sufficient for OOS simulation. The format is Excel/CSV per year per tour,
   requires annual download and parse.

---

## Recommended Build Order

1. `tennis_sackmann.py` — download and normalize Sackmann CSVs into a clean match DB
2. `tennis_engineer.py` — ELO ratings first, then rolling serve/return stats, then H2H
3. Exploratory analysis: check feature correlations, validate ELO tracks actual win rates
4. `tennis_train.py` — train model, check OOS accuracy vs. ATP ranking baseline
5. `tennis_odds.py` — join tennis-data.co.uk odds to match DB
6. `tennis_oos_report.py` — run full Kelly simulation, review validation checklist
7. `config_tennis.py` — tune governance thresholds based on OOS output
8. If validation passes: plan live data feed acquisition
