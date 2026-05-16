# Tennis Bassoon

ATP tennis modeling and shadow-mode recommendation pipeline.

## Commands

```bash
python3 -m pip install -e .[dev]
tennis-train
tennis-backtest --years 2022 2023 2024
tennis-backtest-2026-ytd
tennis-holdout-suite --years 2022 2023 2024
tennis-daily-run --run-date 2026-04-28 --mode shadow
tennis-daily-run --run-date 2026-04-28 --mode close_snapshot
```

For live ATP odds via The Odds API, set:

```bash
TENNIS_SOURCE_ADAPTER=odds_api_tennis
ODDS_API_KEY=...
TENNIS_SCHEDULE_SOURCE_URI=/absolute/path/to/schedule.csv
```

Optional live-odds settings:
- `TENNIS_ODDS_API_BOOKMAKERS=draftkings,fanduel,betmgm,caesars,bet365,fanatics`
- `TENNIS_ODDS_API_REGIONS=us`
- `TENNIS_ODDS_API_SPORT_KEYS=tennis_atp_madrid_open`

## Daily Snapshot Contracts

Schedule CSV columns:
- `match_date`
- `tourney_name`
- `surface`
- `round`
- `player_name`
- `opp_name`
- optional `match_id`, `best_of`, `draw_size`, `player_rank`, `opp_rank`

Odds CSV columns:
- `match_date`
- `tourney_name`
- `surface`
- `round`
- `player_name`
- `opp_name`
- `player_decimal_odds`
- `opp_decimal_odds`
- optional `provider`

The Odds API adapter:
- keeps the existing schedule CSV input
- fetches live `h2h` odds for supported ATP tournaments
- builds a median-decimal consensus across returned bookmakers for each match
- falls back to unsupported-tournament review rows when a schedule event is not covered

## Production Notes

- Daily runs are shadow mode only.
- Promoted model bundles live under `data/tennis/model/promoted/current`.
- Each run writes immutable artifacts under `runs/tennis/YYYY/MM/DD/{run_id}/`.
- Shadow market tracking writes opening/current/close odds snapshots under `data/tennis/ledger/`.
- Shadow recommendations are appended to `data/tennis/ledger/shadow_ledger.csv`.
- `close_snapshot` mode finalizes same-day ledger rows with `beat_close` and line-move fields.
- Optional GCS upload is enabled with `TENNIS_OUTPUT_BUCKET` and `TENNIS_UPLOAD_TO_GCS=true`.
- If `ODDS_API_KEY` is not set in the current shell, the runner will also look for it in the sibling `laughing-bassoon/.env` or `laughing-bassoon/remote_config.env` files for local development.
