# Railway Deploy

This repo is set up to run the tennis shadow pipeline on Railway using two cron
services from the same source repo:

- `tennis-shadow-morning`
- `tennis-shadow-close`

Railway cron services run the service start command on a schedule, then exit.
Railway cron schedules are configured in `UTC`.

Sources:
- [Railway Cron Jobs](https://docs.railway.com/cron-jobs)
- [Railway Volumes](https://docs.railway.com/volumes)
- [Railway Config as Code](https://docs.railway.com/config-as-code/reference)
- [Railway Dockerfiles](https://docs.railway.com/builds/dockerfiles)

## What To Create

Create one Railway project with:

- one attached volume
- two services built from this repo

Both services should mount the same volume.

Recommended mount path:

- `/app/persist`

## Why The Shared Volume Matters

The pipeline writes:

- runtime ledger files
- opening/current/close odds snapshots
- run artifacts
- an extended recent-history parquet

Those should survive container restarts.

The image also contains a seed copy of the immutable runtime prerequisites:

- `matches.parquet`
- `rankings.parquet`
- `data/tennis/model/*`
- `atp_players.csv`

At service start, `scripts/prepare_runtime_data.py` copies those seed artifacts
into `TENNIS_DATA_DIR` only if they are missing.

## Shared Environment Variables

Set these on both services:

```bash
ODDS_API_KEY=...
TENNIS_DATA_DIR=/app/persist/data/tennis
TENNIS_RUNS_DIR=/app/persist/runs/tennis
TENNIS_ENV=shadow
PYTHONUNBUFFERED=1
```

Optional:

```bash
TENNIS_UPLOAD_TO_GCS=true
TENNIS_OUTPUT_BUCKET=...
TENNIS_NOTIFICATION_WEBHOOK=...
```

## Service Commands

Morning shadow picks service:

```bash
bash scripts/railway_shadow.sh
```

Evening close snapshot service:

```bash
bash scripts/railway_close_snapshot.sh
```

## Cron Schedules

Railway cron is `UTC`.

If you want the runs at approximately `7:00 AM` and `9:00 PM` Denver time during
daylight time (`MDT`, `UTC-6`), use:

- Morning shadow: `0 13 * * *`
- Close snapshot: `0 3 * * *`

Important:

- When Denver moves to standard time (`MST`, `UTC-7`), those same UTC schedules
  will fire one hour later locally.
- If you want fixed local wall-clock times year-round, you will need to update
  the UTC cron expressions at DST boundaries.

## Railway UI Steps

1. Create a new project from this repo.
2. Add a volume and mount it at `/app/persist`.
3. Create the first service:
   `tennis-shadow-morning`
4. Set its start command to:
   `bash scripts/railway_shadow.sh`
5. Set its cron schedule to:
   `0 13 * * *`
6. Duplicate the service or create a second one from the same repo:
   `tennis-shadow-close`
7. Set its start command to:
   `bash scripts/railway_close_snapshot.sh`
8. Set its cron schedule to:
   `0 3 * * *`
9. Add the shared environment variables to both services.
10. Trigger one manual deploy/run for each service and verify output under the
    mounted paths.

## Expected Persistent Paths

With the recommended env vars:

- data root: `/app/persist/data/tennis`
- runs root: `/app/persist/runs/tennis`

Useful subpaths:

- `/app/persist/data/tennis/ledger/shadow_ledger.csv`
- `/app/persist/data/tennis/ledger/opening_odds/`
- `/app/persist/data/tennis/ledger/odds_snapshots/`
- `/app/persist/runs/tennis/YYYY/MM/DD/`

## Manual Smoke Test

Inside a Railway service shell, or by temporarily changing the service start
command:

```bash
python3 scripts/prepare_runtime_data.py
python3 scripts/run_shadow.py --date 2026-05-15
python3 scripts/run_shadow.py --date 2026-05-15 --mode close_snapshot
```

## Current Limitation

The recent-history source can lag a few days. The runner is now wired to use the
enriched recent-history path automatically, but if the source feed is behind, the
latest few days of completed matches may still be missing from the rolling
history context.
