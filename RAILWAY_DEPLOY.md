# Railway Deploy

This repo is set up to run the tennis shadow pipeline on Railway using one cron
service:

- `tennis-shadow-daily`

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
- one service built from this repo

The service must mount the volume.

Recommended mount path:

- `/app/persist`

## Why The Shared Volume Matters

The pipeline writes:

- runtime ledger files
- opening/current/close odds snapshots
- run artifacts
- an extended recent-history parquet

Those should survive container restarts.

The daily shadow and close snapshot must use the same volume because the close
snapshot updates `shadow_ledger.csv` created by the opening shadow run. Railway
volumes are attached to a service, so the production setup uses one cron service
that runs both phases in sequence instead of two separate cron services.

The image also contains a seed copy of the immutable runtime prerequisites:

- `matches.parquet`
- `rankings.parquet`
- `data/tennis/model/*`
- `atp_players.csv`

At service start, `scripts/prepare_runtime_data.py` copies those seed artifacts
into `TENNIS_DATA_DIR` only if they are missing.

## Environment Variables

Set these on the service:

```bash
ODDS_API_KEY=...
TENNIS_DATA_DIR=/app/persist/data/tennis
TENNIS_RUNS_DIR=/app/persist/runs/tennis
TENNIS_ENV=shadow
PYTHONUNBUFFERED=1
```

Optional:

```bash
TENNIS_CLOSE_SNAPSHOT_UTC=03:00
TENNIS_UPLOAD_TO_GCS=true
TENNIS_OUTPUT_BUCKET=...
TENNIS_NOTIFICATION_WEBHOOK=...
```

## Service Command

The service command runs the opening shadow job, waits until the close snapshot
time for the same UTC run date, then runs the close snapshot against the same
mounted volume:

```bash
bash scripts/railway_shadow.sh
```

## Cron Schedules

Railway cron is `UTC`.

The shadow run should happen just after the UTC match day starts, not in the
Denver morning. Tennis is global, and a `7:00 AM` Denver run is already after
many European starts and can miss earlier Asia/Australia matches entirely.

Recommended schedules:

- Daily shadow + close snapshot: `5 0 * * *`

Important:

- `5 0 * * *` runs at `00:05 UTC`, which is `6:05 PM` Denver on the previous
  local date during daylight time (`MDT`, `UTC-6`) and `5:05 PM` during standard
  time (`MST`, `UTC-7`).
- The runner defaults `run_date` from UTC, so this is the cleanest daily boundary
  for the current pipeline.
- `scripts/run_shadow_and_close.py` defaults the close snapshot to `03:00 UTC`.
  Set `TENNIS_CLOSE_SNAPSHOT_UTC=HH:MM` if this should move.
- Railway cron settings live in Railway unless you add config-as-code; update the
  service schedule there if it was already created with the older `0 13 * * *`
  recommendation.

## Railway UI Steps

1. Create a new project from this repo.
2. Add a volume and mount it at `/app/persist`.
3. Create the service:
   `tennis-shadow-daily`
4. Set its start command to:
   `bash scripts/railway_shadow.sh`
5. Set its cron schedule to:
   `5 0 * * *`
6. Add the environment variables.
7. Trigger one manual deploy/run and verify output under the
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
