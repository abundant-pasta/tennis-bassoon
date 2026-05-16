#!/usr/bin/env bash
set -euo pipefail

python3 scripts/prepare_runtime_data.py
exec python3 scripts/run_shadow.py --mode close_snapshot
