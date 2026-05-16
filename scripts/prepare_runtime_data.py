#!/usr/bin/env python3
"""Seed Railway/runtime data directories from bundled image artifacts if needed."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists() or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    seed_root = Path(os.getenv("TENNIS_SEED_DATA_DIR", "/opt/tennis-seed/data/tennis"))
    target_root = Path(os.getenv("TENNIS_DATA_DIR", "/app/data/tennis")).expanduser()

    # Seed only immutable runtime prerequisites. Ledger and run outputs are created
    # on demand inside the writable target directory / volume.
    for rel in [
        Path("features/matches.parquet"),
        Path("features/rankings.parquet"),
        Path("model"),
        Path("raw/atp_players.csv"),
    ]:
        _copy_if_missing(seed_root / rel, target_root / rel)

    (target_root / "ledger").mkdir(parents=True, exist_ok=True)
    print(f"Runtime data ready at {target_root}")


if __name__ == "__main__":
    main()
