#!/usr/bin/env python3
"""Seed Railway/runtime data directories from bundled image artifacts if needed."""

from __future__ import annotations

import os
import shutil
from hashlib import sha256
from pathlib import Path


def _file_digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_digest(path: Path) -> str:
    h = sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        h.update(str(item.relative_to(path)).encode("utf-8"))
        h.update(b"\0")
        h.update(_file_digest(item).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def _is_current(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return False
    if src.is_dir() != dst.is_dir():
        return False
    if src.is_dir():
        return _tree_digest(src) == _tree_digest(dst)
    return _file_digest(src) == _file_digest(dst)


def _copy_if_missing_or_changed(src: Path, dst: Path) -> None:
    if not src.exists() or _is_current(src, dst):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
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
        Path("ledger/extended_matches.parquet"),
        Path("ledger/extended_matches.date"),
        Path("model"),
        Path("raw/atp_players.csv"),
    ]:
        _copy_if_missing_or_changed(seed_root / rel, target_root / rel)

    (target_root / "ledger").mkdir(parents=True, exist_ok=True)
    print(f"Runtime data ready at {target_root}")


if __name__ == "__main__":
    main()
