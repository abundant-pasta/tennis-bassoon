"""Artifact persistence helpers for local runs and optional GCS upload."""

from __future__ import annotations

from pathlib import Path
import json
import shutil

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_bytes(path: Path, data: bytes) -> Path:
    ensure_dir(path.parent)
    path.write_bytes(data)
    return path


def write_text(path: Path, text: str) -> Path:
    ensure_dir(path.parent)
    path.write_text(text)
    return path


def write_json(path: Path, payload: dict) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def write_csv(path: Path, df: pd.DataFrame) -> Path:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def write_parquet(path: Path, df: pd.DataFrame) -> Path:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path


def copy_file(src: Path, dest: Path) -> Path:
    ensure_dir(dest.parent)
    shutil.copy2(src, dest)
    return dest


def upload_directory_to_gcs(local_dir: Path, bucket_name: str, prefix: str) -> list[str]:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is required for GCS uploads. "
            "Install the optional gcp dependencies first."
        ) from exc

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    uploaded: list[str] = []
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        blob = bucket.blob(f"{prefix.rstrip('/')}/{rel}")
        blob.upload_from_filename(path)
        uploaded.append(f"gs://{bucket_name}/{blob.name}")
    return uploaded

