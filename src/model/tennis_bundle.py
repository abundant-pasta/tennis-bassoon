"""Versioned model bundle helpers for production inference."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import shutil

import joblib

import config_tennis as cfg
from src.features.tennis_engineer import get_feature_columns

_BUNDLE_FILES = {
    "model": "model.pkl",
    "scaler": "scaler.pkl",
    "medians": "medians.pkl",
    "metadata": "metadata.json",
    "feature_columns": "feature_columns.json",
    "manifest": "bundle_manifest.json",
}


def feature_contract(feature_columns: list[str] | None = None) -> dict:
    columns = feature_columns or get_feature_columns()
    checksum = sha256("\n".join(columns).encode("utf-8")).hexdigest()
    return {"feature_columns": columns, "feature_checksum": checksum}


def bundle_dir_for_version(version: str) -> Path:
    return cfg.MODEL_BUNDLES_DIR / version


def create_candidate_bundle(version: str | None = None) -> Path:
    version = version or datetime.now(timezone.utc).strftime("candidate_%Y%m%dT%H%M%SZ")
    bundle_dir = bundle_dir_for_version(version)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = cfg.PRODUCTION_METADATA_PATH if cfg.PRODUCTION_METADATA_PATH.exists() else cfg.MODEL_METADATA_PATH
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    contract = feature_contract()

    shutil.copy2(cfg.MODEL_PATH, bundle_dir / _BUNDLE_FILES["model"])
    shutil.copy2(cfg.SCALER_PATH, bundle_dir / _BUNDLE_FILES["scaler"])
    shutil.copy2(cfg.MEDIANS_PATH, bundle_dir / _BUNDLE_FILES["medians"])
    (bundle_dir / _BUNDLE_FILES["metadata"]).write_text(json.dumps(metadata, indent=2, default=str))
    (bundle_dir / _BUNDLE_FILES["feature_columns"]).write_text(
        json.dumps(contract["feature_columns"], indent=2)
    )

    manifest = {
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_role": "candidate",
        "source_model_path": str(cfg.MODEL_PATH),
        "source_scaler_path": str(cfg.SCALER_PATH),
        "source_medians_path": str(cfg.MEDIANS_PATH),
        **contract,
    }
    (bundle_dir / _BUNDLE_FILES["manifest"]).write_text(json.dumps(manifest, indent=2))
    return bundle_dir


def load_bundle(bundle_dir: Path | None = None) -> dict:
    bundle_dir = bundle_dir or cfg.PROMOTED_CURRENT_DIR
    manifest = json.loads((bundle_dir / _BUNDLE_FILES["manifest"]).read_text())
    metadata = json.loads((bundle_dir / _BUNDLE_FILES["metadata"]).read_text())
    feature_columns = json.loads((bundle_dir / _BUNDLE_FILES["feature_columns"]).read_text())
    return {
        "bundle_dir": bundle_dir,
        "manifest": manifest,
        "metadata": metadata,
        "feature_columns": feature_columns,
        "model": joblib.load(bundle_dir / _BUNDLE_FILES["model"]),
        "scaler": joblib.load(bundle_dir / _BUNDLE_FILES["scaler"]),
        "medians": joblib.load(bundle_dir / _BUNDLE_FILES["medians"]),
    }


def validate_bundle(bundle_dir: Path) -> dict:
    manifest = json.loads((bundle_dir / _BUNDLE_FILES["manifest"]).read_text())
    feature_columns = json.loads((bundle_dir / _BUNDLE_FILES["feature_columns"]).read_text())
    expected = feature_contract(feature_columns)
    if manifest.get("feature_checksum") != expected["feature_checksum"]:
        raise ValueError(
            f"Feature contract mismatch for bundle {bundle_dir}: "
            f"{manifest.get('feature_checksum')} != {expected['feature_checksum']}"
        )
    for key, filename in _BUNDLE_FILES.items():
        if not (bundle_dir / filename).exists():
            raise FileNotFoundError(f"Bundle is missing {key}: {bundle_dir / filename}")
    return manifest


def evaluate_promotion_gates(train_metrics: dict, oos_summary: dict, ytd_summary: dict) -> dict:
    checks = {
        "val_auc": float(train_metrics.get("val_auc_roc", 0.0)) >= cfg.PROMOTION_MIN_VAL_AUC,
        "val_log_loss": float(train_metrics.get("val_log_loss_selected", 1.0)) <= cfg.PROMOTION_MAX_VAL_LOG_LOSS,
        "oos_flat_roi": float(oos_summary.get("flat_bet", {}).get("flat_roi") or -1.0) >= cfg.PROMOTION_MIN_OOS_FLAT_ROI,
        "ytd_flat_roi": float(ytd_summary.get("flat_bet", {}).get("flat_roi") or -1.0) >= cfg.PROMOTION_MIN_YTD_FLAT_ROI,
    }
    return {"passed": all(checks.values()), "checks": checks}


def promote_bundle(bundle_dir: Path, gate_report: dict, oos_summary: dict, ytd_summary: dict) -> Path:
    validate_bundle(bundle_dir)
    promoted_dir = cfg.PROMOTED_CURRENT_DIR
    promoted_dir.mkdir(parents=True, exist_ok=True)
    for filename in _BUNDLE_FILES.values():
        shutil.copy2(bundle_dir / filename, promoted_dir / filename)

    manifest = json.loads((promoted_dir / _BUNDLE_FILES["manifest"]).read_text())
    promotion_record = {
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_version": manifest["version"],
        "gate_report": gate_report,
        "oos_summary": oos_summary,
        "ytd_summary": ytd_summary,
    }
    cfg.PROMOTION_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.PROMOTION_MANIFEST_PATH.write_text(json.dumps(promotion_record, indent=2, default=str))
    return promoted_dir
