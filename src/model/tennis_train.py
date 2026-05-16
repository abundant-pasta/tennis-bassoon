"""
XGBoost training for the tennis predictor.

Split strategy (walk-forward, no leakage):
  Train:  2008–2020
  Val:    2021  (Optuna tuning)
  OOS:    2022–2024 (reported separately via tennis_oos_report.py)

Walk-forward Platt + temperature scaling calibration mirrors the MLB approach.

One-row-per-match framing: higher-ranked player is "player" (home analogue).
Target: won = 1 if the nominated "player" won.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config_tennis as cfg
from src.features.tennis_engineer import get_feature_columns
from src.model.tennis_calibration import TennisCalibratedModel, fit_temperature, to_logits

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)

_CALIBRATION_WINDOW_YEARS = 5


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def _deduplicate_to_one_row_per_match(df: pd.DataFrame) -> pd.DataFrame:
    """Convert player-perspective rows to one row per match.

    Always keep the higher-ranked player's perspective (lower rank number = higher rank).
    Tiebreaker when ranks are equal: lower player_id for stability.
    Never use `won` as a tiebreaker — that would be leakage.
    """
    df = df.copy()
    df["_rank_p"] = df["player_rank"].fillna(9999)
    df["_rank_o"] = df["opp_rank"].fillna(9999)
    df["_pid"] = pd.to_numeric(df["player_id"], errors="coerce").fillna(9999999)
    df["_oid"] = pd.to_numeric(df["opp_id"], errors="coerce").fillna(9999999)
    df["_keep"] = (df["_rank_p"] < df["_rank_o"]) | (
        (df["_rank_p"] == df["_rank_o"]) & (df["_pid"] <= df["_oid"])
    )
    # Sort so the preferred row appears first, then deduplicate
    df = df.sort_values(["match_id", "_keep"], ascending=[True, False])
    kept = df[df["_keep"]].drop_duplicates(subset=["match_id"]).copy()
    kept.drop(columns=["_rank_p", "_rank_o", "_pid", "_oid", "_keep"], inplace=True)
    return kept


def prepare_splits(
    features_df: pd.DataFrame,
    train_years: list[int] | None = None,
    val_year: int | None = None,
    oos_years: list[int] | None = None,
) -> tuple:
    feat_cols = get_feature_columns()
    df = features_df.dropna(subset=["won"]).copy()
    df["year"] = pd.to_datetime(df["tourney_date"]).dt.year

    # Filter to training scope and deduplicate
    df = df[df["year"] >= 2008].copy()
    df = _deduplicate_to_one_row_per_match(df)

    train_years = train_years or cfg.TRAIN_YEARS
    val_year = val_year or cfg.VAL_YEAR
    oos_years = oos_years or cfg.OOS_YEARS

    train = df[df["year"].isin(train_years)].copy()
    val = df[df["year"] == val_year].copy()
    train_medians = train[feat_cols].median(numeric_only=True)
    train[feat_cols] = train[feat_cols].fillna(train_medians)
    val[feat_cols] = val[feat_cols].fillna(train_medians)

    # Calibration slice: last N years before val
    cal_end = val_year
    cal_start = cal_end - _CALIBRATION_WINDOW_YEARS
    cal_slice = df[df["year"].between(cal_start, cal_end - 1)].copy()
    cal_slice[feat_cols] = cal_slice[feat_cols].fillna(train_medians)

    if train.empty or val.empty:
        raise RuntimeError(f"Empty split: train={len(train)}, val={len(val)}")

    X_train = train[feat_cols].values
    y_train = train["won"].values.astype(int)
    X_val = val[feat_cols].values
    y_val = val["won"].values.astype(int)

    split_info = {
        "train_years": train_years,
        "val_year": val_year,
        "oos_years": oos_years,
        "cal_years": list(range(cal_start, cal_end)),
    }
    return X_train, X_val, y_train, y_val, cal_slice, split_info, feat_cols, df, train_medians


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------

def _objective(trial, X_train, y_train, X_val, y_val) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "eval_metric": "logloss",
        "random_state": cfg.RANDOM_STATE,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params, early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    return log_loss(y_val, preds)


# ---------------------------------------------------------------------------
# Walk-forward calibration
# ---------------------------------------------------------------------------

def _walk_forward_calibrate(
    base_model,
    cal_slice: pd.DataFrame,
    scaler: StandardScaler,
    cal_years: list[int],
    feat_cols: list[str],
    fill_values: pd.Series,
) -> tuple[LogisticRegression, float]:
    coef_list, intercept_list = [], []
    all_platt_probs, all_labels = [], []

    for year in cal_years:
        year_df = cal_slice[cal_slice["year"] == year]
        if year_df.empty:
            continue
        year_df = year_df.copy()
        year_df[feat_cols] = year_df[feat_cols].fillna(fill_values)
        X_y = scaler.transform(year_df[feat_cols].values)
        y_y = year_df["won"].values.astype(int)
        raw_probs = base_model.predict_proba(X_y)[:, 1]
        try:
            cal = LogisticRegression(C=1.0, solver="lbfgs")
            cal.fit(to_logits(raw_probs), y_y)
            coef_list.append(float(cal.coef_[0][0]))
            intercept_list.append(float(cal.intercept_[0]))
            all_platt_probs.append(cal.predict_proba(to_logits(raw_probs))[:, 1])
            all_labels.append(y_y)
        except Exception:
            continue

    if not coef_list:
        fallback_df = cal_slice[cal_slice["year"] == cal_years[-1]]
        fallback_df = fallback_df.copy()
        fallback_df[feat_cols] = fallback_df[feat_cols].fillna(fill_values)
        X_cal = scaler.transform(fallback_df[feat_cols].values)
        y_cal = fallback_df["won"].values.astype(int)
        raw = base_model.predict_proba(X_cal)[:, 1]
        calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        calibrator.fit(to_logits(raw), y_cal)
        return calibrator, 1.0

    dummy_X = np.array([[0.0], [1.0]])
    dummy_y = np.array([0, 1])
    avg_cal = LogisticRegression(C=1.0, solver="lbfgs")
    avg_cal.fit(dummy_X, dummy_y)
    avg_cal.coef_ = np.array([[np.mean(coef_list)]])
    avg_cal.intercept_ = np.array([np.mean(intercept_list)])

    pooled_probs = np.concatenate(all_platt_probs)
    pooled_labels = np.concatenate(all_labels)
    T = fit_temperature(pooled_probs, pooled_labels)

    print(f"  Walk-forward calibration: {len(coef_list)} folds averaged "
          f"(A={np.mean(coef_list):.4f}, B={np.mean(intercept_list):.4f})")
    print(f"  Temperature T={T:.4f}")
    return avg_cal, T


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _artifact_paths(artifact_dir: Path | None) -> dict[str, Path]:
    if artifact_dir is None:
        return {
            "holdout_model": cfg.HOLDOUT_MODEL_PATH,
            "holdout_scaler": cfg.HOLDOUT_SCALER_PATH,
            "holdout_medians": cfg.HOLDOUT_MEDIANS_PATH,
            "holdout_meta": cfg.MODEL_METADATA_PATH,
            "prod_model": cfg.MODEL_PATH,
            "prod_scaler": cfg.SCALER_PATH,
            "prod_medians": cfg.MEDIANS_PATH,
            "prod_meta": cfg.PRODUCTION_METADATA_PATH,
            "feature_contract": cfg.MODEL_FEATURE_CONTRACT_PATH,
        }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return {
        "holdout_model": artifact_dir / "model.pkl",
        "holdout_scaler": artifact_dir / "scaler.pkl",
        "holdout_medians": artifact_dir / "medians.pkl",
        "holdout_meta": artifact_dir / "metadata.json",
        "prod_model": artifact_dir / "production_model.pkl",
        "prod_scaler": artifact_dir / "production_scaler.pkl",
        "prod_medians": artifact_dir / "production_medians.pkl",
        "prod_meta": artifact_dir / "production_metadata.json",
        "feature_contract": artifact_dir / "feature_columns.json",
    }


def train(
    features_df: pd.DataFrame,
    n_trials: int = cfg.OPTUNA_TRIALS,
    train_years: list[int] | None = None,
    val_year: int | None = None,
    oos_years: list[int] | None = None,
    artifact_dir: Path | None = None,
    skip_production_refit: bool = False,
) -> tuple[TennisCalibratedModel, StandardScaler, dict]:
    feat_cols_: list[str]
    artifact_paths = _artifact_paths(artifact_dir)
    result = prepare_splits(
        features_df,
        train_years=train_years,
        val_year=val_year,
        oos_years=oos_years,
    )
    X_train, X_val, y_train, y_val, cal_slice, split_info, feat_cols_, full_df, train_medians = result

    print(f"Features: {len(feat_cols_)}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    print(f"Running Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective(trial, X_train_s, y_train, X_val_s, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_params = study.best_params
    print(f"Best params: {best_params}")

    final_params = {
        **best_params,
        "eval_metric": "logloss",
        "random_state": cfg.RANDOM_STATE,
        "n_jobs": -1,
    }
    final_params.pop("early_stopping_rounds", None)
    base_model = XGBClassifier(**final_params)
    base_model.fit(X_train_s, y_train, verbose=False)

    print("Calibrating...")
    cal_slice = cal_slice.copy()
    cal_slice["year"] = pd.to_datetime(cal_slice["tourney_date"]).dt.year
    calibrator, temperature = _walk_forward_calibrate(
        base_model, cal_slice, scaler, split_info["cal_years"], feat_cols_, train_medians
    )
    val_probs_raw = base_model.predict_proba(X_val_s)[:, 1]
    calibrated = TennisCalibratedModel(base_model, calibrator, temperature=temperature)
    val_probs_cal = calibrated.predict_proba(X_val_s)[:, 1]

    use_calibration = log_loss(y_val, val_probs_cal) < log_loss(y_val, val_probs_raw)
    selected_calibrator = calibrator if use_calibration else None
    selected_temperature = temperature if use_calibration else 1.0
    selected_model = TennisCalibratedModel(base_model, selected_calibrator, temperature=selected_temperature)
    val_probs_selected = selected_model.predict_proba(X_val_s)[:, 1]
    val_preds = (val_probs_selected >= 0.5).astype(int)

    # Baseline: fraction of matches where the higher-ranked player wins (post-dedup, won.mean())
    # After dedup, 'player' is always the higher-ranked player, so won.mean() = baseline accuracy.
    feat_df_val = full_df[full_df["year"] == split_info["val_year"]].copy()
    feat_df_val = _deduplicate_to_one_row_per_match(feat_df_val)
    rank_baseline = float(feat_df_val["won"].mean())

    metrics = {
        "val_accuracy": round(accuracy_score(y_val, val_preds), 4),
        "val_log_loss_raw": round(log_loss(y_val, val_probs_raw), 4),
        "val_log_loss_cal": round(log_loss(y_val, val_probs_cal), 4),
        "val_log_loss_selected": round(log_loss(y_val, val_probs_selected), 4),
        "val_brier_score": round(brier_score_loss(y_val, val_probs_selected), 4),
        "val_auc_roc": round(roc_auc_score(y_val, val_probs_selected), 4),
        "rank_baseline_accuracy": round(rank_baseline, 4),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": len(feat_cols_),
        "best_params": best_params,
        "temperature": round(temperature, 4),
        "calibration_selected": use_calibration,
        "calibration_mode": "platt_temperature" if use_calibration else "raw_identity",
        **split_info,
    }

    print("\nVal metrics:")
    for k, v in metrics.items():
        if k not in ("best_params", "train_years", "oos_years", "cal_years"):
            print(f"  {k}: {v}")

    if metrics["val_accuracy"] <= metrics["rank_baseline_accuracy"]:
        print(
            f"\nWARNING: Model accuracy ({metrics['val_accuracy']:.3f}) does not beat "
            f"ATP ranking baseline ({metrics['rank_baseline_accuracy']:.3f}). "
            "Investigate features before running OOS simulation."
        )

    # Save holdout model (val-season only; production model is below)
    if not use_calibration:
        print("  Calibration skipped for deployment: validation log loss was worse than raw model.")

    joblib.dump(selected_model, artifact_paths["holdout_model"])
    joblib.dump(scaler, artifact_paths["holdout_scaler"])
    joblib.dump(train_medians.to_dict(), artifact_paths["holdout_medians"])
    holdout_meta = {
        **metrics,
        "artifact_role": "holdout",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifact_paths["holdout_meta"].write_text(json.dumps(holdout_meta, indent=2, default=str))
    print(f"\nHoldout model saved → {artifact_paths['holdout_model']}")

    # Production refit on all completed data
    prod_model = selected_model
    prod_scaler = scaler
    if not skip_production_refit:
        print("Refitting production model on full dataset...")
        prod_df = full_df.dropna(subset=["won"]).copy()
        prod_medians = prod_df[feat_cols_].median(numeric_only=True)
        prod_df[feat_cols_] = prod_df[feat_cols_].fillna(prod_medians)
        X_prod_raw = prod_df[feat_cols_].values
        y_prod = prod_df["won"].values.astype(int)
        prod_scaler = StandardScaler()
        X_prod_s = prod_scaler.fit_transform(X_prod_raw)
        prod_base = XGBClassifier(**final_params)
        prod_base.fit(X_prod_s, y_prod, verbose=False)
        prod_model = TennisCalibratedModel(prod_base, selected_calibrator, temperature=selected_temperature)

        joblib.dump(prod_model, artifact_paths["prod_model"])
        joblib.dump(prod_scaler, artifact_paths["prod_scaler"])
        joblib.dump(prod_medians.to_dict(), artifact_paths["prod_medians"])
        prod_meta = {
            **metrics,
            "artifact_role": "production",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "production_refit_matches": len(y_prod),
        }
        artifact_paths["prod_meta"].write_text(json.dumps(prod_meta, indent=2, default=str))
        artifact_paths["feature_contract"].write_text(json.dumps(feat_cols_, indent=2))
        print(f"Production model saved → {artifact_paths['prod_model']}")
    else:
        artifact_paths["feature_contract"].write_text(json.dumps(feat_cols_, indent=2))

    return prod_model, prod_scaler, metrics


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    base = getattr(model, "base_model", model)
    importances = base.feature_importances_
    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(df["feature"][:20][::-1], df["importance"][:20][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Tennis Feature Importances")
    plt.tight_layout()
    out_path = Path("outputs/tennis/feature_importance.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Feature importance plot saved → {out_path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from src.features.tennis_engineer import load_features

    print("Loading features...")
    features = load_features()
    print(f"  {len(features)} rows")

    model, scaler, metrics = train(features)
    feat_cols = get_feature_columns()
    imp = get_feature_importance(model, feat_cols)
    print("\nTop 15 features:")
    print(imp.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
