"""
Model evaluation: calibration curves, feature importance, edge bucket ROI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

OUTPUTS_DIR = Path("outputs/tennis")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
                           title: str = "Calibration") -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    out = OUTPUTS_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Calibration curve → {out}")


def print_metrics(y_true: np.ndarray, y_prob: np.ndarray, label: str = "OOS") -> dict:
    preds = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": round(accuracy_score(y_true, preds), 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
        "brier": round(brier_score_loss(y_true, y_prob), 4),
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4),
    }
    print(f"\n{label} metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def edge_bucket_roi(kelly_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise ROI by edge bucket."""
    if kelly_df.empty or "edge_bucket" not in kelly_df.columns:
        return pd.DataFrame()
    rows = []
    for bkt, grp in kelly_df.groupby("edge_bucket"):
        staked = grp["stake"].sum()
        rows.append({
            "edge_bucket": bkt,
            "bets": len(grp),
            "wins": int(grp["won"].sum()),
            "win_rate": round(grp["won"].mean(), 4),
            "total_staked": round(staked, 2),
            "total_profit": round(grp["profit"].sum(), 2),
            "roi": round(grp["profit"].sum() / staked, 4) if staked > 0 else None,
        })
    df = pd.DataFrame(rows).sort_values("edge_bucket")
    print("\nEdge bucket ROI:")
    print(df.to_string(index=False))
    return df


def surface_roi(kelly_df: pd.DataFrame) -> pd.DataFrame:
    if kelly_df.empty or "surface" not in kelly_df.columns:
        return pd.DataFrame()
    rows = []
    for surf, grp in kelly_df.groupby("surface"):
        staked = grp["stake"].sum()
        rows.append({
            "surface": surf,
            "bets": len(grp),
            "win_rate": round(grp["won"].mean(), 4),
            "total_profit": round(grp["profit"].sum(), 2),
            "roi": round(grp["profit"].sum() / staked, 4) if staked > 0 else None,
        })
    df = pd.DataFrame(rows)
    print("\nSurface ROI:")
    print(df.to_string(index=False))
    return df
