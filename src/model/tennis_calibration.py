"""Platt + temperature scaling calibration (mirrors the MLB calibration module)."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def to_logits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) / np.clip(1 - probs, 1e-7, 1 - 1e-7))

    def nll(T):
        scaled = expit(logits / T)
        scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(scaled) + (1 - y) * np.log(1 - scaled))

    try:
        result = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
        return float(result.x) if result.success else 1.0
    except Exception:
        return 1.0


class TennisCalibratedModel:
    """XGBoost + Platt scaling + temperature scaling for tennis predictions."""

    def __init__(
        self,
        base_model: XGBClassifier,
        calibrator: LogisticRegression | None,
        temperature: float = 1.0,
    ):
        self.base_model = base_model
        self.calibrator = calibrator
        self.temperature = float(temperature)
        self.estimator = base_model

    def __getattr__(self, name):
        if name == "temperature":
            return 1.0
        raise AttributeError(name)

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1]
        if self.calibrator is None:
            cal = raw
            return np.column_stack([1 - cal, cal])

        platt = self.calibrator.predict_proba(to_logits(raw))[:, 1]
        if abs(self.temperature - 1.0) < 1e-6:
            cal = platt
        else:
            logit_platt = np.log(
                np.clip(platt, 1e-7, 1 - 1e-7) / np.clip(1 - platt, 1e-7, 1 - 1e-7)
            )
            cal = expit(logit_platt / self.temperature)
        return np.column_stack([1 - cal, cal])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
