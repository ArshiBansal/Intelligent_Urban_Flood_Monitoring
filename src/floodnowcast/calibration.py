from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .data import PRECIP_FORECASTS
from .metrics import rainfall_metrics


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[PRECIP_FORECASTS].astype(float)
    features = values.copy()
    features.columns = [f"forecast_{i}" for i in range(len(PRECIP_FORECASTS))]
    features["ensemble_mean"] = values.mean(axis=1)
    features["ensemble_std"] = values.std(axis=1).fillna(0)
    features["ensemble_min"] = values.min(axis=1)
    features["ensemble_max"] = values.max(axis=1)
    features["ensemble_wet_fraction"] = (values >= 0.1).mean(axis=1)
    features["lead_hours"] = frame["lead_hours"].astype(float)
    features["hour_sin"] = np.sin(2 * np.pi * frame["valid_time"].dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * frame["valid_time"].dt.hour / 24)
    features["doy_sin"] = np.sin(2 * np.pi * frame["valid_time"].dt.dayofyear / 366)
    features["doy_cos"] = np.cos(2 * np.pi * frame["valid_time"].dt.dayofyear / 366)
    features["lat"] = frame["lat"].astype(float)
    features["lon"] = frame["lon"].astype(float)
    features["elevation_km"] = frame.get("elevation_m", pd.Series(0, index=frame.index)).fillna(0) / 1000
    features["forecast_missing_count"] = values.isna().sum(axis=1)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def train_models(train: pd.DataFrame, threshold: float, seed: int) -> dict:
    x = make_features(train)
    y = train["truth_precipitation"].astype(float).to_numpy()
    if len(train) == 0:
        raise ValueError("Training partition is empty")
    wet = y >= threshold
    if np.unique(wet).size < 2:
        raise ValueError("Training partition contains only one wet/dry class")
    classifier = HistGradientBoostingClassifier(
        max_iter=120, max_leaf_nodes=31, learning_rate=0.08, random_state=seed
    )
    classifier.fit(x, wet)
    positive = wet
    regressor = HistGradientBoostingRegressor(
        max_iter=160, max_leaf_nodes=31, learning_rate=0.06, loss="squared_error", random_state=seed
    )
    if positive.sum() < 20:
        raise ValueError("Too few positive rainfall rows for the positive-amount model")
    regressor.fit(x.loc[positive], np.cbrt(y[positive]))
    return {"classifier": classifier, "regressor": regressor, "wet_threshold": threshold}


def predict(models: dict, frame: pd.DataFrame, threshold: float = 0.1, probability_threshold: float = 0.5) -> np.ndarray:
    x = make_features(frame)
    wet_probability = models["classifier"].predict_proba(x)[:, 1]
    amount = np.maximum(models["regressor"].predict(x), 0) ** 3
    return np.where(wet_probability >= probability_threshold, amount, 0.0)


def predict_with_proba(models: dict, frame: pd.DataFrame, threshold: float = 0.1, probability_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    x = make_features(frame)
    wet_probability = models["classifier"].predict_proba(x)[:, 1]
    amount = np.maximum(models["regressor"].predict(x), 0) ** 3
    prediction = np.where(wet_probability >= probability_threshold, amount, 0.0)
    return prediction, wet_probability


def evaluate(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    threshold: float = 0.1,
    probabilities: np.ndarray | None = None,
    probability_threshold: float = 0.5,
) -> dict:
    return rainfall_metrics(
        frame["truth_precipitation"].to_numpy(),
        prediction,
        threshold,
        probabilities=probabilities,
        probability_threshold=probability_threshold,
    )


def save_metrics(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
