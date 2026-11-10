from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def rainfall_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    threshold: float = 0.1,
    probabilities: np.ndarray | None = None,
    probability_threshold: float = 0.5,
) -> dict[str, float]:
    """Comprehensive deterministic and categorical precipitation evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0.0)

    # Deterministic metrics
    mae = float(mean_absolute_error(y_true, prediction))
    rmse = float(np.sqrt(mean_squared_error(y_true, prediction)))
    bias = float(np.mean(prediction - y_true))

    # Pearson correlation with zero-variance guard
    std_true = np.std(y_true)
    std_pred = np.std(prediction)
    if std_true > 1e-6 and std_pred > 1e-6:
        corr = float(np.corrcoef(y_true, prediction)[0, 1])
    else:
        corr = 0.0

    # Categorical verification at primary threshold
    actual_wet = y_true >= threshold
    predicted_wet = prediction >= threshold

    tp = np.sum(actual_wet & predicted_wet)
    fp = np.sum(~actual_wet & predicted_wet)
    fn = np.sum(actual_wet & ~predicted_wet)
    tn = np.sum(~actual_wet & ~predicted_wet)

    precision = float(precision_score(actual_wet, predicted_wet, zero_division=0))
    recall = float(recall_score(actual_wet, predicted_wet, zero_division=0))
    f1 = float(f1_score(actual_wet, predicted_wet, zero_division=0))
    csi = float(tp / max(tp + fp + fn, 1))
    far = float(fp / max(tp + fp, 1))
    pofd = float(fp / max(fp + tn, 1))
    tss = float(recall - pofd)

    # Multi-threshold CSI evaluation
    csi_at_5mm = float(
        np.sum((y_true >= 5.0) & (prediction >= 5.0))
        / max(np.sum((y_true >= 5.0) | (prediction >= 5.0)), 1)
    )
    csi_at_15mm = float(
        np.sum((y_true >= 15.0) & (prediction >= 15.0))
        / max(np.sum((y_true >= 15.0) | (prediction >= 15.0)), 1)
    )

    metrics = {
        "mae_mm": mae,
        "rmse_mm": rmse,
        "bias_mm": bias,
        "pearson_r": corr,
        "wet_precision": precision,
        "wet_recall_pod": recall,
        "wet_f1": f1,
        "wet_csi_threat_score": csi,
        "false_alarm_ratio_far": far,
        "true_skill_statistic_tss": tss,
        "csi_at_5mm": csi_at_5mm,
        "csi_at_15mm": csi_at_15mm,
    }

    if probabilities is not None:
        probs = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
        brier_score = float(np.mean((probs - actual_wet.astype(float)) ** 2))
        metrics["brier_score"] = brier_score
        metrics["probability_threshold"] = probability_threshold

    return metrics
