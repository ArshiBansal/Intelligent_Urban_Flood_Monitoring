from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "outputs" / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floodnowcast.calibration import evaluate, make_features, predict_with_proba, save_metrics, train_models
from floodnowcast.data import PRECIP_FORECASTS, load_rows, sample_frame, validate_precipitation_frame


def split_data(frame: pd.DataFrame, train_fraction: float, spatial_fraction: float, seed: int):
    """Split on complete forecast-origin cycles and locations before sampling."""
    frame = frame.sort_values(["valid_time", "loc_id", "lead_hours"]).reset_index(drop=True)
    if frame["lead_hours"].isna().any():
        raise ValueError("Cannot split rows with missing lead_hours")
    frame = frame.copy()
    frame["forecast_origin"] = (
        frame["valid_time"] - pd.to_timedelta(frame["lead_hours"], unit="h")
    )
    origins = frame["forecast_origin"].drop_duplicates().sort_values().reset_index(drop=True)
    cutoff = origins.iloc[max(1, int(len(origins) * train_fraction))]
    locations = np.array(sorted(frame["loc_id"].dropna().unique()))
    holdout_count = max(1, int(len(locations) * spatial_fraction))
    spatial_holdout = set(np.random.default_rng(seed).choice(locations, holdout_count, replace=False).tolist())
    seen = ~frame["loc_id"].isin(spatial_holdout)
    train = frame[seen & (frame["forecast_origin"] < cutoff)].copy()
    temporal = frame[seen & (frame["forecast_origin"] >= cutoff)].copy()
    spatial = frame[frame["loc_id"].isin(spatial_holdout)].copy()
    if min(map(len, (train, temporal, spatial))) == 0:
        raise ValueError("A leakage-safe split produced an empty partition")
    if set(train["forecast_origin"]).intersection(temporal["forecast_origin"]):
        raise ValueError("Forecast-origin leakage detected between train and temporal partitions")
    return train, temporal, spatial, cutoff, sorted(spatial_holdout)


def lead_metrics(frame, raw, calibrated, probability=None, threshold=0.1, probability_threshold=0.5):
    result = {}
    buckets = {"0_to_3h": (0, 3), "0_to_6h": (0, 6), "7_to_24h": (7, 24),
               "25_to_72h": (25, 72), "73_to_191h": (73, 191)}
    for name, (low, high) in buckets.items():
        subset = frame["lead_hours"].between(low, high)
        if not subset.any():
            result[name] = {"sample_count": 0, "status": "unsupported"}
            continue
        f = frame.loc[subset]
        result[name] = {
            "sample_count": int(subset.sum()),
            "raw_ensemble_mean": evaluate(f, raw[subset], threshold),
            "calibrated_hurdle_model": evaluate(
                f, calibrated[subset], threshold,
                None if probability is None else probability[subset],
                probability_threshold,
            ),
        }
    return result


def evaluate_partition(frame, models, train_mean, threshold, probability_threshold):
    raw = frame[PRECIP_FORECASTS].mean(axis=1).fillna(0).to_numpy()
    calibrated, probability = predict_with_proba(models, frame, threshold, probability_threshold)
    baselines = {
        "climatology": evaluate(frame, np.full(len(frame), train_mean), threshold),
        "raw_gfs": evaluate(frame, frame[PRECIP_FORECASTS[0]].fillna(0).to_numpy(), threshold),
        "raw_ecmwf": evaluate(frame, frame[PRECIP_FORECASTS[1]].fillna(0).to_numpy(), threshold),
        "raw_icon": evaluate(frame, frame[PRECIP_FORECASTS[2]].fillna(0).to_numpy(), threshold),
        "raw_gem": evaluate(frame, frame[PRECIP_FORECASTS[3]].fillna(0).to_numpy(), threshold),
        "raw_ensemble_mean": evaluate(frame, raw, threshold),
        "calibrated_hurdle_model": evaluate(frame, calibrated, threshold, probability, probability_threshold),
    }
    baselines["lead_time_breakdown"] = lead_metrics(
        frame, raw, calibrated, probability, threshold, probability_threshold
    )
    return baselines, raw, calibrated


def generate_plots(output_dir, test, raw, calibrated):
    output_dir.mkdir(parents=True, exist_ok=True)
    lead_ranges = [(0, 3), (0, 6), (7, 24), (25, 72), (73, 191)]
    labels = ["0-3", "0-6", "7-24", "25-72", "73-191"]
    raw_mae, cal_mae = [], []
    for low, high in lead_ranges:
        mask = test["lead_hours"].between(low, high)
        y = test.loc[mask, "truth_precipitation"].to_numpy()
        raw_mae.append(np.mean(np.abs(raw[mask] - y)) if len(y) else np.nan)
        cal_mae.append(np.mean(np.abs(calibrated[mask] - y)) if len(y) else np.nan)
    plt.figure(figsize=(8, 4))
    plt.plot(labels, raw_mae, "r--o", label="Raw ensemble mean")
    plt.plot(labels, cal_mae, "b-s", label="Calibrated")
    plt.xlabel("Lead range (hours)")
    plt.ylabel("MAE (mm/h)")
    plt.title("Future-time test MAE by lead range")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lead_time_mae.png", dpi=150)
    plt.close()

    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    y_true = test["truth_precipitation"].to_numpy()
    raw_csi, calibrated_csi = [], []
    for threshold in thresholds:
        actual = y_true >= threshold
        raw_event = raw >= threshold
        calibrated_event = calibrated >= threshold
        raw_csi.append(np.sum(actual & raw_event) / max(np.sum(actual | raw_event), 1))
        calibrated_csi.append(
            np.sum(actual & calibrated_event) / max(np.sum(actual | calibrated_event), 1)
        )
    positions = np.arange(len(thresholds))
    width = 0.38
    plt.figure(figsize=(8, 4))
    plt.bar(positions - width / 2, raw_csi, width, label="Raw ensemble mean")
    plt.bar(positions + width / 2, calibrated_csi, width, label="Calibrated")
    plt.xticks(positions, [f"{value:g} mm" for value in thresholds])
    plt.xlabel("Rainfall event threshold")
    plt.ylabel("Critical Success Index")
    plt.title("Future-time test CSI by rainfall threshold")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "csi_by_threshold.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="weathergpt-d1-mos-dataset/d1_mos")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-rows", type=int, default=150000, help="Training sample cap after partitioning")
    parser.add_argument("--seed", type=int, default=26085)
    parser.add_argument("--wet-threshold", type=float, default=0.1)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    args = parser.parse_args()
    started = time.time()
    print("Loading the complete audited corpus before splitting...")
    frame = load_rows(args.data_dir, max_rows=None, seed=args.seed)
    validate_precipitation_frame(frame.drop_duplicates(["loc_id", "valid_time", "lead_hours"]))
    full_rows = len(frame)
    train_all, temporal_all, spatial_all, cutoff, held_out = split_data(
        frame, 0.70, 0.20, args.seed
    )
    # Equal caps keep the evaluation partitions deterministic while preserving their definitions.
    train = sample_frame(train_all, args.max_rows, args.seed)
    temporal = sample_frame(temporal_all, args.max_rows, args.seed + 1)
    spatial = sample_frame(spatial_all, args.max_rows, args.seed + 2)
    for part in (train, temporal, spatial):
        validate_precipitation_frame(part.drop_duplicates(["loc_id", "valid_time", "lead_hours"]))
    models = train_models(train, args.wet_threshold, args.seed)
    train_mean = float(train["truth_precipitation"].mean())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "precipitation_model.pkl").open("wb") as handle:
        pickle.dump({"models": models, "features": list(make_features(train).columns),
                     "wet_threshold": args.wet_threshold,
                     "probability_threshold": args.probability_threshold,
                     "seed": args.seed}, handle)
    evaluations = {}
    plots = {}
    for name, partition in (("future_time_seen_locations", temporal), ("spatial_holdout", spatial)):
        evaluations[name], raw, cal = evaluate_partition(
            partition, models, train_mean, args.wet_threshold, args.probability_threshold
        )
        _, probability = predict_with_proba(models, partition, args.wet_threshold, args.probability_threshold)
        sensitivity = {}
        for decision_threshold in (0.3, 0.5, 0.7):
            prediction, _ = predict_with_proba(
                models, partition, args.wet_threshold, decision_threshold
            )
            sensitivity[str(decision_threshold)] = evaluate(
                partition, prediction, args.wet_threshold, probability, decision_threshold
            )
        evaluations[name]["probability_threshold_sensitivity"] = sensitivity
        plots[name] = (raw, cal)
    generate_plots(output, temporal, *plots["future_time_seen_locations"])
    results = {
        "status": "completed",
        "dataset_rows_audited": full_rows,
        "dataset_rows_loaded": full_rows,
        "training_rows_sampled": len(train),
        "future_time_test_rows_sampled": len(temporal),
        "spatial_holdout_test_rows_sampled": len(spatial),
        "partition_boundaries": {
            "split_key": "forecast_origin = valid_time - lead_hours",
            "train_end_exclusive": cutoff.isoformat(),
            "future_time_test_start_inclusive": cutoff.isoformat(),
            "train_rows_before_sampling": len(train_all),
            "future_time_test_rows_before_sampling": len(temporal_all),
            "spatial_holdout_rows_before_sampling": len(spatial_all),
            "train_forecast_origin_min": train["forecast_origin"].min().isoformat(),
            "train_forecast_origin_max": train["forecast_origin"].max().isoformat(),
            "future_time_forecast_origin_min": temporal["forecast_origin"].min().isoformat(),
            "future_time_forecast_origin_max": temporal["forecast_origin"].max().isoformat(),
            "forecast_origin_leakage_check": bool(
                set(train["forecast_origin"]).isdisjoint(temporal["forecast_origin"])
            ),
        },
        "spatial_holdout_locations": held_out,
        "configuration": {"seed": args.seed, "wet_threshold_mm": args.wet_threshold,
                          "probability_threshold": args.probability_threshold,
                          "training_sample_cap_per_partition": args.max_rows},
        "evaluations": evaluations,
        "model_artifact": "outputs/precipitation_model.pkl",
        "elapsed_seconds": round(time.time() - started, 2),
        "limitations": [
            "ERA5-Land is a reanalysis reference, not independent station ground truth.",
            "Forecast origin is reconstructed as valid_time - lead_hours and complete origin cycles are kept on one split side.",
        ],
    }
    save_metrics(output / "precipitation_metrics.json", results)
    print(json.dumps(results["partition_boundaries"], indent=2))
    print(f"Saved metrics and model artifact in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
