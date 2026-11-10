from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "outputs" / ".matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from floodnowcast.calibration import predict
from floodnowcast.data import PRECIP_FORECASTS
from floodnowcast.drainage import CoupledDrainageSystem


def aggregate_monotonicity_checks(checks: dict) -> bool:
    """Return true only when every boolean monotonicity check is true."""
    values = []
    for key, value in checks.items():
        if key == "all_monotonicity_checks_passed":
            continue
        if isinstance(value, dict):
            values.append(aggregate_monotonicity_checks(value))
        elif isinstance(value, bool):
            values.append(value)
    return bool(values) and all(values)


def peak_surface_burden(df: pd.DataFrame) -> float:
    """Peak cumulative spill plus water still stored at each timestep."""
    cumulative_spill = df["overflow_spill_m3"].cumsum()
    return float((cumulative_spill + df["surface_storage_total_m3"]).max())


def load_pilot_mumbai_storm(data_dir: Path, target_date: str = "2026-07-05") -> pd.DataFrame:
    """Extract one forecast-origin sequence for an illustrative historical comparison."""
    mumbai_path = data_dir / "loc_1275339.parquet"
    if not mumbai_path.exists():
        return pd.DataFrame()

    table = pq.read_table(mumbai_path)
    df = table.to_pandas()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    origin = pd.Timestamp(target_date, tz="UTC")
    forecast_origin = df["valid_time"] - pd.to_timedelta(df["lead_hours"], unit="h")
    sub_day = df[forecast_origin == origin].sort_values("valid_time").reset_index(drop=True)
    if sub_day.empty:
        raise ValueError(f"No Mumbai rows support forecast origin {origin.isoformat()}")
    sub_origins = sub_day["valid_time"] - pd.to_timedelta(sub_day["lead_hours"], unit="h")
    if sub_origins.nunique() != 1 or sub_day["valid_time"].duplicated().any():
        raise ValueError("Mumbai demonstration contains mixed forecast origins or duplicate valid times")
    expected = pd.date_range(sub_day["valid_time"].min(), sub_day["valid_time"].max(), freq="h", tz="UTC")
    if list(sub_day["valid_time"]) != list(expected):
        raise ValueError("Mumbai demonstration has missing hourly valid times")
    return sub_day


def run_flood_simulations(data_dir: Path, output_dir: Path, seed: int = 26085,
                          model_path: Path | None = None, config_path: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    system = CoupledDrainageSystem(name="Pilot Urban Drainage Network (Synthetic Demo)",
                                   config_path=str(config_path) if config_path else None)

    with open(config_path, encoding="utf-8") as handle:
        scenario_config = yaml.safe_load(handle).get("synthetic_scenarios", {})
    baseline_rain = np.asarray(scenario_config["baseline_rain_mm"], dtype=float)
    heavy_rain = baseline_rain * float(scenario_config["heavy_rain_multiplier"])
    blockage_fraction = float(scenario_config["blockage_fraction"])
    tide_fraction = float(scenario_config["outfall_restriction_fraction"])

    scenarios = {}

    # Scenario 1: Baseline design storm
    df_base, sum_base = system.simulate_timeseries(baseline_rain)
    scenarios["baseline"] = (df_base, sum_base, "Baseline 12h convective storm (peak 28.5 mm/h), 100% capacity, free outfall")

    # Scenario 2: Extreme rainfall (1.75x)
    df_heavy, sum_heavy = system.simulate_timeseries(heavy_rain)
    scenarios["heavy_rain"] = (df_heavy, sum_heavy, "Extreme rainfall deluge (1.75x baseline, peak 49.9 mm/h), 100% capacity")

    # Scenario 3: Trunk Conduit C4 Blockage (50% reduction in C4 capacity downstream of Sag N4)
    df_block_c4, sum_block_c4 = system.simulate_timeseries(baseline_rain, blockages={"C4": blockage_fraction})
    scenarios["conduit_blockage_c4"] = (df_block_c4, sum_block_c4, "Baseline rain with 50% silt/debris blockage in trunk conduit C4 draining Sag N4")

    # Scenario 4: Midstream Junction C3 Blockage (50% reduction in C3 capacity upstream of Sag N4)
    df_block_c3, sum_block_c3 = system.simulate_timeseries(baseline_rain, blockages={"C3": blockage_fraction})
    scenarios["conduit_blockage_c3"] = (df_block_c3, sum_block_c3, "Baseline rain with 50% silt/debris blockage in midstream conduit C3 surcharging at Junction N3")

    # Scenario 5: Coastal High Tide Backwater (60% outfall restriction at N5)
    df_tide, sum_tide = system.simulate_timeseries(baseline_rain, outfall_restriction_fraction=tide_fraction)
    scenarios["coastal_high_tide"] = (df_tide, sum_tide, "Baseline rain with spring high tide / storm surge restricting Outfall N5 discharge by 60%")

    # Scenario 6: Compound Disaster (1.75x rain + 50% C4 blockage + 60% high tide outfall backwater)
    df_comp, sum_comp = system.simulate_timeseries(heavy_rain, blockages={"C4": blockage_fraction}, outfall_restriction_fraction=tide_fraction)
    scenarios["compound_disaster"] = (df_comp, sum_comp, "Compound multi-hazard: 1.75x rain + 50% C4 blockage + 60% tidal outfall restriction")

    # Save CSVs for synthetic scenarios
    scenario_metadata = {}
    for sname, (df_scen, sum_scen, desc) in scenarios.items():
        csv_file = output_dir / f"scenario_{sname}.csv"
        df_scen.to_csv(csv_file, index=False)
        scenario_metadata[sname] = {
            "description": desc,
            "peak_rainfall_mm": sum_scen["peak_rainfall_mm"],
            "peak_flood_depth_sag_cm": sum_scen["peak_flood_depth_sag_cm"],
            "depth_cap_m": system.max_surface_depth_m,
            "depth_cap_reached": bool(
                (df_scen["ponding_depth_sag_N4_m"] >= system.max_surface_depth_m).any()
            ),
            "peak_surface_storage_m3": float(df_scen["surface_storage_total_m3"].max()),
            "final_surface_storage_m3": float(df_scen["surface_storage_total_m3"].iloc[-1]),
            "total_overflow_spill_m3": sum_scen["total_overflow_spill_m3"],
            "depth_interpretation": "Capped synthetic storage-equivalent depth; not a physical street-water prediction.",
            "hours_high_risk": sum_scen["hours_with_high_risk"],
            "relative_mass_error": sum_scen["relative_mass_balance_error"],
            "accounting_check_passed": sum_scen["mass_conserved_strictly"],
        }

    # 2. Illustrative historical Mumbai forecast comparison (2026-07-05)
    mumbai_data = load_pilot_mumbai_storm(data_dir, "2026-07-05")
    mumbai_coupling_info = {}
    if not mumbai_data.empty and len(mumbai_data) >= 12:
        mumbai_subset = mumbai_data.iloc[:24].copy()
        
        print("Loading the persisted precipitation model for the Mumbai comparison...")
        if model_path is None or not model_path.exists():
            raise FileNotFoundError("Run train_precipitation.py first to create outputs/precipitation_model.pkl")
        import pickle
        model_artifact = pickle.loads(model_path.read_bytes())
        models = model_artifact["models"]

        mumbai_subset["fc_ensemble_mean"] = mumbai_subset[PRECIP_FORECASTS].mean(axis=1).fillna(0)
        calibrated_mumbai_precip = predict(
            models,
            mumbai_subset,
            threshold=float(model_artifact.get("wet_threshold", 0.1)),
            probability_threshold=float(model_artifact.get("probability_threshold", 0.5)),
        )
        mumbai_subset["calibrated_precipitation"] = calibrated_mumbai_precip

        # Simulate through drainage network with:
        # a) ERA5-Land truth rainfall
        truth_series = mumbai_subset["truth_precipitation"].to_numpy()
        df_mumbai_truth, sum_mumbai_truth = system.simulate_timeseries(truth_series)
        df_mumbai_truth.to_csv(output_dir / "scenario_mumbai_pilot_era5_truth.csv", index=False)

        # b) Raw NWP ensemble forecast rainfall
        raw_series = mumbai_subset["fc_ensemble_mean"].to_numpy()
        df_mumbai_raw, sum_mumbai_raw = system.simulate_timeseries(raw_series)
        df_mumbai_raw.to_csv(output_dir / "scenario_mumbai_pilot_raw_nwp.csv", index=False)

        # c) Calibrated hurdle model rainfall
        df_mumbai_cal, sum_mumbai_cal = system.simulate_timeseries(calibrated_mumbai_precip)
        df_mumbai_cal.to_csv(output_dir / "scenario_mumbai_pilot_calibrated_historical_comparison.csv", index=False)

        mumbai_coupling_info = {
            "location": "Mumbai (loc_1275339, 19.0728 N, 72.8826 E)",
            "date": "2026-07-05 (illustrative historical comparison; forecast origin 00:00 UTC)",
            "sequence_semantics": "One historical forecast origin across contiguous lead hours; not an operational nowcast.",
            "forecast_origin": "2026-07-05T00:00:00+00:00",
            "hours": len(mumbai_subset),
            "peak_rainfall": {
                "era5_truth_mm": float(np.max(truth_series)),
                "raw_nwp_ensemble_mm": float(np.max(raw_series)),
                "calibrated_model_mm": float(np.max(calibrated_mumbai_precip)),
            },
            "peak_simulated_flood_depth_sag_cm": {
                "era5_truth_cm": sum_mumbai_truth["peak_flood_depth_sag_cm"],
                "raw_nwp_ensemble_cm": sum_mumbai_raw["peak_flood_depth_sag_cm"],
                "calibrated_model_cm": sum_mumbai_cal["peak_flood_depth_sag_cm"],
            },
            "flood_hours_high_risk": {
                "era5_truth": sum_mumbai_truth["hours_with_high_risk"],
                "raw_nwp_ensemble": sum_mumbai_raw["hours_with_high_risk"],
                "calibrated_model": sum_mumbai_cal["hours_with_high_risk"],
            }
        }

        # Plot Mumbai coupling comparison
        plt.figure(figsize=(10, 5), dpi=150)
        plt.subplot(2, 1, 1)
        hours = np.arange(len(mumbai_subset))
        plt.plot(hours, truth_series, "k-o", label="ERA5-Land reanalysis reference", linewidth=2)
        plt.plot(hours, raw_series, "r--s", label="Raw NWP Ensemble Mean", alpha=0.8)
        plt.plot(hours, calibrated_mumbai_precip, "b-^", label="Calibrated historical forecast", linewidth=1.5)
        plt.ylabel("Rainfall (mm/h)", fontsize=10)
        plt.title("Mumbai historical comparison (2026-07-05 forecast origin)", fontsize=11, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right", fontsize=9)

        plt.subplot(2, 1, 2)
        plt.plot(hours, df_mumbai_truth["ponding_depth_sag_N4_cm"], "k-o", label="Depth from ERA5-Land reference", linewidth=2)
        plt.plot(hours, df_mumbai_raw["ponding_depth_sag_N4_cm"], "r--s", label="Depth from Raw NWP", alpha=0.8)
        plt.plot(hours, df_mumbai_cal["ponding_depth_sag_N4_cm"], "b-^", label="Depth from Calibrated historical comparison", linewidth=1.5)
        plt.axhline(20.0, color="red", linestyle=":", label="HIGH Risk Threshold (20 cm)")
        plt.axhline(5.0, color="orange", linestyle=":", label="MEDIUM Risk Threshold (5 cm)")
        plt.xlabel("Simulation Hour (UTC)", fontsize=10)
        plt.ylabel("Ponding Depth at Sag (cm)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        p_mumbai = output_dir / "mumbai_event_coupling.png"
        plt.savefig(p_mumbai)
        plt.close()

    # 3. Plot Synthetic Scenario Comparison
    plt.figure(figsize=(9, 5), dpi=150)
    hours_syn = np.arange(len(baseline_rain))
    plt.plot(hours_syn, df_base["ponding_depth_sag_N4_cm"], "b-o", label="Baseline (Clear Drains)", linewidth=2)
    plt.plot(hours_syn, df_heavy["ponding_depth_sag_N4_cm"], "g-s", label="Heavy Rain (1.75x)", linewidth=2)
    plt.plot(hours_syn, df_block_c4["ponding_depth_sag_N4_cm"], "m--^", label="Trunk C4 50% Blocked", linewidth=1.8)
    plt.plot(hours_syn, df_tide["ponding_depth_sag_N4_cm"], "c-.v", label="High Spring Tide Outfall Restricted", linewidth=1.8)
    plt.plot(hours_syn, df_comp["ponding_depth_sag_N4_cm"], "r-d", label="Compound Disaster", linewidth=2.5)
    plt.axhline(20.0, color="red", linestyle=":", label="HIGH Risk (20 cm)")
    plt.axhline(5.0, color="orange", linestyle=":", label="MEDIUM Risk (5 cm)")
    plt.xlabel("Storm Hour", fontsize=11)
    plt.ylabel("Sag Node N4 Flood Depth (cm)", fontsize=11)
    plt.title("Urban Flood Risk Sensitivity across Controlled Hydraulic Scenarios", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    p_scen = output_dir / "flood_scenario_ponding_depths.png"
    plt.savefig(p_scen)
    plt.close()

    # 4. Strict Engineering Sanity & Monotonicity Verification
    peak_depth_base = sum_base["peak_flood_depth_sag_cm"]
    peak_depth_heavy = sum_heavy["peak_flood_depth_sag_cm"]
    peak_depth_block_c4 = sum_block_c4["peak_flood_depth_sag_cm"]
    peak_depth_tide = sum_tide["peak_flood_depth_sag_cm"]
    peak_depth_comp = sum_comp["peak_flood_depth_sag_cm"]

    burden_base = peak_surface_burden(df_base)
    burden_heavy = peak_surface_burden(df_heavy)
    burden_block_c4 = peak_surface_burden(df_block_c4)
    burden_block_c3 = peak_surface_burden(df_block_c3)
    burden_tide = peak_surface_burden(df_tide)
    burden_comp = peak_surface_burden(df_comp)

    depth_n3_base = float(df_base["ponding_depth_junction_N3_cm"].max())
    depth_n3_block_c3 = float(df_block_c3["ponding_depth_junction_N3_cm"].max())

    checks = {
        "status": "completed",
        "pilot_area": {
            "name": "Synthetic 5-Node Pilot Urban Catchment (Representing Coastal Metros like Mumbai/Chennai)",
            "total_area_km2": 10.0,
            "average_imperviousness": 0.77,
            "nodes_modeled": len(system.nodes),
            "conduits_modeled": len(system.conduits),
            "critical_sag_hotspot": "Node N4 (Elevation 7.5m, Underpass depression)",
            "outfall": "Node N5 (Elevation 2.0m, Coastal/Estuary flap-gate)",
        },
        "mass_balance_accounting_checks": {
            "baseline_relative_error": sum_base["relative_mass_balance_error"],
            "heavy_rain_relative_error": sum_heavy["relative_mass_balance_error"],
            "blockage_c4_relative_error": sum_block_c4["relative_mass_balance_error"],
            "blockage_c3_relative_error": sum_block_c3["relative_mass_balance_error"],
            "compound_relative_error": sum_comp["relative_mass_balance_error"],
            "all_scenarios_accounted_within_tolerance": all(
                s["relative_mass_balance_error"] < 1e-5
                for s in [sum_base, sum_heavy, sum_block_c4, sum_block_c3, sum_tide, sum_comp]
            ),
        },
        "hydraulic_monotonicity_checks": {
            "heavy_rain_sag_depth_not_lower": bool(peak_depth_heavy >= peak_depth_base),
            "heavy_rain_sag_increase_cm": round(peak_depth_heavy - peak_depth_base, 2),
            "c4_blockage_sag_depth_not_lower": bool(peak_depth_block_c4 >= peak_depth_base),
            "c4_blockage_sag_increase_cm": round(peak_depth_block_c4 - peak_depth_base, 2),
            "c3_blockage_n3_depth_not_lower": bool(depth_n3_block_c3 >= depth_n3_base),
            "c3_blockage_n3_increase_cm": round(depth_n3_block_c3 - depth_n3_base, 2),
            "coastal_high_tide_sag_depth_not_lower": bool(peak_depth_tide >= peak_depth_base),
            "coastal_high_tide_sag_increase_cm": round(peak_depth_tide - peak_depth_base, 2),
            "compound_disaster_produces_highest_sag_depth": bool(
                peak_depth_comp >= max(peak_depth_base, peak_depth_heavy, peak_depth_block_c4, peak_depth_tide)
            ),
            "compound_disaster_peak_sag_depth_cm": round(peak_depth_comp, 2),
            "cumulative_surface_burden_monotonicity": {
                "definition": "maximum over time of cumulative spill plus current surface storage (m3)",
                "baseline_burden_m3": round(burden_base, 2),
                "heavy_rain_burden_m3": round(burden_heavy, 2),
                "c4_blockage_burden_m3": round(burden_block_c4, 2),
                "c3_blockage_burden_m3": round(burden_block_c3, 2),
                "high_tide_burden_m3": round(burden_tide, 2),
                "compound_burden_m3": round(burden_comp, 2),
                "heavy_rain_increases_surface_burden": bool(burden_heavy >= burden_base),
                "c4_blockage_increases_surface_burden": bool(burden_block_c4 >= burden_base),
                "c3_blockage_increases_surface_burden": bool(burden_block_c3 >= burden_base),
                "high_tide_increases_surface_burden": bool(burden_tide >= burden_base),
                "compound_disaster_produces_highest_surface_burden": bool(
                    burden_comp >= max(burden_base, burden_heavy, burden_block_c4, burden_tide)
                ),
            },
        },
        "scenarios_evaluated": scenario_metadata,
        "nwp_pilot_coupling_event": mumbai_coupling_info,
        "generated_artifacts": [
            "outputs/scenario_baseline.csv",
            "outputs/scenario_heavy_rain.csv",
            "outputs/scenario_conduit_blockage_c4.csv",
            "outputs/scenario_conduit_blockage_c3.csv",
            "outputs/scenario_coastal_high_tide.csv",
            "outputs/scenario_compound_disaster.csv",
            "outputs/flood_scenario_ponding_depths.png",
            "outputs/scenario_mumbai_pilot_era5_truth.csv",
            "outputs/scenario_mumbai_pilot_raw_nwp.csv",
            "outputs/scenario_mumbai_pilot_calibrated_historical_comparison.csv",
            "outputs/mumbai_event_coupling.png" if mumbai_coupling_info else None,
        ],
        "disclaimer_and_provenance": (
            "All drainage conduit dimensions, slopes, manhole depths, and surface depression areas are "
            "engineered synthetic representations for prototype testing. No municipal sewer CAD/GIS, "
            "real storm drain telemetry, or street flood gauge data were supplied in the project dataset. "
            "Precipitation inputs are real NWP and ERA5-Land reanalysis values; flood depths are not empirical or operational predictions."
        )
    }
    monotonicity = checks["hydraulic_monotonicity_checks"]
    monotonicity["all_monotonicity_checks_passed"] = aggregate_monotonicity_checks(monotonicity)

    checks_file = output_dir / "flood_demo_checks.json"
    checks_file.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"Flood simulation demo successfully executed. Checks written to {checks_file}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="weathergpt-d1-mos-dataset/d1_mos")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=26085)
    parser.add_argument("--model-path", default="outputs/precipitation_model.pkl")
    parser.add_argument("--config-path", default="config.yaml")
    args = parser.parse_args()

    results = run_flood_simulations(
        Path(args.data_dir), Path(args.output_dir), seed=args.seed,
        model_path=Path(args.model_path), config_path=Path(args.config_path)
    )
    print("\n--- Summary of Hydraulic Monotonicity and Conservation Checks ---")
    print(f"All Scenarios Accounted (rel error < 1e-5): {results['mass_balance_accounting_checks']['all_scenarios_accounted_within_tolerance']}")
    print(f"All Monotonicity Checks Passed: {results['hydraulic_monotonicity_checks']['all_monotonicity_checks_passed']}")
    print(f"Baseline Peak Sag Depth: {results['scenarios_evaluated']['baseline']['peak_flood_depth_sag_cm']} cm")
    print(f"Heavy Rain Peak Sag Depth: {results['scenarios_evaluated']['heavy_rain']['peak_flood_depth_sag_cm']} cm")
    print(f"Conduit C4 Blockage Peak Sag Depth: {results['scenarios_evaluated']['conduit_blockage_c4']['peak_flood_depth_sag_cm']} cm")
    print(f"Conduit C3 Blockage Peak Sag Depth: {results['scenarios_evaluated']['conduit_blockage_c3']['peak_flood_depth_sag_cm']} cm")
    print(f"High Tide Backwater Peak Sag Depth: {results['scenarios_evaluated']['coastal_high_tide']['peak_flood_depth_sag_cm']} cm")
    print(f"Compound Disaster Peak Sag Depth: {results['scenarios_evaluated']['compound_disaster']['peak_flood_depth_sag_cm']} cm")


if __name__ == "__main__":
    main()
