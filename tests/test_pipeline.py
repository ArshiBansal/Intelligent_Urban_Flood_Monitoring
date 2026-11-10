import sys
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from floodnowcast.drainage import CoupledDrainageSystem
from floodnowcast.metrics import rainfall_metrics
from scripts.make_dashboard import build_dashboard
from scripts.run_flood_demo import aggregate_monotonicity_checks, load_pilot_mumbai_storm, peak_surface_burden
from scripts.train_precipitation import split_data


class PipelineTests(unittest.TestCase):
    def test_fixed_origin_event_is_contiguous_and_single_origin(self):
        frame = load_pilot_mumbai_storm(ROOT / "weathergpt-d1-mos-dataset" / "d1_mos")
        origins = frame.valid_time - pd.to_timedelta(frame.lead_hours, unit="h")
        self.assertEqual(origins.nunique(), 1)
        self.assertFalse(frame.valid_time.duplicated().any())
        self.assertEqual(frame.valid_time.diff().dropna().unique()[0], pd.Timedelta(hours=1))

    def test_metrics_has_expected_zero_safe_outputs(self):
        result = rainfall_metrics(np.array([0.0, 0.2]), np.array([0.0, 0.1]), probabilities=np.array([0.1, 0.8]))
        self.assertIn("mae_mm", result)
        self.assertIn("brier_score", result)

    def test_drainage_accounting_and_storage_sanity(self):
        system = CoupledDrainageSystem(config_path=str(ROOT / "config.yaml"))
        output, summary = system.simulate_timeseries(np.array([0.0, 10.0, 20.0]), outfall_restriction_fraction=0.5)
        self.assertLess(summary["relative_mass_balance_error"], 1e-5)
        self.assertTrue((output["surface_storage_total_m3"] >= 0).all())
        self.assertTrue((output["ponding_depth_sag_N4_m"] <= 1.0).all())
        self.assertIn("conduit_C4_capacity_reached", output.columns)
        self.assertIn("conduit_C4_capacity_exceeded", output.columns)
        self.assertIn("overflow_spill_m3", output.columns)
        self.assertAlmostEqual(
            output["overflow_spill_m3"].sum(), summary["total_overflow_spill_m3"], places=2
        )

    def test_monotonicity_aggregation_includes_surface_burden_checks(self):
        checks = {
            "depth_check": True,
            "cumulative_surface_burden_monotonicity": {
                "definition": "test metric",
                "baseline_burden_m3": 10.0,
                "heavy_rain_increases_surface_burden": False,
            },
        }
        self.assertFalse(aggregate_monotonicity_checks(checks))

    def test_surface_burden_includes_cumulative_spill_and_current_storage(self):
        frame = pd.DataFrame({
            "overflow_spill_m3": [2.0, 3.0],
            "surface_storage_total_m3": [10.0, 4.0],
        })
        self.assertEqual(peak_surface_burden(frame), 12.0)

    def test_forecast_origin_split_has_no_temporal_leakage(self):
        frame = pd.DataFrame({
            "valid_time": pd.to_datetime([
                "2026-01-01 00:00Z", "2026-01-01 01:00Z",
                "2026-01-02 00:00Z", "2026-01-02 01:00Z",
                "2026-01-03 00:00Z", "2026-01-03 01:00Z",
            ], utc=True),
            "loc_id": [1, 1, 1, 2, 2, 2],
            "lead_hours": [0, 1, 0, 1, 0, 1],
        })
        train, temporal, _, _, _ = split_data(frame, 0.5, 0.1, 26085)
        self.assertTrue(
            set(train["forecast_origin"]).isdisjoint(temporal["forecast_origin"])
        )

    def test_dashboard_uses_current_metric_keys_and_check_values(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "data_audit.json").write_text(
                json.dumps({"total_rows": 1, "unique_locations_count": 1}),
                encoding="utf-8",
            )
            metrics = {
                "evaluations": {
                    "future_time_seen_locations": {
                        "climatology": {"mae_mm": 1, "rmse_mm": 1, "bias_mm": 0,
                                        "wet_csi_threat_score": 0, "false_alarm_ratio_far": 0},
                        "raw_ensemble_mean": {"mae_mm": 1, "rmse_mm": 1, "bias_mm": 0},
                        "calibrated_hurdle_model": {"mae_mm": 1, "rmse_mm": 1, "bias_mm": 0},
                    },
                    "spatial_holdout": {
                        "raw_ensemble_mean": {"mae_mm": 1},
                        "calibrated_hurdle_model": {"mae_mm": 1},
                    },
                },
                "spatial_holdout_locations": [],
            }
            (output / "precipitation_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            checks = {
                "mass_balance_accounting_checks": {
                    "all_scenarios_accounted_within_tolerance": False,
                },
                "hydraulic_monotonicity_checks": {
                    "heavy_rain_sag_depth_not_lower": False,
                    "cumulative_surface_burden_monotonicity": {"failed_check": False},
                },
                "scenarios_evaluated": {},
            }
            (output / "flood_demo_checks.json").write_text(
                json.dumps(checks), encoding="utf-8"
            )
            columns = {
                "step_hour": [0], "rainfall_mm": [0],
                "ponding_depth_sag_N4_cm": [0],
                "ponding_depth_junction_N3_cm": [0],
                "surface_storage_total_m3": [0], "flood_risk_level": ["LOW"],
                "conduit_C3_capacity_exceeded": [False],
                "conduit_C4_capacity_exceeded": [False],
                "overflow_spill_m3": [0],
            }
            for name in ("baseline", "heavy_rain", "conduit_blockage_c4",
                         "conduit_blockage_c3", "coastal_high_tide",
                         "compound_disaster"):
                pd.DataFrame(columns).to_csv(
                    output / f"scenario_{name}.csv", index=False
                )
            content = build_dashboard(output).read_text(encoding="utf-8")
            self.assertIn("Climatology Baseline", content)
            self.assertNotIn("climatology_baseline", content)
            self.assertIn("FAILED", content)
            self.assertNotIn("Depth increased by +207.56 cm", content)


if __name__ == "__main__":
    unittest.main()
