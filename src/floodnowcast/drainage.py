"""Coupled Rainfall-Runoff and Directed Drainage Network Simulator.

This module models:
1. Hydrologic runoff generation (SCS / Continuous bucket with impervious abstraction).
2. Hydraulic routing across a directed drainage network (inlets, trunk conduits, outfalls)
   using Manning's open-channel/gravity pipe flow formulation.
3. Manhole surcharge and surface depression ponding at sag nodes when conduit capacity
   or tailwater boundary conditions are exceeded.
4. Rigorous mass conservation accounting and sensitivity/monotonicity auditing.

DISCLAIMER: All drainage conduits, invert levels, and ponding areas are synthetic
engineering representations for prototype demonstration, as real municipal asset
data are not present in the supplied dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class DrainageNode:
    node_id: str
    name: str
    surface_elevation_m: float
    invert_elevation_m: float
    catchment_area_km2: float
    impervious_fraction: float
    ponding_area_m2: float  # Surface depression area for surcharge accumulation
    max_surface_storage_m3: float
    is_outfall: bool = False
    surface_storage_m3: float = 0.0
    total_inflow_m3: float = 0.0
    total_surcharged_m3: float = 0.0
    depression_storage_remaining_mm: float = 0.0


@dataclass
class DrainageConduit:
    conduit_id: str
    from_node: str
    to_node: str
    length_m: float
    width_m: float
    height_m: float  # For circular pipe, width = height = diameter
    is_box: bool = False
    slope: float = 0.005
    manning_n: float = 0.015
    blockage_fraction: float = 0.0

    @property
    def cross_section_area_m2(self) -> float:
        if self.is_box:
            return self.width_m * self.height_m
        return math.pi * (self.width_m / 2.0) ** 2

    @property
    def wetted_perimeter_m(self) -> float:
        if self.is_box:
            return 2.0 * (self.width_m + self.height_m)
        return math.pi * self.width_m

    @property
    def hydraulic_radius_m(self) -> float:
        return self.cross_section_area_m2 / max(self.wetted_perimeter_m, 1e-6)

    @property
    def full_flow_capacity_m3_s(self) -> float:
        """Manning's equation for gravity full flow capacity: Q = (1/n) * A * R^(2/3) * S^(1/2)"""
        s = max(self.slope, 0.0001)
        q = (1.0 / self.manning_n) * self.cross_section_area_m2 * (self.hydraulic_radius_m ** (2.0 / 3.0)) * math.sqrt(s)
        effective_capacity = q * (1.0 - np.clip(self.blockage_fraction, 0.0, 0.99))
        return float(effective_capacity)


class CoupledDrainageSystem:
    def __init__(self, name: str = "Synthetic Pilot Urban Catchment", config_path: str | None = None):
        self.name = name
        self.nodes: Dict[str, DrainageNode] = {}
        self.conduits: Dict[str, DrainageConduit] = {}
        self.runoff_coefficient_perv = 0.25
        self.runoff_coefficient_imperv = 0.85
        self.depression_storage_imperv_mm = 1.5
        self.depression_storage_perv_mm = 5.0
        self.max_surface_depth_m = 1.0
        if config_path:
            self._load_network_from_config(config_path)
        else:
            self._build_default_pilot_network()

    def _load_network_from_config(self, config_path: str) -> None:
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        network = config.get("drainage_network", {})
        self.nodes = {}
        max_depth = float(config.get("parameters", {}).get("max_surface_depth_m", 1.0))
        self.max_surface_depth_m = max_depth
        for item in network.get("nodes", []):
            area = float(item["ponding_area_m2"])
            self.nodes[item["id"]] = DrainageNode(
                item["id"], item["name"], float(item["surface_elevation_m"]),
                float(item["invert_elevation_m"]), float(item["area_km2"]),
                float(item["impervious_fraction"]), area, area * max_depth,
                bool(item.get("is_outfall", False)),
            )
        self.conduits = {}
        for item in network.get("conduits", []):
            self.conduits[item["id"]] = DrainageConduit(
                item["id"], item["from"], item["to"], float(item["length_m"]),
                float(item["width_m"]), float(item["height_m"]),
                item.get("shape", "circular").lower() == "box",
                float(item["slope"]), float(item["manning_n"]),
            )
        self._validate_network()

    def _validate_network(self) -> None:
        if not self.nodes or not self.conduits:
            raise ValueError("Drainage configuration must define nodes and conduits")
        if any(n.impervious_fraction < 0 or n.impervious_fraction > 1 for n in self.nodes.values()):
            raise ValueError("Impervious fractions must be in [0, 1]")
        if any(c.width_m <= 0 or c.height_m <= 0 or c.length_m <= 0 or c.slope <= 0 or c.manning_n <= 0
               for c in self.conduits.values()):
            raise ValueError("Conduit dimensions, lengths, slopes, and Manning n must be positive")
        if any(c.from_node not in self.nodes or c.to_node not in self.nodes for c in self.conduits.values()):
            raise ValueError("Every conduit endpoint must reference a configured node")

    def _build_default_pilot_network(self) -> None:
        """Create a representative 5-node directed urban drainage network:
        - N1: Upstream commercial/hill area (high elevation, steep)
        - N2: Upstream residential suburb
        - N3: Midstream central junction
        - N4: Lowland underpass/sag depression (high flood risk hotspot)
        - N5: Estuary/Tidal discharge outfall
        """
        self.nodes = {
            "N1": DrainageNode("N1", "North Ridge (Commercial/Hill)", 28.0, 26.5, 2.5, 0.80, 25000.0, 25000.0),
            "N2": DrainageNode("N2", "East Suburb (Residential)", 22.0, 20.8, 2.5, 0.65, 30000.0, 30000.0),
            "N3": DrainageNode("N3", "Central Urban Junction", 15.0, 13.5, 2.0, 0.85, 40000.0, 40000.0),
            "N4": DrainageNode("N4", "South Underpass Sag (Depression Hotspot)", 7.5, 5.5, 2.0, 0.90, 50000.0, 50000.0),
            "N5": DrainageNode("N5", "Coastal/Creek Outfall", 2.0, 0.5, 1.0, 0.50, 60000.0, 60000.0, is_outfall=True),
        }

        # Directed conduits: N1 -> N3, N2 -> N3, N3 -> N4, N4 -> N5
        self.conduits = {
            "C1": DrainageConduit("C1", "N1", "N3", length_m=800.0, width_m=1.4, height_m=1.4, is_box=False, slope=0.016),
            "C2": DrainageConduit("C2", "N2", "N3", length_m=750.0, width_m=1.2, height_m=1.2, is_box=False, slope=0.010),
            "C3": DrainageConduit("C3", "N3", "N4", length_m=650.0, width_m=2.2, height_m=1.6, is_box=True, slope=0.006),
            "C4": DrainageConduit("C4", "N4", "N5", length_m=900.0, width_m=3.0, height_m=2.0, is_box=True, slope=0.003),
        }

    def reset_state(self) -> None:
        for node in self.nodes.values():
            node.surface_storage_m3 = 0.0
            node.total_inflow_m3 = 0.0
            node.total_surcharged_m3 = 0.0
            node.depression_storage_remaining_mm = (
                node.impervious_fraction * self.depression_storage_imperv_mm
                + (1.0 - node.impervious_fraction) * self.depression_storage_perv_mm
            )

    def calculate_node_runoff_m3(self, node: DrainageNode, rain_mm: float, dt_seconds: float) -> Tuple[float, float]:
        """Calculates runoff with one finite depression-storage reservoir per node."""
        if not np.isfinite(rain_mm) or rain_mm < 0:
            raise ValueError("Rainfall values must be finite and non-negative")
        area_m2 = node.catchment_area_km2 * 1e6
        total_precip_m3 = (rain_mm / 1000.0) * area_m2

        # Effective runoff depth
        c_eff = (node.impervious_fraction * self.runoff_coefficient_imperv +
                 (1.0 - node.impervious_fraction) * self.runoff_coefficient_perv)
        
        # Initial abstraction loss
        loss_mm = min(rain_mm, node.depression_storage_remaining_mm)
        node.depression_storage_remaining_mm -= loss_mm
        
        net_rain_mm = max(0.0, rain_mm - loss_mm)
        runoff_mm = net_rain_mm * c_eff
        runoff_m3 = (runoff_mm / 1000.0) * area_m2
        loss_m3 = total_precip_m3 - runoff_m3
        return float(runoff_m3), float(loss_m3)

    def simulate_timeseries(
        self,
        rainfall_series_mm: np.ndarray,
        blockages: Optional[Dict[str, float]] = None,
        outfall_restriction_fraction: float = 0.0,
        dt_hours: float = 1.0,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Simulate coupled runoff generation, drainage routing, and surface surcharge."""
        rainfall_series_mm = np.asarray(rainfall_series_mm, dtype=float)
        if rainfall_series_mm.ndim != 1 or rainfall_series_mm.size == 0:
            raise ValueError("Rainfall series must be a non-empty one-dimensional array")
        if not np.isfinite(rainfall_series_mm).all() or (rainfall_series_mm < 0).any():
            raise ValueError("Rainfall series must contain finite non-negative mm values")
        if not np.isfinite(dt_hours) or dt_hours <= 0:
            raise ValueError("dt_hours must be positive")
        if not 0 <= outfall_restriction_fraction < 1:
            raise ValueError("outfall_restriction_fraction must be in [0, 1)")
        self.reset_state()
        dt_seconds = dt_hours * 3600.0
        blockages = blockages or {}

        # Apply blockages
        for cid, frac in blockages.items():
            if cid not in self.conduits:
                raise ValueError(f"Unknown conduit blockage: {cid}")
            if not 0 <= frac < 1:
                raise ValueError(f"Blockage fraction for {cid} must be in [0, 1)")
            self.conduits[cid].blockage_fraction = frac

        steps_log = []
        cum_precip_m3 = 0.0
        cum_loss_m3 = 0.0
        cum_outfall_discharged_m3 = 0.0
        cum_spilled_m3 = 0.0

        for t_step, rain_mm in enumerate(rainfall_series_mm):
            step_precip_m3 = 0.0
            step_loss_m3 = 0.0
            step_runoff_m3 = 0.0
            step_spilled_m3 = 0.0

            # 1. Compute runoff for each node
            node_generated_runoff_m3 = {}
            for nid, node in self.nodes.items():
                r_m3, l_m3 = self.calculate_node_runoff_m3(node, rain_mm, dt_seconds)
                node_generated_runoff_m3[nid] = r_m3
                step_precip_m3 += (rain_mm / 1000.0) * (node.catchment_area_km2 * 1e6)
                step_loss_m3 += l_m3
                step_runoff_m3 += r_m3

            cum_precip_m3 += step_precip_m3
            cum_loss_m3 += step_loss_m3

            # 2. Network routing in topological order (N1, N2 -> N3 -> N4 -> N5)
            # Upstream to downstream flow
            conduit_flow_m3 = {}
            conduit_capacity_m3_step = {}
            for cid, cond in self.conduits.items():
                cap_m3 = cond.full_flow_capacity_m3_s * dt_seconds
                conduit_capacity_m3_step[cid] = cap_m3

            # Inflow to N1 and N2
            # N1
            inflow_N1 = node_generated_runoff_m3["N1"] + self.nodes["N1"].surface_storage_m3
            flow_C1 = min(inflow_N1, conduit_capacity_m3_step["C1"])
            self.nodes["N1"].surface_storage_m3 = min(max(0.0, inflow_N1 - flow_C1), self.nodes["N1"].max_surface_storage_m3)
            step_spilled_m3 += max(0.0, inflow_N1 - flow_C1 - self.nodes["N1"].surface_storage_m3)
            conduit_flow_m3["C1"] = flow_C1

            # N2
            inflow_N2 = node_generated_runoff_m3["N2"] + self.nodes["N2"].surface_storage_m3
            flow_C2 = min(inflow_N2, conduit_capacity_m3_step["C2"])
            self.nodes["N2"].surface_storage_m3 = min(max(0.0, inflow_N2 - flow_C2), self.nodes["N2"].max_surface_storage_m3)
            step_spilled_m3 += max(0.0, inflow_N2 - flow_C2 - self.nodes["N2"].surface_storage_m3)
            conduit_flow_m3["C2"] = flow_C2

            # N3 receives generated runoff + flow from C1 and C2 + previous surface storage
            inflow_N3 = (node_generated_runoff_m3["N3"] +
                         flow_C1 + flow_C2 +
                         self.nodes["N3"].surface_storage_m3)
            flow_C3 = min(inflow_N3, conduit_capacity_m3_step["C3"])
            self.nodes["N3"].surface_storage_m3 = min(max(0.0, inflow_N3 - flow_C3), self.nodes["N3"].max_surface_storage_m3)
            step_spilled_m3 += max(0.0, inflow_N3 - flow_C3 - self.nodes["N3"].surface_storage_m3)
            conduit_flow_m3["C3"] = flow_C3

            # N4 (Sag Lowland) receives generated runoff + flow from C3 + previous surface storage
            inflow_N4 = (node_generated_runoff_m3["N4"] +
                         flow_C3 +
                         self.nodes["N4"].surface_storage_m3)
            flow_C4 = min(inflow_N4, conduit_capacity_m3_step["C4"])
            self.nodes["N4"].surface_storage_m3 = min(max(0.0, inflow_N4 - flow_C4), self.nodes["N4"].max_surface_storage_m3)
            step_spilled_m3 += max(0.0, inflow_N4 - flow_C4 - self.nodes["N4"].surface_storage_m3)
            conduit_flow_m3["C4"] = flow_C4

            # N5 (Outfall) receives flow from C4 + local runoff, and discharges to sea/river
            inflow_N5 = node_generated_runoff_m3["N5"] + flow_C4 + self.nodes["N5"].surface_storage_m3
            outfall_cap = 40.0 * (1.0 - outfall_restriction_fraction) * dt_seconds
            discharged_N5 = min(inflow_N5, outfall_cap)
            self.nodes["N5"].surface_storage_m3 = min(max(0.0, inflow_N5 - discharged_N5), self.nodes["N5"].max_surface_storage_m3)
            step_spilled_m3 += max(0.0, inflow_N5 - discharged_N5 - self.nodes["N5"].surface_storage_m3)
            cum_spilled_m3 += step_spilled_m3
            cum_outfall_discharged_m3 += discharged_N5

            # Total surface storage across entire catchment
            total_surface_storage_m3 = sum(n.surface_storage_m3 for n in self.nodes.values())

            # Flood depth at sag hotspot N4 (meters and cm)
            depth_N4_m = self.nodes["N4"].surface_storage_m3 / self.nodes["N4"].ponding_area_m2
            depth_N3_m = self.nodes["N3"].surface_storage_m3 / self.nodes["N3"].ponding_area_m2
            max_depth_m = max(depth_N4_m, depth_N3_m)

            # Categorical risk
            if max_depth_m >= 0.20:
                risk_level = "HIGH"
            elif max_depth_m >= 0.05:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            step_record = {
                "step_hour": t_step,
                "rainfall_mm": float(rain_mm),
                "generated_runoff_m3": round(step_runoff_m3, 2),
                "conduit_flow_C1_m3": round(conduit_flow_m3["C1"], 2),
                "conduit_flow_C2_m3": round(conduit_flow_m3["C2"], 2),
                "conduit_flow_C3_m3": round(conduit_flow_m3["C3"], 2),
                "conduit_flow_C4_m3": round(conduit_flow_m3["C4"], 2),
                "conduit_C3_capacity_reached": bool(abs(conduit_flow_m3["C3"] - conduit_capacity_m3_step["C3"]) < 1e-4),
                "conduit_C4_capacity_reached": bool(abs(conduit_flow_m3["C4"] - conduit_capacity_m3_step["C4"]) < 1e-4),
                "conduit_C3_capacity_exceeded": bool(inflow_N3 > conduit_capacity_m3_step["C3"]),
                "conduit_C4_capacity_exceeded": bool(inflow_N4 > conduit_capacity_m3_step["C4"]),
                "overflow_spill_m3": round(step_spilled_m3, 2),
                "discharged_outfall_m3": round(discharged_N5, 2),
                "surface_storage_total_m3": round(total_surface_storage_m3, 2),
                "ponding_depth_sag_N4_m": round(depth_N4_m, 4),
                "ponding_depth_sag_N4_cm": round(depth_N4_m * 100.0, 2),
                "ponding_depth_junction_N3_cm": round(depth_N3_m * 100.0, 2),
                "max_flood_depth_m": round(max_depth_m, 4),
                "flood_risk_level": risk_level,
            }
            steps_log.append(step_record)

        df_steps = pd.DataFrame(steps_log)

        # Mass balance verification
        total_remaining_storage_m3 = sum(n.surface_storage_m3 for n in self.nodes.values())
        mass_balance_error_m3 = cum_precip_m3 - (cum_loss_m3 + cum_outfall_discharged_m3 + total_remaining_storage_m3 + cum_spilled_m3)
        relative_mass_error = abs(mass_balance_error_m3) / max(cum_precip_m3, 1e-6)

        summary = {
            "total_precipitation_m3": round(cum_precip_m3, 2),
            "total_losses_infiltrated_m3": round(cum_loss_m3, 2),
            "total_discharged_m3": round(cum_outfall_discharged_m3, 2),
            "total_overflow_spill_m3": round(cum_spilled_m3, 2),
            "final_surface_storage_m3": round(total_remaining_storage_m3, 2),
            "mass_balance_error_m3": round(mass_balance_error_m3, 6),
            "relative_mass_balance_error": float(relative_mass_error),
            "mass_conserved_strictly": bool(relative_mass_error < 1e-5),
            "peak_rainfall_mm": float(np.max(rainfall_series_mm)),
            "peak_flood_depth_m": float(df_steps["max_flood_depth_m"].max()),
            "peak_flood_depth_sag_cm": float(df_steps["ponding_depth_sag_N4_cm"].max()),
            "hours_with_high_risk": int((df_steps["flood_risk_level"] == "HIGH").sum()),
            "hours_with_medium_risk": int((df_steps["flood_risk_level"] == "MEDIUM").sum()),
        }

        # Reset blockages
        for cond in self.conduits.values():
            cond.blockage_fraction = 0.0

        return df_steps, summary
