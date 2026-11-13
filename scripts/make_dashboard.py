from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import pandas as pd


def image_to_base64_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def build_location_map_svg(locations: list[dict], spatial_holdout_ids: set[str]) -> str:
    """Build a self-contained latitude/longitude plot for the dataset locations."""
    if not locations:
        return '<p class="location-empty">Location details are unavailable. Run the dataset audit to add them.</p>'

    width, height = 760, 470
    left, right, top, bottom = 62, 730, 24, 420
    lats = [float(item["lat"]) for item in locations]
    lons = [float(item["lon"]) for item in locations]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    lat_span = max(lat_max - lat_min, 1.0)
    lon_span = max(lon_max - lon_min, 1.0)
    lat_min -= lat_span * 0.06
    lat_max += lat_span * 0.06
    lon_min -= lon_span * 0.06
    lon_max += lon_span * 0.06

    grid = []
    for step in range(5):
        fraction = step / 4
        x = left + fraction * (right - left)
        y = top + fraction * (bottom - top)
        longitude = lon_min + fraction * (lon_max - lon_min)
        latitude = lat_max - fraction * (lat_max - lat_min)
        grid.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="map-grid" />'
            f'<text x="{x:.1f}" y="{bottom + 22}" class="map-axis-label" text-anchor="middle">{longitude:.1f}°E</text>'
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="map-grid" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" class="map-axis-label" text-anchor="end">{latitude:.1f}°N</text>'
        )

    points = []
    for item in locations:
        loc_id = str(item.get("loc_id", ""))
        x = left + (float(item["lon"]) - lon_min) / (lon_max - lon_min) * (right - left)
        y = top + (lat_max - float(item["lat"])) / (lat_max - lat_min) * (bottom - top)
        label = str(item.get("location_label", f"loc_{loc_id}"))
        state = str(item.get("admin1", "unknown"))
        fill_class = "heldout" if loc_id in spatial_holdout_ids else "seen"
        description = (
            f"{label} | {state} | {float(item['lat']):.4f}°N, "
            f"{float(item['lon']):.4f}°E"
        )
        points.append(
            f'<circle class="location-point {fill_class}" data-location-id="{html.escape(loc_id, quote=True)}" '
            f'cx="{x:.2f}" cy="{y:.2f}" r="5.5" tabindex="0" role="button" '
            f'aria-label="{html.escape(description, quote=True)}"><title>{html.escape(description)}</title></circle>'
        )

    return (
        f'<svg class="location-map" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Latitude and longitude plot of dataset locations">'
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" rx="8" class="map-plot-bg" />'
        f'{"".join(grid)}{"".join(points)}'
        f'<text x="{(left + right) / 2}" y="{height - 8}" class="map-axis-title" text-anchor="middle">Longitude</text>'
        '</svg>'
    )


def build_dashboard(output_dir: Path) -> Path:
    audit_file = output_dir / "data_audit.json"
    metrics_file = output_dir / "precipitation_metrics.json"
    checks_file = output_dir / "flood_demo_checks.json"

    audit = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.exists() else {}
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    checks = json.loads(checks_file.read_text(encoding="utf-8")) if checks_file.exists() else {}

    # Read scenarios into JSON for interactive UI
    scenario_names = [
        ("baseline", "Baseline Design Storm (100% capacity)"),
        ("heavy_rain", "Extreme Deluge (1.75x, peak 49.9 mm/h)"),
        ("conduit_blockage_c4", "Trunk Conduit C4 50% Blocked"),
        ("conduit_blockage_c3", "Junction Conduit C3 50% Blocked"),
        ("coastal_high_tide", "Coastal High Tide 60% Backwater"),
        ("compound_disaster", "Compound Multi-Hazard Disaster"),
    ]
    scenarios_data = {}
    scenario_summaries = checks.get("scenarios_evaluated", {})
    for sname, label in scenario_names:
        csv_p = output_dir / f"scenario_{sname}.csv"
        if csv_p.exists():
            df = pd.read_csv(csv_p)
            scenarios_data[sname] = {
                "label": label,
                "hours": df["step_hour"].tolist(),
                "rain_mm": df["rainfall_mm"].tolist(),
                "depth_n4_cm": df["ponding_depth_sag_N4_cm"].tolist(),
                "depth_n3_cm": df["ponding_depth_junction_N3_cm"].tolist(),
                "storage_m3": df["surface_storage_total_m3"].tolist(),
                "risk": df["flood_risk_level"].tolist(),
                "c3_exceeded": df["conduit_C3_capacity_exceeded"].tolist(),
                "c4_exceeded": df["conduit_C4_capacity_exceeded"].tolist(),
                "spill_m3": df["overflow_spill_m3"].tolist(),
            }

    # Embedded Plots as base64
    p_lead = image_to_base64_data_uri(output_dir / "lead_time_mae.png")
    p_csi = image_to_base64_data_uri(output_dir / "csi_by_threshold.png")
    p_flood = image_to_base64_data_uri(output_dir / "flood_scenario_ponding_depths.png")
    p_mumbai = image_to_base64_data_uri(output_dir / "mumbai_event_coupling.png")

    scenarios_json_str = json.dumps(scenarios_data)

    # Performance stats extraction
    temp_eval = metrics.get("evaluations", {}).get("future_time_seen_locations", {})
    spat_eval = metrics.get("evaluations", {}).get("spatial_holdout", {})

    temp_raw = temp_eval.get("raw_ensemble_mean", {})
    temp_cal = temp_eval.get("calibrated_hurdle_model", {})
    spat_raw = spat_eval.get("raw_ensemble_mean", {})
    spat_cal = spat_eval.get("calibrated_hurdle_model", {})

    temp_mae_gain = round((temp_raw.get("mae_mm", 1) - temp_cal.get("mae_mm", 1)) / max(temp_raw.get("mae_mm", 1), 1e-6) * 100, 2)
    spat_mae_gain = round((spat_raw.get("mae_mm", 1) - spat_cal.get("mae_mm", 1)) / max(spat_raw.get("mae_mm", 1), 1e-6) * 100, 2)
    locations = audit.get("locations", metrics.get("locations", []))
    locations = sorted(locations, key=lambda item: int(item.get("loc_id", 0)))
    spatial_holdout_ids = {
        str(loc_id) for loc_id in metrics.get("spatial_holdout_locations", [])
    }
    location_rows = "".join(
        f'<tr class="location-row" data-location-id="{html.escape(str(item.get("loc_id", "")), quote=True)}" '
        f'data-search="{html.escape(" ".join(str(item.get(key, "")) for key in ("loc_id", "location_label", "admin1")), quote=True).lower()}">'
        f"<td>{html.escape(str(item.get('location_label', item.get('loc_id', ''))))}</td>"
        f"<td>{html.escape(str(item.get('admin1', 'unknown')))}</td>"
        f"<td>{float(item.get('lat', 0)):.4f}</td><td>{float(item.get('lon', 0)):.4f}</td>"
        f"<td>{float(item.get('elevation_m', 0)):.1f}</td>"
        f"<td>{int(item.get('row_count', 0)):,}</td>"
        f"<td>{html.escape(str(item.get('first_valid_time', '—'))[:10])} – {html.escape(str(item.get('last_valid_time', '—'))[:10])}</td>"
        f"<td>{float(item.get('mean_truth_precipitation_mm', 0)):.3f}</td>"
        f"<td>{int(item.get('wet_hours', 0)):,} ({float(item.get('wet_fraction', 0)) * 100:.1f}%)</td>"
        f"<td>{sum((item.get('forecast_missing_values') or {}).values()):,}</td>"
        f"<td>{'Spatial holdout' if str(item.get('loc_id', '')) in spatial_holdout_ids else 'Seen locations'}</td></tr>"
        for item in locations
    )
    if not location_rows:
        location_rows = '<tr><td colspan="11">Run the dataset audit to populate location statistics.</td></tr>'
    location_map = build_location_map_svg(locations, spatial_holdout_ids)
    location_state_count = len({str(item.get("admin1", "unknown")) for item in locations})
    csi_block = (
        f'<div class="chart-card"><h3>Categorical Threat Score (CSI) from current metrics</h3>'
        f'<img src="{p_csi}" alt="CSI by threshold"></div>'
        if p_csi else ""
    )
    check_rows = []
    monotonicity = checks.get("hydraulic_monotonicity_checks", {})
    check_definitions = [
        ("Heavy rain sag depth", "heavy_rain_sag_depth_not_lower"),
        ("C4 blockage sag depth", "c4_blockage_sag_depth_not_lower"),
        ("C3 blockage N3 depth", "c3_blockage_n3_depth_not_lower"),
        ("High-tide sag depth", "coastal_high_tide_sag_depth_not_lower"),
        ("Compound sag depth", "compound_disaster_produces_highest_sag_depth"),
    ]
    for label, key in check_definitions:
        value = bool(monotonicity.get(key, False))
        status = "PASSED" if value else "FAILED"
        badge = "badge-real" if value else "badge-danger"
        check_rows.append(
            f"<tr><td>{label}</td><td>Reported monotonicity condition is true</td>"
            f"<td>{html.escape(str(monotonicity.get(key, 'missing')))}</td>"
            f'<td><span class="badge {badge}">{status}</span></td></tr>'
        )
    total_checks = monotonicity.get("cumulative_surface_burden_monotonicity", {})
    burden_definition = total_checks.get("definition", "")
    for key, value in total_checks.items():
        if key == "definition" or key.endswith("_m3"):
            continue
        check_rows.append(
            f"<tr><td>Surface burden: {key.replace('_', ' ')}</td><td>{html.escape(burden_definition)}</td>"
            f"<td>{html.escape(str(value))}</td>"
            f'<td><span class="badge {"badge-real" if value else "badge-danger"}">'
            f'{"PASSED" if value else "FAILED"}</span></td></tr>'
        )
    accounting_ok = bool(checks.get("mass_balance_accounting_checks", {}).get(
        "all_scenarios_accounted_within_tolerance", False
    ))
    check_rows.insert(
        0,
        f"<tr><td>Mass-balance accounting</td><td>All scenario accounting errors &lt; 10<sup>-5</sup></td>"
        f"<td>{accounting_ok}</td><td><span class=\"badge "
        f"{'badge-real' if accounting_ok else 'badge-danger'}\">"
        f"{'PASSED' if accounting_ok else 'FAILED'}</span></td></tr>",
    )
    check_rows_html = "".join(check_rows)

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Urban Flood Nowcasting System (SIH26085) — Interactive Prototype</title>
  <style>
    :root {{
      --primary: #0284c7;
      --primary-dark: #0369a1;
      --success: #16a34a;
      --warning: #d97706;
      --danger: #dc2626;
      --bg: #f8fafc;
      --surface: #ffffff;
      --text: #0f172a;
      --text-muted: #475569;
      --border: #e2e8f0;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding-bottom: 4rem;
    }}
    header {{
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: white;
      padding: 2rem 1.5rem;
      border-bottom: 4px solid var(--primary);
    }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 0 1rem; }}
    .badge {{
      display: inline-block;
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .badge-real {{ background: #059669; color: white; }}
    .badge-synthetic {{ background: #d97706; color: white; }}
    .badge-info {{ background: #0284c7; color: white; }}
    .badge-danger {{ background: var(--danger); color: white; }}
    
    .provenance-banner {{
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-left: 6px solid var(--primary);
      padding: 1rem 1.25rem;
      margin: 1.5rem 0;
      border-radius: 0.5rem;
      font-size: 0.95rem;
    }}
    .grid-kpi {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin: 1.5rem 0;
    }}
    .kpi-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 1.25rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .kpi-title {{ font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }}
    .kpi-val {{ font-size: 1.8rem; font-weight: 700; color: var(--primary-dark); margin: 0.25rem 0; }}
    .kpi-sub {{ font-size: 0.82rem; color: var(--text-muted); }}

    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 1.5rem;
      margin-bottom: 2rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    h2 {{ font-size: 1.35rem; margin-bottom: 1rem; color: #0f172a; border-bottom: 2px solid var(--border); padding-bottom: 0.5rem; }}
    h3 {{ font-size: 1.05rem; margin: 1rem 0 0.5rem 0; color: #334155; }}
    
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
    th, td {{ border: 1px solid var(--border); padding: 0.6rem 0.75rem; text-align: left; }}
    th {{ background: #f1f5f9; font-weight: 600; color: #334155; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .location-row {{ cursor:pointer; }}
    .location-row.selected-location-row {{ outline:2px solid #0284c7; outline-offset:-2px; background:#eff6ff !important; }}
    .location-toolbar {{ display:flex; flex-wrap:wrap; gap:0.75rem; align-items:center; margin:1rem 0; }}
    .location-search {{ flex:1 1 260px; max-width:460px; padding:0.65rem 0.8rem; border:1px solid #cbd5e1; border-radius:0.4rem; font-size:0.95rem; }}
    .location-layout {{ display:grid; grid-template-columns:minmax(320px, 1fr) minmax(280px, 0.8fr); gap:1rem; align-items:start; }}
    .location-map-wrap {{ min-width:0; border:1px solid var(--border); border-radius:0.5rem; background:#f8fafc; padding:0.5rem; }}
    .location-map {{ display:block; width:100%; height:auto; }}
    .map-plot-bg {{ fill:#f8fafc; stroke:#cbd5e1; }}
    .map-grid {{ stroke:#dbe3ed; stroke-width:1; }}
    .map-axis-label {{ fill:#475569; font-size:12px; }}
    .map-axis-title {{ fill:#334155; font-size:13px; font-weight:600; }}
    .location-point {{ stroke:#fff; stroke-width:1.5; cursor:pointer; transition:r 0.12s ease; }}
    .location-point.seen {{ fill:#0284c7; }}
    .location-point.heldout {{ fill:#f97316; }}
    .location-point:hover, .location-point:focus, .location-point.selected {{ r:8; stroke:#0f172a; stroke-width:2; outline:none; }}
    .location-point[hidden] {{ display:none; }}
    .location-legend {{ display:flex; gap:1rem; flex-wrap:wrap; color:var(--text-muted); font-size:0.85rem; padding:0.4rem 0.5rem; }}
    .legend-dot {{ display:inline-block; width:0.7rem; height:0.7rem; border-radius:50%; margin-right:0.35rem; }}
    .legend-seen {{ background:#0284c7; }} .legend-heldout {{ background:#f97316; }}
    .location-profile {{ min-height:3.5rem; padding:0.8rem 1rem; background:#eff6ff; border-radius:0.4rem; color:#1e3a8a; margin-top:0.75rem; }}
    .location-table-wrap {{ max-height:30rem; overflow:auto; margin-top:1rem; }}
    #datasetLocationTable {{ min-width:1120px; }}
    .location-empty {{ padding:1rem; color:var(--text-muted); }}
    @media (max-width:800px) {{ .location-layout {{ grid-template-columns:1fr; }} .chart-container {{ grid-template-columns:1fr; }} }}

    .scenario-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      align-items: center;
      background: #f8fafc;
      padding: 1rem;
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      margin-bottom: 1.5rem;
    }}
    select {{
      padding: 0.5rem 1rem;
      border-radius: 0.375rem;
      border: 1px solid #cbd5e1;
      font-size: 0.95rem;
      background: white;
      font-weight: 500;
    }}
    
    .network-svg {{
      width: 100%;
      height: 240px;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      margin-top: 1rem;
    }}

    .chart-container {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 1.5rem;
      margin: 1.5rem 0;
    }}
    .chart-card {{
      background: white;
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 1rem;
    }}
    .chart-card img {{ width: 100%; height: auto; border-radius: 0.25rem; display: block; }}
    
    pre {{
      background: #0f172a;
      color: #f8fafc;
      padding: 1rem;
      border-radius: 0.375rem;
      overflow-x: auto;
      font-size: 0.85rem;
      margin: 0.75rem 0;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
  </style>
</head>
<body>

<header>
  <div class="container">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
      <div>
        <h1 style="font-size:1.75rem; font-weight:800; letter-spacing:-0.02em;">Urban Flood Nowcasting System</h1>
        <p style="color:#94a3b8; font-size:1rem; margin-top:0.25rem;">Drainage and Rainfall Coupling — SIH26085 Prototype</p>
      </div>
      <div>
        <span class="badge badge-real">NWP & ERA5: Real</span>
        <span class="badge badge-synthetic">Drainage & Assets: Synthetic</span>
        <span class="badge badge-info">Pilot: Coastal Metros (Mumbai/Chennai)</span>
      </div>
    </div>
  </div>
</header>

<main class="container">

  <div class="provenance-banner">
    <strong>Prototype Provenance & Operational Boundary:</strong>
    Precipitation calibration is fit on a reproducible sample drawn after leakage-safe partitioning of the 9,582,912-row WeatherGPT D1 corpus and evaluated against ERA5-Land reanalysis (not independent station truth).
    All urban drainage conduit geometries, invert levels, surface depressions, and flood depth labels are <strong>engineered synthetic placeholders</strong> for pilot demonstration because municipal GIS drainage infrastructure and real water level gauges were not provided in the dataset.
  </div>

  <div class="grid-kpi">
    <div class="kpi-card">
      <div class="kpi-title">Data Coverage</div>
      <div class="kpi-val">{audit.get("total_rows", 0):,}</div>
      <div class="kpi-sub">Hourly rows across {audit.get("unique_locations_count", 0)} Indian district HQs</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Temporal MAE Gain</div>
      <div class="kpi-val">+{temp_mae_gain}%</div>
      <div class="kpi-sub">{temp_raw.get("mae_mm", 0):.4f} &rarr; {temp_cal.get("mae_mm", 0):.4f} mm/h error</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Spatial Generalization</div>
      <div class="kpi-val">+{spat_mae_gain}%</div>
      <div class="kpi-sub">Gain on {len(metrics.get("spatial_holdout_locations", []))} unseen held-out districts</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Mass-Balance Accounting</div>
      <div class="kpi-val">{"PASSED" if accounting_ok else "FAILED"}</div>
      <div class="kpi-sub">Accounting includes explicit surface spill (relative error &lt; 10<sup>-5</sup>)</div>
    </div>
  </div>

  <section class="card" id="datasetLocations">
    <h2>Dataset Locations & Per-Location Statistics</h2>
    <p style="color:var(--text-muted); font-size:0.95rem;">
      Every weather-data row is tied to a source location ID. The source gives state/UT and coordinates, but not district names;
      the map is a latitude/longitude plot, not an administrative boundary map. Rainfall statistics below use ERA5-Land reference values.
    </p>
    <div class="location-toolbar">
      <input id="datasetLocationSearch" class="location-search" type="search" placeholder="Search by location ID or state / UT" aria-label="Search dataset locations">
      <span><strong>{len(locations)}</strong> locations across <strong>{location_state_count}</strong> states / UTs</span>
      <span id="locationResultCount" aria-live="polite"></span>
    </div>
    <div class="location-layout">
      <div class="location-map-wrap">
        {location_map}
        <div class="location-legend">
          <span><i class="legend-dot legend-seen"></i>Seen locations</span>
          <span><i class="legend-dot legend-heldout"></i>Spatial test holdout</span>
        </div>
      </div>
      <div>
        <h3 style="margin-top:0;">Selected location</h3>
        <div class="location-profile" id="selectedDatasetLocation">Choose a point on the plot or a row in the table to see its identifier.</div>
        <p style="color:var(--text-muted); font-size:0.85rem; margin-top:0.75rem;">
          Blue markers are locations seen during model training/testing; orange markers are the held-out locations used to assess spatial generalization.
        </p>
      </div>
    </div>
    <div class="location-table-wrap">
      <table id="datasetLocationTable">
        <thead><tr><th>Location ID</th><th>State / UT</th><th>Latitude</th><th>Longitude</th><th>Elevation (m)</th><th>Rows</th><th>Data period (UTC)</th><th>Mean reference rain (mm/h)</th><th>Wet reference hours / % (&ge;0.1 mm)</th><th>Missing forecast cells</th><th>Evaluation set</th></tr></thead>
        <tbody>{location_rows}</tbody>
      </table>
    </div>
  </section>

  <!-- Interactive Drainage Simulation -->
  <section class="card">
    <h2>1. Interactive Coupled Drainage Network & Hydraulic Scenarios</h2>
    <p style="color:var(--text-muted); font-size:0.95rem; margin-bottom:1rem;">
      Select a controlled storm/blockage scenario to observe runoff generation, conduit flow routing, and manhole surcharge at low-lying Sag Hotspot Node N4.
    </p>

    <div class="scenario-controls">
      <label for="scenarioSelect" style="font-weight:600;">Active Hydraulic Scenario:</label>
      <select id="scenarioSelect" onchange="updateScenario(this.value)">
        <option value="baseline">Baseline Convective Storm (Peak 28.5 mm/h, 100% capacity)</option>
        <option value="heavy_rain">Extreme Cloudburst Deluge (1.75x Baseline, Peak 49.9 mm/h)</option>
        <option value="conduit_blockage_c4">Trunk Conduit C4 Siltation Blockage (50% reduced capacity)</option>
        <option value="conduit_blockage_c3">Midstream Conduit C3 Blockage (Surcharging at Junction N3)</option>
        <option value="coastal_high_tide">Coastal High Tide / Storm Surge Backwater (60% outfall restriction)</option>
        <option value="compound_disaster">Compound Multi-Hazard (1.75x rain + 50% C4 blockage + 60% tidal backwater)</option>
      </select>
      <div id="riskBadge" style="margin-left:auto;">
        <span class="badge badge-info" style="font-size:0.9rem; padding:0.4rem 0.8rem;">Risk Level: CALCULATING</span>
      </div>
    </div>

    <!-- SVG Network Map -->
    <svg class="network-svg" viewBox="0 0 850 200" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
        </marker>
      </defs>
      <!-- Conduits (Edges) -->
      <line x1="120" y1="50" x2="350" y2="100" stroke="#0284c7" stroke-width="4" marker-end="url(#arrow)" />
      <line x1="120" y1="150" x2="350" y2="100" stroke="#0284c7" stroke-width="4" marker-end="url(#arrow)" />
      <line id="edgeC3" x1="350" y1="100" x2="570" y2="100" stroke="#0284c7" stroke-width="6" marker-end="url(#arrow)" />
      <line id="edgeC4" x1="570" y1="100" x2="750" y2="100" stroke="#0284c7" stroke-width="8" marker-end="url(#arrow)" />

      <!-- Conduit labels -->
      <text x="210" y="65" fill="#64748b" font-size="11" font-weight="600">C1 (Diam 1.4m)</text>
      <text x="210" y="145" fill="#64748b" font-size="11" font-weight="600">C2 (Diam 1.2m)</text>
      <text id="lblC3" x="430" y="90" fill="#0369a1" font-size="12" font-weight="700">C3 Trunk Culvert</text>
      <text id="lblC4" x="635" y="90" fill="#0369a1" font-size="12" font-weight="700">C4 Main Outfall Culvert</text>

      <!-- Nodes -->
      <!-- N1 -->
      <circle cx="120" cy="50" r="18" fill="#38bdf8" stroke="#0369a1" stroke-width="2" />
      <text x="120" y="54" text-anchor="middle" fill="white" font-weight="bold" font-size="11">N1</text>
      <text x="120" y="24" text-anchor="middle" fill="#334155" font-size="11" font-weight="bold">North Ridge (28m)</text>

      <!-- N2 -->
      <circle cx="120" cy="150" r="18" fill="#38bdf8" stroke="#0369a1" stroke-width="2" />
      <text x="120" y="154" text-anchor="middle" fill="white" font-weight="bold" font-size="11">N2</text>
      <text x="120" y="184" text-anchor="middle" fill="#334155" font-size="11" font-weight="bold">East Suburb (22m)</text>

      <!-- N3 -->
      <circle id="nodeN3" cx="350" cy="100" r="22" fill="#0ea5e9" stroke="#0284c7" stroke-width="3" />
      <text x="350" y="105" text-anchor="middle" fill="white" font-weight="bold" font-size="12">N3</text>
      <text x="350" y="138" text-anchor="middle" fill="#334155" font-size="11" font-weight="bold">Central Junction (15m)</text>

      <!-- N4 (Sag Hotspot) -->
      <circle id="nodeN4" cx="570" cy="100" r="26" fill="#f59e0b" stroke="#d97706" stroke-width="4" />
      <text x="570" y="105" text-anchor="middle" fill="white" font-weight="bold" font-size="13">N4</text>
      <text x="570" y="145" text-anchor="middle" fill="#dc2626" font-size="12" font-weight="bold">Sag Underpass Hotspot (7.5m)</text>

      <!-- N5 (Outfall) -->
      <rect x="750" y="80" width="40" height="40" rx="6" fill="#059669" stroke="#047857" stroke-width="3" />
      <text x="770" y="105" text-anchor="middle" fill="white" font-weight="bold" font-size="12">N5</text>
      <text x="770" y="138" text-anchor="middle" fill="#059669" font-size="11" font-weight="bold">Tidal Outfall (2m)</text>
    </svg>

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1.5rem;">
      <div>
        <h3>Storm Hyetograph & Runoff Inflow</h3>
        <table id="tblScenarioHyetograph">
          <thead>
            <tr><th>Hour</th><th>Rainfall (mm/h)</th><th>Sag N4 Depth (cm)</th><th>Risk Level</th><th>Conduit C4 Flow</th></tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
      <div>
        <h3>Hydraulic Scenario Summary</h3>
        <table id="tblScenarioMetrics">
          <tr><th>Peak Rainfall Intensity</th><td id="mPeakRain">--</td></tr>
          <tr><th>Peak Flood Depth at Sag Node N4</th><td id="mPeakDepth">--</td></tr>
          <tr><th>Hours of Severe Flood Risk (HIGH &ge; 20cm)</th><td id="mHighHours">--</td></tr>
          <tr><th>Peak Catchment Surface Storage Volume</th><td id="mPeakVol">--</td></tr>
          <tr><th>Total spill from storage cap</th><td id="mSpill">--</td></tr>
          <tr><th>Conduit Capacity Surcharged?</th><td id="mSurcharge">--</td></tr>
        </table>
      </div>
    </div>
  </section>

  <!-- Model Performance & Baselines -->
  <section class="card">
    <h2>2. Machine Learning Rainfall Calibration & Baseline Comparison</h2>
    <p style="color:var(--text-muted); font-size:0.95rem;">
      Target: Hourly ERA5-Land precipitation calibrated from 4 raw NWP forecasts using a Two-Stage Hurdle Model (Classifier + Cube-Root Regressor).
    </p>

    <table>
      <thead>
        <tr>
          <th>Model / Evaluation Set</th>
          <th>MAE (mm/h)</th>
          <th>RMSE (mm/h)</th>
          <th>Bias (mm/h)</th>
          <th>Pearson <i>r</i></th>
          <th>CSI Threat Score (&ge;0.1mm)</th>
          <th>False Alarm Ratio (FAR)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Climatology Baseline (Training Mean)</strong></td>
          <td>{temp_eval.get("climatology", {}).get("mae_mm", 0):.4f}</td>
          <td>{temp_eval.get("climatology", {}).get("rmse_mm", 0):.4f}</td>
          <td>{temp_eval.get("climatology", {}).get("bias_mm", 0):+.4f}</td>
          <td>0.000</td>
          <td>{temp_eval.get("climatology", {}).get("wet_csi_threat_score", 0):.4f}</td>
          <td>{temp_eval.get("climatology", {}).get("false_alarm_ratio_far", 0)*100:.1f}%</td>
        </tr>
        <tr>
          <td><strong>Raw GFS Seamless Forecast</strong></td>
          <td>{temp_eval.get("raw_gfs", {}).get("mae_mm", 0):.4f}</td>
          <td>{temp_eval.get("raw_gfs", {}).get("rmse_mm", 0):.4f}</td>
          <td>{temp_eval.get("raw_gfs", {}).get("bias_mm", 0):+.4f}</td>
          <td>{temp_eval.get("raw_gfs", {}).get("pearson_r", 0):.3f}</td>
          <td>{temp_eval.get("raw_gfs", {}).get("wet_csi_threat_score", 0):.4f}</td>
          <td>{temp_eval.get("raw_gfs", {}).get("false_alarm_ratio_far", 0)*100:.1f}%</td>
        </tr>
        <tr>
          <td><strong>Raw ECMWF IFS 0.25&deg; Forecast</strong></td>
          <td>{temp_eval.get("raw_ecmwf", {}).get("mae_mm", 0):.4f}</td>
          <td>{temp_eval.get("raw_ecmwf", {}).get("rmse_mm", 0):.4f}</td>
          <td>{temp_eval.get("raw_ecmwf", {}).get("bias_mm", 0):+.4f}</td>
          <td>{temp_eval.get("raw_ecmwf", {}).get("pearson_r", 0):.3f}</td>
          <td>{temp_eval.get("raw_ecmwf", {}).get("wet_csi_threat_score", 0):.4f}</td>
          <td>{temp_eval.get("raw_ecmwf", {}).get("false_alarm_ratio_far", 0)*100:.1f}%</td>
        </tr>
        <tr style="background:#fef2f2;">
          <td><strong>Raw 4-Model Ensemble Mean (Temporal Test)</strong></td>
          <td>{temp_raw.get("mae_mm", 0):.4f}</td>
          <td>{temp_raw.get("rmse_mm", 0):.4f}</td>
          <td>{temp_raw.get("bias_mm", 0):+.4f}</td>
          <td>{temp_raw.get("pearson_r", 0):.3f}</td>
          <td>{temp_raw.get("wet_csi_threat_score", 0):.4f}</td>
          <td>{temp_raw.get("false_alarm_ratio_far", 0)*100:.1f}%</td>
        </tr>
        <tr style="background:#f0fdf4; font-weight:600;">
          <td><strong>Calibrated Hurdle Model (Temporal Test)</strong></td>
          <td>{temp_cal.get("mae_mm", 0):.4f}</td>
          <td>{temp_cal.get("rmse_mm", 0):.4f}</td>
          <td>{temp_cal.get("bias_mm", 0):+.4f}</td>
          <td>{temp_cal.get("pearson_r", 0):.3f}</td>
          <td>{temp_cal.get("wet_csi_threat_score", 0):.4f}</td>
          <td>{temp_cal.get("false_alarm_ratio_far", 0)*100:.1f}%</td>
        </tr>
        <tr style="background:#f8fafc; border-top:2px solid #cbd5e1;">
          <td><strong>Spatial Generalization: Raw Ensemble Mean</strong></td>
          <td>{spat_raw.get("mae_mm", 0):.4f}</td>
          <td>{spat_raw.get("rmse_mm", 0):.4f}</td>
          <td>{spat_raw.get("bias_mm", 0):+.4f}</td>
          <td>{spat_raw.get("pearson_r", 0):.3f}</td>
          <td>{spat_raw.get("wet_csi_threat_score", 0):.4f}</td>
          <td>{spat_raw.get("false_alarm_ratio_far", 0)*100:.1f}%</td>
        </tr>
        <tr style="background:#f0fdf4; font-weight:600;">
          <td><strong>Spatial Generalization: Calibrated Hurdle Model</strong></td>
          <td>{spat_cal.get("mae_mm", 0):.4f}</td>
          <td>{spat_cal.get("rmse_mm", 0):.4f}</td>
          <td>{spat_cal.get("bias_mm", 0):+.4f}</td>
          <td>{spat_cal.get("pearson_r", 0):.3f}</td>
          <td>{spat_cal.get("wet_csi_threat_score", 0):.4f}</td>
          <td>{spat_cal.get("false_alarm_ratio_far", 0)*100:.1f}%</td>
        </tr>
      </tbody>
    </table>

    <div class="chart-container">
      <div class="chart-card">
        <h3>Lead-Time Skill Curve (0 to 191 Hours)</h3>
        <img src="{p_lead}" alt="Lead Time MAE">
      </div>
      {csi_block}
    </div>
  </section>

  <!-- Illustrative historical comparison & synthetic hydrologic coupling -->
  <section class="card">
    <h2>3. Illustrative Historical Comparison & Synthetic Hydraulic Sensitivity</h2>
    <div class="chart-container">
      <div class="chart-card">
        <h3>Controlled Hydraulic Sensitivity Scenarios</h3>
        <img src="{p_flood}" alt="Hydraulic Scenarios Comparison">
      </div>
      <div class="chart-card">
        <h3>Mumbai illustrative historical comparison (2026-07-05)</h3>
        <img src="{p_mumbai}" alt="Mumbai Event Coupling">
      </div>
    </div>
  </section>

  <!-- Strict Mass Balance and Monotonicity Audit -->
  <section class="card">
    <h2>4. Engineering Sanity Checks & Monotonicity Verification</h2>
    <p style="color:var(--text-muted); font-size:0.95rem;">
      Hydraulic accounting checks track: &Sigma; Precip = &Sigma; Losses + &Sigma; Outflow + &Delta; Surface Storage + Spill. The drainage network and depths are synthetic demonstrations, not street-flood predictions.
    </p>

    <table>
      <thead>
        <tr>
          <th>Check Name</th>
          <th>Condition Evaluated</th>
          <th>Observed Result</th>
          <th>Audit Status</th>
        </tr>
      </thead>
      <tbody>
        {check_rows_html}
      </tbody>
    </table>
  </section>

  <!-- Local Reproduction Instructions -->
  <section class="card">
    <h2>5. Local Reproduction Commands</h2>
    <p style="color:var(--text-muted); font-size:0.95rem;">
      Run the complete pipeline from scratch in PowerShell or bash:
    </p>
    <pre><code># 1. Audit full 9,582,912-row dataset
python scripts/audit_dataset.py

# 2. Train and evaluate two-stage hurdle rainfall calibration models
python scripts/train_precipitation.py --max-rows 150000

# 3. Execute coupled rainfall-runoff-drainage simulations & mass balance checks
python scripts/run_flood_demo.py

# 4. Generate local interactive dashboard
python scripts/make_dashboard.py

# 5. Serve locally
python -m http.server 8000 --directory outputs</code></pre>
  </section>

</main>

<script>
  const locationRows = Array.from(document.querySelectorAll("#datasetLocationTable tbody .location-row"));
  const locationPoints = Array.from(document.querySelectorAll(".location-point"));
  const locationSearch = document.getElementById("datasetLocationSearch");
  const locationResultCount = document.getElementById("locationResultCount");

  function selectDatasetLocation(row) {{
    if (!row) return;
    locationRows.forEach(item => item.classList.toggle("selected-location-row", item === row));
    locationPoints.forEach(point => point.classList.toggle("selected", point.dataset.locationId === row.dataset.locationId));
    const cells = Array.from(row.cells).map(cell => cell.textContent.trim());
    document.getElementById("selectedDatasetLocation").textContent =
      `${{cells[0]}} · ${{cells[1]}} · ${{cells[2]}}°N, ${{cells[3]}}°E · ${{cells[5]}} rows · mean rain ${{cells[7]}} mm/h · ${{cells[8]}} wet`;
  }}

  function filterDatasetLocations() {{
    const query = locationSearch.value.trim().toLocaleLowerCase();
    const visibleIds = new Set();
    locationRows.forEach(row => {{
      const visible = row.dataset.search.includes(query);
      row.hidden = !visible;
      if (visible) visibleIds.add(row.dataset.locationId);
    }});
    locationPoints.forEach(point => {{ point.hidden = !visibleIds.has(point.dataset.locationId); }});
    locationResultCount.textContent = `${{visibleIds.size}} shown`;
    const selected = locationRows.find(row => row.classList.contains("selected-location-row"));
    if (selected && selected.hidden) selectDatasetLocation(locationRows.find(row => !row.hidden));
  }}

  locationRows.forEach(row => {{
    row.addEventListener("click", () => selectDatasetLocation(row));
    row.addEventListener("keydown", event => {{
      if (event.key === "Enter" || event.key === " ") {{ event.preventDefault(); selectDatasetLocation(row); }}
    }});
    row.tabIndex = 0;
  }});
  locationPoints.forEach(point => {{
    point.addEventListener("click", () => selectDatasetLocation(locationRows.find(row => row.dataset.locationId === point.dataset.locationId)));
    point.addEventListener("keydown", event => {{
      if (event.key === "Enter" || event.key === " ") {{
        event.preventDefault();
        selectDatasetLocation(locationRows.find(row => row.dataset.locationId === point.dataset.locationId));
      }}
    }});
  }});
  locationSearch.addEventListener("input", filterDatasetLocations);
  locationResultCount.textContent = `${{locationRows.length}} shown`;
  if (locationRows.length) selectDatasetLocation(locationRows[0]);

  const scenarioData = {scenarios_json_str};

  function updateScenario(name) {{
    const scen = scenarioData[name];
    if (!scen) return;

    // Update table
    const tbody = document.querySelector("#tblScenarioHyetograph tbody");
    tbody.innerHTML = "";
    let maxDepth = 0;
    let maxStorage = 0;
    let highHours = 0;
    let maxRain = 0;
    let c4Surcharged = false;

    for (let i = 0; i < scen.hours.length; i++) {{
      const tr = document.createElement("tr");
      const rain = scen.rain_mm[i];
      const depth = scen.depth_n4_cm[i];
      const risk = scen.risk[i];
      const exceeded = scen.c4_exceeded[i];

      if (depth > maxDepth) maxDepth = depth;
      if (scen.storage_m3[i] > maxStorage) maxStorage = scen.storage_m3[i];
      if (rain > maxRain) maxRain = rain;
      if (risk === "HIGH") highHours++;
      if (exceeded) c4Surcharged = true;

      const riskBadge = risk === "HIGH" ? '<span class="badge badge-danger">HIGH</span>' :
                        risk === "MEDIUM" ? '<span class="badge badge-synthetic">MEDIUM</span>' :
                        '<span class="badge badge-real">LOW</span>';

      tr.innerHTML = `<td>Hour ${{scen.hours[i]}}</td><td>${{rain.toFixed(1)}} mm</td><td>${{depth.toFixed(1)}} cm</td><td>${{riskBadge}}</td><td>${{exceeded ? "Surcharged" : "Normal"}}</td>`;
      tbody.appendChild(tr);
    }}

    // Update Summary Metrics
    document.getElementById("mPeakRain").innerText = `${{maxRain.toFixed(1)}} mm/h`;
    document.getElementById("mPeakDepth").innerText = `${{maxDepth.toFixed(1)}} cm (${{(maxDepth/100).toFixed(2)}} m)`;
    document.getElementById("mHighHours").innerText = `${{highHours}} hours`;
    document.getElementById("mPeakVol").innerText = `${{maxStorage.toLocaleString()}} m³`;
    const totalSpill = scen.spill_m3.reduce((sum, value) => sum + value, 0);
    document.getElementById("mSpill").innerText = `${{totalSpill.toLocaleString()}} m³ (synthetic cap)`;
    document.getElementById("mSurcharge").innerText = c4Surcharged ? "YES - Capacity Exceeded (Surcharge Active)" : "NO - Gravity Conveyance";

    // Update Badge
    const overallRisk = maxDepth >= 20.0 ? "HIGH" : maxDepth >= 5.0 ? "MEDIUM" : "LOW";
    const badgeEl = document.getElementById("riskBadge");
    if (overallRisk === "HIGH") {{
      badgeEl.innerHTML = '<span class="badge badge-danger" style="font-size:0.9rem; padding:0.4rem 0.8rem;">Risk Level: HIGH ALERT (&ge;20cm)</span>';
      document.getElementById("nodeN4").setAttribute("fill", "#dc2626");
      document.getElementById("nodeN4").setAttribute("stroke", "#991b1b");
    }} else if (overallRisk === "MEDIUM") {{
      badgeEl.innerHTML = '<span class="badge badge-synthetic" style="font-size:0.9rem; padding:0.4rem 0.8rem;">Risk Level: MEDIUM WATCH (5-20cm)</span>';
      document.getElementById("nodeN4").setAttribute("fill", "#f59e0b");
      document.getElementById("nodeN4").setAttribute("stroke", "#d97706");
    }} else {{
      badgeEl.innerHTML = '<span class="badge badge-real" style="font-size:0.9rem; padding:0.4rem 0.8rem;">Risk Level: LOW (Normal)</span>';
      document.getElementById("nodeN4").setAttribute("fill", "#10b981");
      document.getElementById("nodeN4").setAttribute("stroke", "#059669");
    }}

    // Highlight blocked edges
    const edgeC3 = document.getElementById("edgeC3");
    const edgeC4 = document.getElementById("edgeC4");
    if (name === "conduit_blockage_c4" || name === "compound_disaster") {{
      edgeC4.setAttribute("stroke", "#dc2626");
      edgeC4.setAttribute("stroke-dasharray", "6,4");
    }} else {{
      edgeC4.setAttribute("stroke", "#0284c7");
      edgeC4.removeAttribute("stroke-dasharray");
    }}

    if (name === "conduit_blockage_c3") {{
      edgeC3.setAttribute("stroke", "#dc2626");
      edgeC3.setAttribute("stroke-dasharray", "6,4");
      document.getElementById("nodeN3").setAttribute("fill", "#dc2626");
    }} else {{
      edgeC3.setAttribute("stroke", "#0284c7");
      edgeC3.removeAttribute("stroke-dasharray");
      document.getElementById("nodeN3").setAttribute("fill", "#0ea5e9");
    }}
  }}

  // Initialize with baseline
  updateScenario("baseline");
</script>

</body>
</html>"""

    dashboard_path = output_dir / "index.html"
    dashboard_path.write_text(html_content, encoding="utf-8")
    print(f"Interactive dashboard successfully generated: {dashboard_path}")
    return dashboard_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    build_dashboard(Path(args.output_dir))


if __name__ == "__main__":
    main()
