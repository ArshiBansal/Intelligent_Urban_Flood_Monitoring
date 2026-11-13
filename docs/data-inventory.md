# Data Inventory, Provenance, and Gap Analysis

This document provides a comprehensive audit of all datasets, assets, and assumptions in the **Urban Flood Nowcasting System (SIH26085)** prototype.

---

## 1. Data Present and Verified

The dataset located at `weathergpt-d1-mos-dataset/d1_mos` was fully audited across all shards using columnar PyArrow operations without downsampling:

| Attribute | Audited Value | Significance / Usage |
|---|---|---|
| **Shards (Parquet Files)** | **127** files | One file per distinct geocoded district headquarters across India |
| **Total Rows** | **9,582,912** rows | Exact match with upstream WeatherGPT D1 corpus |
| **Rows per Shard** | **75,456** rows | Strict uniformity across all 127 locations |
| **Temporal Coverage** | `2025-08-01 00:00:00+00:00` to `2026-08-28 23:00:00+00:00` (393.0 days) | Spans complete Indian summer monsoon cycles |
| **Lead Times** | **0 to 191 hours** (192 lead hours) | Historical forecast lead values; this is not radar nowcasting |
| **Target Variable** | `truth_precipitation` | Hourly ERA5-Land reanalysis reference (0 nulls across all 9.58M rows), not station ground truth |
| **Forecast Predictors** | `fc_precipitation_gfs_seamless`<br>`fc_precipitation_ecmwf_ifs025`<br>`fc_precipitation_icon_seamless`<br>`fc_precipitation_gem_seamless` | Multi-model NWP historical forecasts from Open-Meteo API |
| **Missing Values** | GFS: 0 (0.0%)<br>ECMWF: 32,766 (0.34%)<br>ICON: 1,227,201 (12.81%)<br>GEM: 50,292 (0.52%) | Missing forecast features are filled with zero and an explicit missing-count predictor is retained |
| **Data Integrity** | `duplicate_keys = 0`<br>`chunk_misses = 0` | Zero duplicate `(valid_time, lead_hours)` pairs; zero failed chunk fetches |

The generated `outputs/data_audit.json` now includes a complete `locations` catalog.
Each entry identifies its rows using `loc_id`, source file, state/union territory
(`admin1`), latitude, longitude, and elevation. The district name is not present
in the source schema, so the dashboard uses this source-backed identifier rather
than guessing a district name. Per-location statistics include row count, UTC
coverage period, mean ERA5-Land reference rainfall, wet-hour count and fraction,
and missing forecast-cell count. The dashboard plots the locations by
latitude/longitude, lets viewers search by location ID or state/UT, and marks
the spatial holdout locations separately. `outputs/precipitation_metrics.json`
includes the same catalog and adds `spatial_holdout_location_details` for the
locations used in the spatial evaluation.

### Target Distribution (`truth_precipitation`)
- **Min / Max:** 0.0000 mm / 30.6000 mm
- **Mean / Std:** 0.1752 mm / 0.6945 mm
- **Dry Hours (< 0.1 mm):** 75.96% (7,279,376 hours)
- **Wet Hours (&ge; 0.1 mm):** 24.04% (2,303,536 hours)
- **Moderate Rain (&ge; 5.0 mm):** 0.39% (37,373 hours)
- **Heavy Rain (&ge; 15.0 mm):** 0.02% (1,917 hours)
- **Extreme Events (&ge; 30.0 mm):** 24 hours across the entire dataset

### Uncorrected Baseline Performance (Overall 9.58M Rows)
| Model | Usable Pairs | MAE (mm/h) | Mean Bias (mm/h) | Characteristics |
|---|---|---|---|---|
| **GFS Seamless** | 9,582,912 | 0.2154 | -0.0532 | Systematic underforecasting of rainfall |
| **ECMWF IFS 0.25&deg;** | 9,550,146 | 0.2249 | +0.0066 | Lowest overall bias; highest correlation |
| **ICON Seamless** | 8,355,711 | 0.2295 | -0.0134 | 12.8% missingness in forecast cycles |
| **GEM Seamless** | 9,532,620 | 0.2568 | -0.0088 | Highest raw error across lead times |

---

## 2. Unit and Sampling Analysis

1. **Precipitation Units:**
   - The Parquet schema defines `truth_precipitation` and forecast columns as `double` without embedded unit strings.
   - Standard Open-Meteo API outputs and ERA5-Land reanalysis record precipitation in **millimeters of liquid water equivalent accumulated over the preceding 1-hour interval** (mm/h).
   - This prototype treats precipitation as **mm/hour** for runoff volume computations ($1\text{ mm} \times 1\text{ km}^2 = 1{,}000\text{ m}^3$).
   - *Operational Requirement:* Upstream metadata confirmation must be obtained from IMD / Open-Meteo prior to live municipal deployment.

2. **Temporal Resolution vs Nowcasting Scope:**
   - The dataset provides **hourly** snapshots.
   - Traditional radar nowcasting operates at 5–15 minute intervals with Doppler velocity and convective reflectivity fields ($Z \approx a R^b$).
   - Because high-frequency radar imagery is not included in the tabular dataset, this prototype implements **NWP Statistical Post-Processing (Model Output Statistics)** coupled to hourly drainage routing, rather than an optical-flow radar nowcast.

---

## 3. Data Not Supplied (Critical Operational Gaps)

The following operational assets are **not present** in the repository and are required for full municipal deployment:

| Missing Layer | Operational Impact | Recommended Public / Government Source |
|---|---|---|
| **Doppler Weather Radar (DWR)** | Essential for convective storm tracking at 5–10 min intervals | IMD DWR network (e.g., Mumbai Colaba / Veravali, Chennai Sriharikota radar) |
| **Automated Rain Gauge (ARG) Telemetry** | Real-time ground truth for radar calibration | Municipal telemetry (MCGM, GCC, CPCB, IMD automated stations) |
| **High-Resolution DEM / LiDAR (1–5m)** | Required for 2D hydrodynamic overland routing and depression mapping | Survey of India (SoI) National Topographic Database, Cartosat-3 DEM, ISRO Bhuvan |
| **Municipal Storm Sewer GIS** | Conduit diameters, materials, slopes, invert levels, catchpit inlet locations | Municipal Corporation GIS portals (e.g., MCGM Stormwater Drain Dept - SWD) |
| **Ground-Truth Flood Sensors / Labels** | IoT ultrasonic water-level sensors, traffic camera CCTV logs, emergency 112 dispatch calls | Municipal Disaster Management Cell (DMIC) logs, traffic police waterlogging logs |

---

## 4. Synthetic Demonstration Assets (Visibly Labeled)

Because municipal infrastructure layers were not supplied, the drainage demonstration uses **rigorously engineered synthetic assets**:
- **5-Node Catchment Network:** Represents a 10 km² coastal metropolitan pilot (e.g., Mumbai Mithi River basin or Chennai Adyar/Cooum basin) with elevations from 28m down to 2m tidal outfall.
- **Hydraulic Dimensions:** Conduits C1–C4 sized using Manning's equation ($n = 0.015$, concrete pipe/box culvert).
- **Surface Depressions:** Sag depression Node N4 (elevation 7.5m, representing a vulnerable vehicular underpass hotspot).
- **Control Scenarios:** Baseline design storm, 1.75x extreme deluge, conduit C4 siltation blockage (50%), midstream conduit C3 blockage (50%), and coastal high spring tide backwater (60% outfall restriction).
- **Strict Labeling:** Outputs are synthetic accounting demonstrations. No simulated flood depth is an empirical municipal observation or operational prediction.
