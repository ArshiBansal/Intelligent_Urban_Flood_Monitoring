# Urban Flood Nowcasting System (SIH26085) — reproducible prototype

This repository contains two explicitly separate demonstrations:

1. A precipitation Model Output Statistics (MOS) model trained on multi-model
   historical NWP forecasts and evaluated against ERA5-Land reanalysis.
2. A synthetic, config-driven rainfall/runoff/drainage demonstration. Its
   depths are accounting outputs, not observed or operational street-flood
   predictions.

The dataset audit covers **9,582,912 rows**. Training and evaluation use
deterministic samples taken *after* chronological and spatial partitions are
defined; the exact boundaries and counts are saved in
`outputs/precipitation_metrics.json`.

## Reproduce from PowerShell

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\audit_dataset.py
.\.venv\Scripts\python.exe scripts\train_precipitation.py --max-rows 150000
.\.venv\Scripts\python.exe scripts\run_flood_demo.py
.\.venv\Scripts\python.exe scripts\make_dashboard.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The training script writes `outputs/precipitation_model.pkl`, including the
model, feature names, wet threshold, probability threshold, and seed. The
Mumbai output is an illustrative historical comparison using one forecast
origin (`2026-07-05 00:00 UTC`) and contiguous lead hours; it is not a
nowcast. ERA5-Land is a reanalysis reference, not independent station truth.

## Results

The following table is regenerated from
[outputs/precipitation_metrics.json](outputs/precipitation_metrics.json);
values are MAE in mm/h and each row states its lead range.

| Partition | Lead range | Raw ensemble MAE | Calibrated MAE |
|---|---:|---:|---:|
| Future-time, seen locations | 0–3 h | 0.280480 | 0.253196 |
| Future-time, seen locations | 0–6 h | 0.270110 | 0.245346 |
| Future-time, seen locations | 7–24 h | 0.332061 | 0.276907 |
| Future-time, seen locations | 25–72 h | 0.330934 | 0.282092 |
| Future-time, seen locations | 73–191 h | 0.341755 | 0.303710 |
| Spatial holdout | 0–3 h | 0.140818 | 0.130039 |
| Spatial holdout | 0–6 h | 0.140385 | 0.131860 |
| Spatial holdout | 7–24 h | 0.175780 | 0.152575 |
| Spatial holdout | 25–72 h | 0.179158 | 0.158564 |
| Spatial holdout | 73–191 h | 0.187604 | 0.169035 |

These metrics use the 0.1 mm wet threshold and 0.5 probability decision
threshold. The forecast-origin split cutoff is 2026-05-01 00:00 UTC. The full JSON also contains individual NWP baselines, climatology,
RMSE, bias, CSI, Brier score, sample counts, and boundaries.

## Evidence boundaries

The prototype has no city-scale radar observations, verified municipal
drainage assets, high-resolution terrain, or observed street-flood labels.
The drainage solver uses full-flow Manning capacities, instant directed
routing, finite surface storage with explicit spill, and a receiving-water
outfall restriction. These are documented simplifications, not a high-fidelity
hydraulic solver. Available scripts form a reproducible pipeline; the project
does not claim complete SIH system validation.
