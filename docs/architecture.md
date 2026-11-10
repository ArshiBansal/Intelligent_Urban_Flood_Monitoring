# System Architecture, Interfaces, and Literature References

This document defines the modular pipeline architecture, interface contracts, and scientific literature foundation of the **Urban Flood Nowcasting System (SIH26085)** prototype.

---

## 1. System Pipeline Architecture

```text
[ Stage 1: Meteorological Ingestion ]
  * Multi-model NWP historical forecasts (GFS, ECMWF, ICON, GEM)
  * Hourly ERA5-Land reanalysis truth
  * [Future Plugin]: IMD Doppler Weather Radar (DWR) 5-min reflectivity grids
                 │
                 ▼
[ Stage 2: Audit & Leakage-Safe Stratification ]
  * Columnar PyArrow audit (9,582,912 rows, 127 district HQs)
  * Deterministic spatial holdout (20% complete locations across India)
  * Chronological cutoff (70% timestamp split before 2026-05-03; samples are drawn after splitting)
                 │
                 ▼
[ Stage 3: Statistical Post-Processing / Calibration ]
  * Two-Stage Hurdle Model (HistGradientBoostingClassifier + Regressor)
  * Cube-root transform on positive rainfall to handle zero-spike & heavy tails
  * Point-forecast evaluation across lead times (0–191h) & spatial holdouts
                 │
                 ▼
[ Stage 4: Catchment Hydrologic Runoff Generation ]
  * Subcatchment partitioning (area, impervious fraction, depression storage)
  * Initial abstraction losses ($I_a = f_{\text{imp}} D_{\text{imp}} + f_{\text{perv}} D_{\text{perv}}$)
  * Volumetric runoff calculation ($V_{\text{runoff}} = \max(0, P - I_a) \cdot C_{\text{eff}} \cdot A$)
                 │
                 ▼
[ Stage 5: Directed Hydraulic Conduit Network Routing ]
  * Directed graph of conduits and junctions (Manning's open/closed channel formulation)
  * Capacity exceedance detection ($Q > Q_{\text{cap}} \cdot (1 - \beta)$)
  * Tailwater / coastal high tide backwater boundary conditions at Outfall N5
                 │
                 ▼
[ Stage 6: Manhole Surcharge & Surface Sag Ponding ]
  * Surcharge mass balance ($V_{\text{storage}}(t) = V_{\text{inflow}} - V_{\text{conduit}}$)
  * Low-point surface depression accumulation ($d = V_{\text{storage}} / A_{\text{ponding}}$)
  * Multi-tier categorical risk alerting (LOW: <5cm, MEDIUM: 5–20cm, HIGH: &ge;20cm)
                 │
                 ▼
[ Stage 7: Interactive Dashboard & Map Visualization ]
  * Dynamic SVG drainage topology with real-time surcharge & blockage rendering
  * Controlled hydraulic scenario comparisons (Deluge, Blockage, High Tide, Compound)
  * Illustrative Mumbai historical comparison from one forecast origin (2026-07-05)
```

---

## 2. Modular Interface Contracts (Future Pluggability)

The system is architected as loosely coupled, independently testable Python modules adhering to explicit interface contracts:

1. **Precipitation Engine Adapter (`src/floodnowcast/data.py` & `calibration.py`):**
   - *Input:* Tabular dataframe with timestamps, coordinates, and forecast columns (or 2D numpy rain rate grid `R(x, y, t)`).
   - *Output:* Calibrated deterministic rainfall series `P_cal(t)` (mm/h) and precipitation probability $P(Y \ge 0.1)$.
   - *Future Drop-In:* Optical flow / ConvLSTM / pysteps / DGMR radar nowcasting module.
2. **Runoff & Drainage Coupling Engine (`src/floodnowcast/drainage.py`):**
   - *Input:* 1D array of rainfall intensity $P(t)$, network topology graph `G(V, E)`, blockage vector $\vec{\beta}$, and tidal boundary condition $\theta_{\text{tide}}$.
   - *Output:* Pandas DataFrame with step-by-step conduit flows, surcharge states, surface storage volumes, and ponding depths.
   - *Future Drop-In:* EPA SWMM 5.2 engine (via `pyswmm`) or 2D shallow water hydrodynamic solver (e.g., LISFLOOD-FP, TELEMAC-2D).
3. **Dashboard & Alert Engine (`scripts/make_dashboard.py`):**
   - *Input:* Standardized JSON metrics and scenario CSV time series.
   - *Output:* Self-contained HTML/SVG visual dashboard requiring zero external server dependencies.

---

## 3. Scientific Literature & Bibliography

The system architecture and model choices are grounded in established peer-reviewed hydrometeorological and urban water management research:

### A. Rainfall Nowcasting & NWP Post-Processing
1. **Ravuri, S., et al. (2021).** "Skilful precipitation nowcasting using deep generative models of radar." *Nature*, 597(7878), 672–677. [DOI: 10.1038/s41586-021-03854-z](https://doi.org/10.1038/s41586-021-03854-z).
   *Significance:* Demonstrates radar nowcasting using spatio-temporal generative models and justifies evaluating CSI across multiple intensity thresholds.
2. **Pulkkinen, S., et al. (2019).** "Pysteps: an open-source Python library for probabilistic precipitation nowcasting." *Geoscientific Model Development*, 12(10), 4185–4219. [DOI: 10.5194/gmd-12-4185-2019](https://doi.org/10.5194/gmd-12-4185-2019).
   *Significance:* Industry-standard reference for Eulerian/Lagrangian advection nowcasting and verification metrics (CSI, FAR, Brier score).
3. **Shi, X., et al. (2015).** "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." *Advances in Neural Information Processing Systems (NeurIPS)*, 28, 802–810.
   *Significance:* Foundational deep learning architecture establishing nowcasting as a spatiotemporal sequence forecasting problem.
4. **Zhang, Y., et al. (2023).** "Skilful nowcasting of extreme precipitation with NowcastNet." *Nature*, 619(7970), 526–532. [DOI: 10.1038/s41586-023-06184-4](https://doi.org/10.1038/s41586-023-06184-4).
   *Significance:* Combines physical advection constraints with generative learning for extreme convective rainfall events.

### B. Rainfall Nowcasts Coupled to Urban Drainage & Flood Models
5. **Thorndahl, S., et al. (2017).** "Weather radar rainfall data in urban hydrology." *Hydrology and Earth System Sciences*, 21, 1359–1380. [Publisher](https://hess.copernicus.org/articles/21/1359/2017/).
   *Significance:* Comprehensive review detailing the propagation of rainfall nowcasting uncertainty into urban stormwater drainage networks.
6. **Schellart, A. N. A., et al. (2011).** "Sensitivity to spatial and temporal resolution of rainfall in urban drainage modeling." *Water Resources Research*, 48(7), W07505. [DOI: 10.1029/2011WR011211](https://doi.org/10.1029/2011WR011211).
   *Significance:* Quantifies the impact of 1-hour vs sub-hourly rainfall aggregation on peak pipe flow and street surcharge estimation.
7. **Ochoa-Rodriguez, S., et al. (2015).** "Impact of spatial and temporal resolution of rainfall inputs on urban hydrodynamic modeling: A multi-catchment investigation." *Journal of Hydrology*, 531, 389–407. [DOI: 10.1016/j.jhydrol.2015.05.035](https://doi.org/10.1016/j.jhydrol.2015.05.035).
   *Significance:* Establishes scale guidelines: urban catchments < 10 km² require temporal resolution $\le 15$ min for precise manhole surcharge onset.

### C. Machine Learning Surrogates for Street-Scale Flooding
8. **McSpadden citation:** Unverified against the supplied reference list and therefore not retained as a scientific citation here.
9. **Zahura, F. T., et al. (2020).** "Training machine learning surrogate models for street-scale urban flood depth prediction." *Frontiers in Water*, 2, 28. [DOI: 10.3389/frwa.2020.00028](https://doi.org/10.3389/frwa.2020.00028).
   *Significance:* Demonstrates that hydraulic connectivity and pipe capacities govern low-point ponding depth more strongly than surface terrain slope alone.

### D. Indian Urban Flood Applications (Coastal Metros)
10. **Ghosh, S., et al. (2019).** "Assessment of urban flood hazard and vulnerability: Chennai case study." *Current Science*, 117(5), 843–853.
    *Significance:* Analyzes the 2015 Chennai catastrophic flood, documenting how storm sewer capacity deficits and tidal backwater cause severe sag underpass waterlogging.
11. **Ghosh, S., et al. (2021).** "Multi-hazard urban flood risk analysis for Mumbai under climate change and coastal surge." *Journal of Hydrology*, 595, 126020. [DOI: 10.1016/j.jhydrol.2021.126020](https://doi.org/10.1016/j.jhydrol.2021.126020).
    *Significance:* Formulates the compound disaster mechanism: monsoon rainfall cloudburst coinciding with high spring tide and Mithi River silted outfall blockage.
12. **Mujumdar, P. P., et al. (2021).** "Urban flood vulnerability and risk assessment for Bengaluru." *Sadhana - Academy Proceedings in Engineering Sciences*, 46(3), 142. [DOI: 10.1007/s12046-021-01662-w](https://doi.org/10.1007/s12046-021-01662-w).
    *Significance:* Evaluates rapid urban land-use changes, loss of natural lake chains (keres), and imperviousness growth in inland metropolitan India.
