# Mathematical Formulation and Model Methodology

This document details the scientific, statistical, and engineering formulations implemented in the **Urban Flood Nowcasting System (SIH26085)** prototype.

---

## 1. Machine Learning Precipitation Calibration (MOS)

### A. Problem Definition & Objectives
- **Target:** Hourly precipitation accumulation $Y \in [0, \infty)$ in mm/h, evaluated against ERA5-Land reanalysis reference (not independent station truth).
- **Predictors:** Raw multi-model NWP precipitation forecasts from GFS ($f_{\text{gfs}}$), ECMWF ($f_{\text{ecmwf}}$), ICON ($f_{\text{icon}}$), and GEM ($f_{\text{gem}}$), plus ensemble statistics, temporal cycles, spatial coordinates, and missingness metadata.
- **Prototype Purpose:** Compare statistical calibration approaches and provide a reproducible rainfall input to the synthetic drainage demonstration.
- **Scope Boundary:** This is a Model Output Statistics (MOS) / statistical downscaling model for tabular NWP, not an optical-flow radar nowcasting model.

### B. Feature Engineering
For each observation at timestamp $t$, location $(\phi, \lambda, z)$, and lead time $L$:
1. **Raw Forecasts:** $[f_{\text{gfs}}, f_{\text{ecmwf}}, f_{\text{icon}}, f_{\text{gem}}]$.
2. **Ensemble Statistics:**
   $$\mu_{\text{ens}} = \frac{1}{K}\sum_{k=1}^K f_k, \quad \sigma_{\text{ens}} = \text{std}(f_k), \quad \min(f_k), \quad \max(f_k)$$
   $$\text{Wet Fraction} = \frac{1}{K}\sum_{k=1}^K \mathbf{1}[f_k \ge 0.1\text{ mm}]$$
3. **Temporal Cycle Features:**
   $$h_{\sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad h_{\cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
   $$d_{\sin} = \sin\left(\frac{2\pi \cdot \text{DOY}}{366}\right), \quad d_{\cos} = \cos\left(\frac{2\pi \cdot \text{DOY}}{366}\right)$$
4. **Spatial Features:** Latitude $\phi$, Longitude $\lambda$, Elevation $z$ (in km).
5. **Forecast Lead Time:** $L$ (hours from initialization).
6. **Data Quality Indicator:** Count of missing forecasts in current ensemble row $N_{\text{missing}} \in \{0, 1, 2, 3\}$.

### C. Two-Stage Hurdle Model Formulation
Hourly Indian precipitation distributions exhibit an extreme spike at zero ($\approx 76\%$ dry hours) and a heavy, positive, continuous tail. A single regressor suffers from variance collapse and severe underforecasting of extreme convective rain.

We formulate a **Two-Stage Hurdle Model**:
1. **Stage 1 (Occurrence Classification):**
   A Gradient Boosted Classifier ($H_c$) predicts the probability of measurable precipitation ($Y \ge 0.1\text{ mm}$):
   $$p_{\text{wet}} = P(Y \ge 0.1 \mid X) = H_c(X)$$
2. **Stage 2 (Positive Amount Regression):**
   Trained strictly on positive precipitation rows ($Y \ge 0.1\text{ mm}$). To stabilize variance and linearize exponential upper tails, target $Y$ is transformed into cube-root space:
   $$z = Y^{1/3}$$
   The regressor ($H_r$) fits $z$ using mean squared error loss:
   $$\hat{z} = H_r(X \mid Y \ge 0.1)$$
   The inverse transform maps back to physical precipitation space:
   $$\hat{Y}_{\text{pos}} = \max(\hat{z}, 0)^3$$
3. **Hurdle Prediction Output:**
   $$\hat{Y} = \begin{cases} \hat{Y}_{\text{pos}} & \text{if } p_{\text{wet}} \ge 0.50 \\ 0.0 & \text{if } p_{\text{wet}} < 0.50 \end{cases}$$

   The wet threshold and probability decision threshold are command-line
   parameters. Results for probability thresholds 0.3, 0.5, and 0.7 are saved
   under `probability_threshold_sensitivity` in the metrics JSON.

### D. Leakage-Safe Splitting Design
Partitions are defined before sampling: a chronological cutoff at the
reconstructed forecast origin `valid_time - lead_hours`, plus a deterministic
20% complete-location spatial holdout. Every row from one forecast-origin cycle
is kept on one temporal side. Samples are then drawn independently from
training, future-time seen-location test, and spatial holdout partitions. The
exact row counts, origin ranges, and disjointness check are recorded in
`outputs/precipitation_metrics.json`.

---

## 2. Hydrologic Runoff Generation

For each subcatchment node $i$ with area $A_i$ and impervious fraction $f_{\text{imp}, i}$:
1. **Effective Runoff Coefficient:**
   $$C_{\text{eff}, i} = f_{\text{imp}, i} \cdot C_{\text{imp}} + (1 - f_{\text{imp}, i}) \cdot C_{\text{perv}}$$
   where $C_{\text{imp}} = 0.85$ and $C_{\text{perv}} = 0.25$.
2. **Finite depression storage:** Impervious and pervious depression stores are initialized empty and retain water between timesteps up to 1.5 mm and 5.0 mm respectively. Each timestep's rainfall first fills the remaining depression storage; only the remainder contributes to runoff. This is a finite reservoir, not an initial abstraction re-applied independently to every timestep.
3. **Runoff volume generation ($m^3$):** Effective rainfall is multiplied by $C_{\text{eff},i}$ and catchment area. Rainfall volume not routed as runoff is tracked as loss, including depression storage and the runoff-coefficient loss.

---

## 3. Directed Hydraulic Drainage Routing & Surcharge Coupling

### A. Pipe & Culvert Gravity Conveyance
For each conduit $k$ connecting node $u \to v$ with slope $S_k$, Manning's roughness $n = 0.015$, and blockage fraction $\beta_k \in [0, 1)$:
1. **Cross-Section Area & Hydraulic Radius:**
   - Circular pipe (diameter $D$): $A_0 = \frac{\pi D^2}{4}, \quad R_h = \frac{D}{4}$
   - Box culvert (width $W$, height $H$): $A_0 = W \cdot H, \quad R_h = \frac{W \cdot H}{2(W + H)}$
2. **Manning Full-Flow Discharge Capacity:**
   $$Q_{\text{full}, k} = \frac{1}{n} A_0 R_h^{2/3} S_k^{1/2}$$
   $$Q_{\text{eff}, k} = Q_{\text{full}, k} \cdot (1 - \beta_k)$$
3. **Step Conveyance Volume Capacity ($m^3$):**
   $$V_{\text{cap}, k} = Q_{\text{eff}, k} \cdot \Delta t$$

### B. Manhole Surcharge & Surface Ponding
At any timestep $t$:
1. Total inflow entering node $u$ comprises local surface runoff, incoming conduit flows from upstream nodes, and carryover surface ponding:
   $$V_{\text{inflow}, u}(t) = V_{\text{runoff}, u}(t) + \sum_{j \in \text{in}(u)} V_{\text{conduit}, j}(t) + V_{\text{storage}, u}(t-1)$$
2. Conveyance into outgoing conduit $k$:
   $$V_{\text{conduit}, k}(t) = \min\left(V_{\text{inflow}, u}(t), V_{\text{cap}, k}(t)\right)$$
3. **Surcharge Accumulation:**
   If $V_{\text{inflow}, u}(t) > V_{\text{cap}, k}(t)$, excess water accumulates at the surface depression until the configured finite storage cap.
4. **Surface Flood Depth ($m$ and $cm$):**
   $$V_{\text{storage}, u}(t) = \min(V_{\text{available}, u}(t), A_{\text{ponding}, u} d_{\max})$$
   $$V_{\text{spill}, u}(t) = \max(0, V_{\text{available}, u}(t) - V_{\text{storage}, u}(t)),\quad d_u(t) = \frac{V_{\text{storage}, u}(t)}{A_{\text{ponding}, u}}$$

### C. Outfall Boundary Condition & Tidal Backwater
At coastal outfall node $N_5$, discharge into receiving water body (sea/creek) is governed by outfall capacity $Q_{\text{outfall}}$ and tidal restriction fraction $\theta_{\text{tide}} \in [0, 1)$:
$$V_{\text{discharged}}(t) = \min\left(V_{\text{inflow}, N_5}(t), Q_{\text{outfall}} \cdot (1 - \theta_{\text{tide}}) \cdot \Delta t\right)$$

---

## 4. Mass Balance Accounting & Sanity Checks

### A. Volume accounting check
Over $T$ timesteps, input rainfall volume should be accounted for as modeled losses, outfall discharge, final node storage, and overflow spill:
$$\sum_{t=1}^T \sum_i V_{\text{precip},i}(t) \approx \sum_{t=1}^T \sum_i V_{\text{loss},i}(t) + \sum_{t=1}^T V_{\text{discharged}}(t) + \sum_i V_{\text{storage},i}(T) + \sum_{t=1}^T V_{\text{spill}}(t)$$
Here $V_{\text{spill}}(t)$ is the incremental spill during timestep $t$; it is not cumulative. Relative accounting error:
$$\epsilon_{\text{rel}} = \frac{\left| \sum V_{\text{precip}} - \left(\sum V_{\text{loss}} + \sum V_{\text{discharged}} + \sum V_{\text{storage}} + \sum V_{\text{spill}}\right) \right|}{\sum V_{\text{precip}}} < 10^{-5}$$

Surface storage is capped at the configured maximum depth and excess is
reported as explicit spill. These are accounting and sanity checks, not
validation of physical street-water depths. Full-flow Manning capacity and
instant routing are deliberate simplifications.

### B. Scenario monotonicity checks
1. **Rainfall Monotonicity:** $\frac{\partial d_u}{\partial P} \ge 0 \quad \forall u$.
2. **Blockage Monotonicity:** $\frac{\partial d_u}{\partial \beta_k} \ge 0 \quad \text{for upstream nodes } u$.
3. **Tidal Backwater Monotonicity:** $\frac{\partial d_{N_4}}{\partial \theta_{\text{tide}}} \ge 0$.
4. **Compound Multi-Hazard Dominance:**
   $$d_{\text{compound}}(t) \ge \max\left(d_{\text{baseline}}(t), d_{\text{heavy}}(t), d_{\text{block}}(t), d_{\text{tide}}(t)\right)$$

The scenario audit also compares the peak surface burden, defined as cumulative
overflow spill plus surface water still stored at that timestep. Peak stored
volume by itself is not a monotonic measure: blockage can move ponding upstream,
while a 1 m storage cap converts additional water into spill. These are synthetic
scenario checks, not universal hydraulic guarantees.
