# Trivial baselines catch label leakage that cross-validation misses

Code and derived data for:

> Jerry Etornam Kudaya. *Trivial baselines catch label leakage that cross-validation
> misses: a flood-mapping case study on a data-scarce West African coast.*
> Tackling Climate Change with Machine Learning workshop, NeurIPS 2026.

Every number in the paper is produced by one of the five scripts below, and each
script prints the paper's value next to the recomputed one.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py         # 20-40 minutes; results go to results/
```

Each script also runs on its own, e.g. `python scripts/01_leak_table1.py`.

## Where each number comes from

| Paper | Quantity | Script |
|---|---|---|
| Abstract, §3 | leaked band fixes the label on 61% of 14,881 rows, no exceptions; 83% of default importance, 47% of total gain | `01_leak_table1.py` |
| Table 1 | model 0.809 / 0.383 / 0.873; one-line rule 0.786 / 0.503 / 0.870; land only 0.514 / 0.032 / 0.515; majority class 0.546 | `01_leak_table1.py` |
| §3 | 81% model/rule agreement; 52% of land flagged, 47% precision, 45% base rate | `01_leak_table1.py` |
| §3 | κ 0.389 (2023) and 0.383 (2024–2026); four families ROC-AUC 0.84–0.87; Brier skill 0.33 | `01_leak_table1.py` |
| §3 | 11 of 24 positive Otsu thresholds; 20 events label 43–52% of land | `02_label_quality.py` |
| §4 | between-location sd 0.119 vs binomial 0.099, true 0.067; 68% noise, 72% at the strict gate | `02_label_quality.py` |
| Table 2(a), §4 | baselines and models; isotonic on elevation 0.487; no-distance ablation 0.484; bootstrap CIs | `03_table2.py` |
| Table 2(b), §4 | NASADEM 0.310, GLO-30 rounded 0.440 (0.377–0.475), GLO-30 0.487; quantisation vs elevation decomposition | `03_table2.py` |
| §4, Fig. 1, App. B | 693,376 land cells; 503 observations per cell; r = 0.755 against JRC occurrence | `04_twl_and_dem.py` |
| §4 | TWL range 3.98 m; Iribarren 0.49–1.73 (1st–99th percentile), minimum 0.36; 4 NASADEM vs 316,207 GLO-30 levels | `04_twl_and_dem.py` |
| Table 3 | land-only, leave-one-event-out comparison under three label-quality gates | `05_table3_gates.py` |

## The bug, in one line

`gee/keta_flood_pipeline_v4_2.js`, line 72:

```js
var distWater = permWater.not().fastDistanceTransform(500).sqrt()
```

`fastDistanceTransform` returns the distance to the nearest **non-zero** pixel, so
applying it to `permWater.not()` gives the distance to the nearest *land*: zero on
all land, positive only over permanent water. The flood label is forced to 0 on
permanent water by the same mask, so the band determines the label. The corrected
line is in `gee/keta_flood_pipeline_v5_0.js` (line 229):

```js
var distWater = permWater.fastDistanceTransform(1024).sqrt()
```

## Contents

```
scripts/   reproduction scripts (01-05), shared helpers, run_all.py
data/      derived inputs (see below)
models/    the 28-feature XGBoost exactly as reported, with its metadata
gee/       Google Earth Engine scripts that exported the data
results/   written by the scripts
```

| File | What it is | Produced by |
|---|---|---|
| `data/keta_samples_v4_2_all_splits.csv` | 14,881 event-pixel samples with the inverted `dist_water` band | `gee/keta_flood_pipeline_v4_2.js` |
| `data/keta_samples_v5_1_all_splits.csv.gz` | corrected samples, 24 events x 1,681 locations, with per-event Sentinel-1 QA | `gee/keta_flood_pipeline_v5_0.js` (v5.1) |
| `data/keta_inundation_frequency_glo30_30m.tif` | 30 m raster: inundation frequency, observations, GLO-30 elevation, distance to water, JRC occurrence | `gee/keta_inundation_frequency.js` |
| `data/keta_inundation_frequency_glo30_samples.csv` | 2,527 sampled cells used in Table 2, GLO-30 elevation | `gee/keta_inundation_frequency.js` |
| `data/keta_inundation_frequency_samples.csv` | the same cells and target with NASADEM elevation | `gee/keta_inundation_frequency.js` |
| `data/keta_dem_comparison_30m.npz` | NASADEM and GLO-30 on the raster grid | `gee/export_dem_comparison.js` |
| `data/keta_tide_table.csv.gz` | hourly FES2022 tide at Keta, 2018–2026 | pyfes over the FES2022 constituents |
| `data/ww3_hourly_series.csv.gz` | WAVEWATCH III Hs and Tp at 5.5°N 1.0°E | PacIOOS ERDDAP `ww3_global` |
| `models/model_a_operational.json` | the 28-feature XGBoost reported in Table 1 | trained on the v4.2 training split |

## What this repository does not show

The paper is a single-site case study. The event labels are Sentinel-1 change
detections, not surveyed flood extents, and Section 3 shows they are close to
arbitrary; the corrected target (long-run inundation frequency) is a different
quantity from event-scale flood susceptibility. Nothing here is an operational
flood forecast.

## Data sources and licences

Code: MIT (see `LICENSE`). Derived data in `data/` and `results/`: CC BY 4.0,
subject to the source terms below.

- Sentinel-1 GRD: Copernicus Sentinel data 2015–2026, via Google Earth Engine.
- JRC Global Surface Water: Pekel et al. (2016), *Nature* 540:418–422.
- Copernicus DEM GLO-30: © DLR e.V. 2010–2014 and © Airbus Defence and Space
  GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all
  rights reserved. https://doi.org/10.5270/ESA-c5d3d65
- NASADEM: NASA JPL, Crippen et al. (2016).
- FES2022: Carrere et al. (2022), distributed by AVISO+. The constituent files are
  not redistributed; only a derived hourly tide series at one point is included.
- WAVEWATCH III: NOAA/NCEP global hindcast via PacIOOS.
- ERA5: contains modified Copernicus Climate Change Service information 2026.
- GPM IMERG (NASA), CHIRPS (UCSB Climate Hazards Center), SMAP L4 (NASA).

## Citation

See `CITATION.cff`.
