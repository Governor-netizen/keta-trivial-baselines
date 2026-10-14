#!/usr/bin/env python3
"""Section 4 full-raster numbers: cross-sensor agreement, water-level range, DEM levels.

Data:
  data/keta_inundation_frequency_glo30_30m.tif  bands: inundation_frequency,
      n_observations, elevation (GLO-30), dist_water, gsw_occurrence (JRC)
      (exported by gee/keta_inundation_frequency.js)
  data/keta_dem_comparison_30m.npz              NASADEM and GLO-30 on the same
      1299 x 1300 grid (exported by gee/export_dem_comparison.js; the two grids
      agree to within a fifth of a pixel and are compared cell by cell)
  data/keta_tide_table.csv.gz                   hourly FES2022 tide, 2018-2026
  data/ww3_hourly_series.csv.gz                 WW3 Hs and Tp at 5.5N 1.0E

    python scripts/04_twl_and_dem.py
"""
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import pearsonr

from common import DATA, G, RESULTS, rule, tide_and_waves

BETA_F = 0.08          # foreshore slope; unmeasured at Keta, literature band 0.05-0.12
rows = []


def log(key, value, paper=""):
    rows.append(dict(quantity=key, value=value, paper=paper))
    print(f"  {key:<60} {value:>12,.4f}   {paper}")


rule("Inundation-frequency raster")
a = tifffile.imread(DATA / "keta_inundation_frequency_glo30_30m.tif").astype(np.float64)
freq, nobs, elev, dw, gsw = (a[:, :, i] for i in range(5))
land = np.isfinite(freq) & np.isfinite(elev) & np.isfinite(dw)
log("land cells", float(land.sum()), "693,376")
log("median Sentinel-1 observations per cell", float(np.median(nobs[land])), "503")
r, _ = pearsonr(freq[land], np.nan_to_num(gsw[land]))
log("Pearson r, inundation frequency vs JRC occurrence", r, "0.755")

rule(f"Total water level: FES2022 tide + general Stockdon R2 (beta_f = {BETA_F})")
_, tide, hs, tp = tide_and_waves()
L0 = G * tp ** 2 / (2 * np.pi)
setup = 0.35 * BETA_F * np.sqrt(hs * L0)
swash = np.sqrt(hs * L0 * (0.563 * BETA_F ** 2 + 0.004))
twl = tide + 1.1 * (setup + swash / 2)
xi = BETA_F / np.sqrt(np.maximum(hs, 1e-6) / L0)
print(f"  {len(twl):,} hours")
log("TWL minimum (m)", twl.min())
log("TWL maximum (m)", twl.max())
log("TWL range (m)", twl.max() - twl.min(), "3.98")
log("Iribarren number, 1st percentile", np.percentile(xi, 1), "0.49")
log("Iribarren number, 99th percentile", np.percentile(xi, 99), "1.73")
log("Iribarren number, minimum", xi.min(), "0.36")

rule("Elevation levels inside the water-level range, land cells")
dem = np.load(DATA / "keta_dem_comparison_30m.npz")
lo, hi = twl.min(), twl.max()
for name, key, paper in (("NASADEM", "dem_nasadem", "four"), ("GLO-30", "dem_glo30", "316,207")):
    e = dem[key].astype(np.float64)
    band = e[land & np.isfinite(e) & (e >= lo) & (e <= hi)]
    log(f"{name}: distinct elevations in [{lo:.2f}, {hi:.2f}] m", float(len(np.unique(band))), paper)

pd.DataFrame(rows).to_csv(RESULTS / "04_twl_and_dem.csv", index=False)
print(f"\n  wrote {RESULTS / '04_twl_and_dem.csv'}")
