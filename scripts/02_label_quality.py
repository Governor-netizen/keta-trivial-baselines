#!/usr/bin/env python3
"""Section 3 (Otsu) and Section 4 (variance decomposition): is the event label informative?

Data: data/keta_samples_v5_1_all_splits.csv.gz (24 events x 1,681 locations, with
per-event Sentinel-1 QA fields from gee/keta_flood_pipeline_v5_0.js).

    python scripts/02_label_quality.py
"""
import numpy as np
import pandas as pd

from common import DATA, RESULTS, rule

d = pd.read_csv(DATA / "keta_samples_v5_1_all_splits.csv.gz")
for c in ("flood", "qa_otsu_threshold_db", "is_water", "latitude", "longitude"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
land = d[d.is_water == 0]

rule("Per-event Otsu thresholds and labelled fractions (land)")
ev = d.groupby("event_id").agg(otsu_threshold_db=("qa_otsu_threshold_db", "first"))
ev["land_flood_fraction"] = land.groupby("event_id").flood.mean()
ev = ev.sort_values("otsu_threshold_db")
print(ev.round(3).to_string())
ev.to_csv(RESULTS / "02_events.csv")
print(f"\n  events with a threshold          : {ev.otsu_threshold_db.notna().sum()}   [paper: 24]")
print(f"  positive thresholds             : {(ev.otsu_threshold_db > 0).sum()}   [paper: 11]")
print(f"  land fraction in [0.43, 0.52]   : {ev.land_flood_fraction.between(0.43, 0.52).sum()}   [paper: 20]")
print(f"  sign-only gate (<= 0 dB)        : {(ev.otsu_threshold_db <= 0).sum()}   [paper: 13]")
print(f"  strict gate (<= -0.25 dB)       : {(ev.otsu_threshold_db <= -0.25).sum()}   [paper: 8]")
print(f"  strict-gate events              : {', '.join(ev.index[ev.otsu_threshold_db <= -0.25])}")

rule("Variance decomposition: between-location spread vs binomial noise")
rows = []
dd = land.rename(columns={"longitude": "lon", "latitude": "lat"})
gated = dd[(dd.qa_otsu_threshold_db <= -0.25) & (dd.qa_otsu_threshold_db >= -25.0)]
for label, sub, paper in (("all events", dd, "0.119 / 0.099 / 0.067 / 68%"),
                          ("strict gate", gated, "72%")):
    key = sub[["lon", "lat"]].round(5).astype(str).agg("|".join, axis=1)
    g = sub.assign(_loc=key).groupby("_loc").flood.agg(["mean", "count"])
    g = g[g["count"] >= 3]
    pbar, n = g["mean"].values, g["count"].values
    obs = pbar.std(ddof=1)
    binom = np.sqrt(np.mean(pbar * (1 - pbar) / n))
    true = np.sqrt(max(obs ** 2 - binom ** 2, 0.0))
    share = binom ** 2 / obs ** 2
    rows.append(dict(set=label, locations=len(g), events=sub.event_id.nunique(),
                     observed_sd=obs, binomial_sd=binom, true_sd=true, noise_share=share))
    print(f"  {label:<12}: {len(g):,} locations, {sub.event_id.nunique()} events   "
          f"observed sd {obs:.3f}  binomial sd {binom:.3f}  true sd {true:.3f}  "
          f"noise {100 * share:.1f}%   [paper: {paper}]")
pd.DataFrame(rows).to_csv(RESULTS / "02_variance_decomposition.csv", index=False)
print(f"\n  wrote {RESULTS / '02_events.csv'} and 02_variance_decomposition.csv")
