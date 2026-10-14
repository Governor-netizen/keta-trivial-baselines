"""Shared paths and helpers for the reproduction scripts."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
G = 9.80665


def rule(title):
    print("\n" + "=" * 78 + "\n  " + title + "\n" + "=" * 78)


def tide_and_waves():
    """Hourly FES2022 tide with WW3 Hs/Tp interpolated onto it, 2018-2026."""
    t = pd.read_csv(DATA / "keta_tide_table.csv.gz", parse_dates=["datetime"])
    w = pd.read_csv(DATA / "ww3_hourly_series.csv.gz", parse_dates=["time"])
    for df, c in ((t, "datetime"), (w, "time")):
        if getattr(df[c].dt, "tz", None) is not None:
            df[c] = df[c].dt.tz_localize(None)
    t = t[(t.datetime >= w.time.min()) & (t.datetime <= w.time.max())]
    i = pd.DatetimeIndex(t.datetime).astype(np.int64)
    j = pd.DatetimeIndex(w.time).astype(np.int64)
    hs = np.interp(i, j, w.hs_m)
    tp = np.interp(i, j, w.tp_s)
    return t.datetime.values, t.combined_m.values, hs, tp


def twl_quantiles_dissipative():
    """TWL quantiles used as model features in Table 2.

    These reproduce the feature set exactly as it was trained. They use the
    dissipative Stockdon form (0.043 sqrt(H0 L0)); the features are freeboards,
    i.e. elevation minus a constant, so the choice only shifts them. The
    physically appropriate general form is used for the TWL range in
    04_twl_and_dem.py.
    """
    _, tide, hs, tp = tide_and_waves()
    runup = 0.043 * np.sqrt(np.maximum(0.0, hs * G * tp ** 2 / (2 * np.pi)))
    twl = tide + runup
    q = {f"twl_p{p}": float(np.percentile(twl, p)) for p in (50, 90, 99)}
    q["twl_max"] = float(twl.max())
    return q
