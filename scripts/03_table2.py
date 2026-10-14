#!/usr/bin/env python3
"""
Two checks on Table 2(b) of the TCCML paper, run in the exact harness that
produced it (archive/legacy_scripts/keta_piml_susceptibility.py: 2,527 cells,
6x6 quantile spatial blocks, 8 features, same XGBoost settings).

  A. Trivial baselines the paper did not report. Does the 8-feature monotone
     model beat elevation alone? Does it need distance-to-water at all?
     (Reviewer sbHb: "no ablation without that feature".)

  B. Rounded-DEM test. NASADEM -> GLO-30 raised R2 0.310 -> 0.487, but GLO-30
     differs from NASADEM in BOTH vertical accuracy and quantisation (NASADEM is
     whole metres). Rounding GLO-30 to whole metres keeps its vertical accuracy
     and gives it NASADEM's quantisation, which separates the two.
     (Reviewer sbHb: "DEM improvement confounds vertical accuracy with
     quantization".)

Differences are reported with a paired block bootstrap over the spatial blocks
(out-of-fold predictions held fixed, blocks resampled with replacement).

    python scripts/03_table2.py
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from common import DATA, RESULTS, SEED, rule, twl_quantiles_dissipative

warnings.filterwarnings("ignore")
SEEDS = (42, 7, 123, 2026, 31337)
N_BOOT = 2000
GQ = twl_quantiles_dissipative()


def blocks(df, n=6):
    qx = pd.qcut(df.lon, n, labels=False, duplicates="drop").astype(int)
    qy = pd.qcut(df.lat, n, labels=False, duplicates="drop").astype(int)
    return qx * n + qy


def prep(path, elev_transform=None):
    """Same feature construction as keta_piml_susceptibility.main()."""
    d = pd.read_csv(path).rename(columns={"longitude": "lon", "latitude": "lat"})
    if elev_transform is not None:
        d["elevation"] = elev_transform(d.elevation.values)
    for k, v in GQ.items():
        d[f"freeboard_{k}"] = d.elevation.values - v
    d["below_p99"] = (d.elevation.values < GQ["twl_p99"]).astype(int)
    d["hydraulic_dist"] = d.dist_water
    d["_blk"] = blocks(d)
    return d


FULL = (["elevation", "dist_water"] + [f"freeboard_{k}" for k in GQ]
        + ["below_p99", "hydraulic_dist"])
NO_DIST = ["elevation"] + [f"freeboard_{k}" for k in GQ] + ["below_p99"]


def mono_of(feats):
    return tuple(-1 if (c.startswith("freeboard") or c in
                        ("elevation", "dist_water", "hydraulic_dist")) else 0
                 for c in feats)


def xgb(feats, mono=True, seed=SEED):
    kw = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=3.0, random_state=seed, verbosity=0)
    if mono:
        kw["monotone_constraints"] = str(mono_of(feats))
    return lambda: XGBRegressor(**kw)


class Iso:
    """Decreasing isotonic regression on a single column."""
    def __init__(self):
        self.m = IsotonicRegression(increasing=False, out_of_bounds="clip")

    def fit(self, X, y):
        self.m.fit(np.asarray(X).ravel(), y)
        return self

    def predict(self, X):
        return self.m.predict(np.asarray(X).ravel())


def cv(d, feats, make):
    y = d.inundation_frequency.values
    blk = d._blk.values
    p = np.zeros(len(y))
    for b in np.unique(blk):
        te, tr = blk == b, blk != b
        if tr.sum() < 50:
            p[te] = y[tr].mean() if tr.sum() else y.mean()
            continue
        p[te] = make().fit(d.loc[tr, feats], y[tr]).predict(d.loc[te, feats])
    return p


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def boot_diff(d, pa, pb, n=N_BOOT, seed=SEED):
    """Paired block bootstrap of R2(pa) - R2(pb)."""
    y = d.inundation_frequency.values
    blk = d._blk.values
    ub = np.unique(blk)
    idx = {b: np.flatnonzero(blk == b) for b in ub}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        s = np.concatenate([idx[b] for b in rng.choice(ub, len(ub), replace=True)])
        out[i] = r2(y[s], pa[s]) - r2(y[s], pb[s])
    return r2(y, pa) - r2(y, pb), np.percentile(out, [2.5, 97.5])


def seed_spread(d, feats, mono=True):
    y = d.inundation_frequency.values
    v = np.array([r2(y, cv(d, feats, xgb(feats, mono, s))) for s in SEEDS])
    return v.mean(), v.min(), v.max()


rows = []


def log(section, name, value, lo=np.nan, hi=np.nan, note=""):
    rows.append(dict(section=section, model=name, r2=value, lo=lo, hi=hi, note=note))
    rng = f"  [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else ""
    print(f"  {name:<52} {value:+.3f}{rng}  {note}")


glo = DATA / "keta_inundation_frequency_glo30_samples.csv"
nas = DATA / "keta_inundation_frequency_samples.csv"
G = prep(glo)
N = prep(nas)
y = G.inundation_frequency.values
assert np.allclose(y, N.inundation_frequency.values), "targets differ between files"
assert np.allclose(G.dist_water.values, N.dist_water.values), "dist_water differs"
print(f"  {len(G):,} cells, {G._blk.nunique()} spatial blocks, "
      f"{len(FULL)} features: {', '.join(FULL)}")
print(f"  dist_water on land cells: min {G.dist_water.min():.1f} m, "
      f"median {G.dist_water.median():.1f} m (a real distance, not the inverted band)")

# ---------------------------------------------------------------- A
rule("A0. Reproduce Table 2(b), GLO-30   [paper: 0.000 / 0.132 / 0.217 / 0.451 / 0.487]")
log("A0", "predict the mean", r2(y, np.full(len(y), y.mean())))
log("A0", "OLS on distance to water", r2(y, cv(G, ["dist_water"], LinearRegression)))
log("A0", "linear, all features", r2(y, cv(G, FULL, LinearRegression)))
log("A0", "XGBoost unconstrained, all features", r2(y, cv(G, FULL, xgb(FULL, False))))
p_full = cv(G, FULL, xgb(FULL))
log("A0", "XGBoost monotone, all features", r2(y, p_full))

rule("A. Trivial baselines and the distance-to-water ablation, GLO-30")
log("A", "OLS on elevation", r2(y, cv(G, ["elevation"], LinearRegression)))
p_iso = cv(G, ["elevation"], Iso)
log("A", "isotonic on elevation (one feature, no ML)", r2(y, p_iso))
p_iso_d = cv(G, ["dist_water"], Iso)
log("A", "isotonic on distance to water", r2(y, p_iso_d))
p_elev = cv(G, ["elevation"], xgb(["elevation"]))
log("A", "XGBoost monotone, elevation only", r2(y, p_elev))
p_nod = cv(G, NO_DIST, xgb(NO_DIST))
log("A", "XGBoost monotone, all features minus distance", r2(y, p_nod),
    note="(sbHb ablation)")
p_2 = cv(G, ["elevation", "dist_water"], xgb(["elevation", "dist_water"]))
log("A", "XGBoost monotone, elevation + distance", r2(y, p_2))

print("\n  paired block bootstrap, full monotone model minus ...")
for name, pb in (("isotonic on elevation", p_iso),
                 ("XGBoost monotone, elevation only", p_elev),
                 ("XGBoost monotone, minus distance", p_nod)):
    dlt, (lo, hi) = boot_diff(G, p_full, pb)
    log("A-diff", f"full - {name}", dlt, lo, hi)

print(f"\n  seed spread over {len(SEEDS)} seeds (mean, min..max)")
for name, feats in (("XGBoost monotone, all features", FULL),
                    ("XGBoost monotone, elevation only", ["elevation"]),
                    ("XGBoost monotone, minus distance", NO_DIST)):
    m, lo, hi = seed_spread(G, feats)
    log("A-seeds", name, m, lo, hi, note="(range over seeds)")

# ---------------------------------------------------------------- B
rule("B. Rounded-DEM test   [paper: NASADEM 0.310 -> GLO-30 0.487]")
e = G.elevation.values
for lo_, hi_ in ((-1, 3), (e.min(), e.max())):
    band = (e >= lo_) & (e <= hi_)
    print(f"  distinct elevations in [{lo_:.1f}, {hi_:.1f}] m: GLO-30 "
          f"{len(np.unique(e[band])):,}   NASADEM "
          f"{len(np.unique(N.elevation.values[band])):,}   "
          f"GLO-30 rounded {len(np.unique(np.round(e[band]))):,}")
diff = N.elevation.values - G.elevation.values
print(f"  NASADEM - GLO-30: mean {diff.mean():+.2f} m, sd {diff.std():.2f} m, "
      f"r {np.corrcoef(N.elevation, G.elevation)[0, 1]:.3f}")
diff_r = N.elevation.values - np.round(G.elevation.values)
print(f"  NASADEM - round(GLO-30): mean {diff_r.mean():+.2f} m, sd {diff_r.std():.2f} m, "
      f"identical on {100 * (diff_r == 0).mean():.1f}% of cells\n")

variants = {
    "NASADEM (whole metres)": N,
    "GLO-30": G,
    "GLO-30 rounded to whole metres": prep(glo, np.round),
}
for off in (0.25, 0.5, 0.75):
    variants[f"GLO-30 rounded, bin edges shifted {off:.2f} m"] = prep(
        glo, lambda v, o=off: np.round(v + o) - o)

preds = {}
for name, d in variants.items():
    pf = cv(d, FULL, xgb(FULL))
    pi = cv(d, ["elevation"], Iso)
    preds[name] = pf
    log("B", f"{name}: monotone, all features", r2(y, pf))
    log("B", f"{name}: isotonic on elevation", r2(y, pi))

print("\n  paired block bootstrap, monotone all-features model")
for a, b in (("GLO-30", "NASADEM (whole metres)"),
             ("GLO-30", "GLO-30 rounded to whole metres"),
             ("GLO-30 rounded to whole metres", "NASADEM (whole metres)")):
    dlt, (lo, hi) = boot_diff(G, preds[a], preds[b])
    log("B-diff", f"{a}  minus  {b}", dlt, lo, hi)

out = RESULTS / "03_table2.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\n  wrote {out}")
