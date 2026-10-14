#!/usr/bin/env python3
"""Section 3 and Table 1: the leak, the one-line rule, and the checks it passed.

Data: data/keta_samples_v4_2_all_splits.csv (14,881 rows exported by
gee/keta_flood_pipeline_v4_2.js, which contains the inverted distance transform).
Model: models/model_a_operational.json, the 28-feature XGBoost as reported.

    python scripts/01_leak_table1.py
"""
import json
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, cohen_kappa_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import DATA, MODELS, RESULTS, SEED, rule

warnings.filterwarnings("ignore")

meta = json.loads((MODELS / "model_a_metadata.json").read_text())
F = meta["feature_order"]
d = pd.read_csv(DATA / "keta_samples_v4_2_all_splits.csv")
for c in F + ["flood"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=F + ["flood"])
d["flood"] = d.flood.astype(int)
tr, te, va = (d[d.split == s] for s in ("train", "test", "validation"))

model = XGBClassifier()
model.load_model(MODELS / "model_a_operational.json")

rows = []


def log(key, value, paper=""):
    rows.append(dict(quantity=key, value=value, paper=paper))
    print(f"  {key:<58} {value:>10.4f}   {paper}")


rule("The leak")
water = d.dist_water > 0
print(f"  rows: {len(d):,}")
log("share of rows with dist_water > 0", water.mean(), "61%")
log("rows with dist_water > 0 and flood = 1 (exceptions)", float((water & (d.flood == 1)).sum()), "0")
for kind, paper in (("gain", "83% (XGBoost default importance)"), ("total_gain", "47%")):
    s = model.get_booster().get_score(importance_type=kind)
    s = {F[int(k[1:])] if k[1:].isdigit() else k: v for k, v in s.items()}
    log(f"dist_water share of {kind} importance", s["dist_water"] / sum(s.values()), paper)
    nxt = sorted((v for k, v in s.items() if k != "dist_water"), reverse=True)[0]
    log(f"  next feature's share of {kind}", nxt / sum(s.values()), "")

rule("Table 1 (validation 2024-2026)")
y = va.flood.values
p = model.predict_proba(va[F].values)[:, 1]
r = (va.dist_water.values <= 0).astype(int)
land = va.dist_water.values == 0
log("28-feature model: accuracy", accuracy_score(y, p >= .5), "0.809")
log("28-feature model: kappa", cohen_kappa_score(y, (p >= .5).astype(int)), "0.383")
log("28-feature model: ROC-AUC", roc_auc_score(y, p), "0.873")
log("one-line rule dist_water <= 0: accuracy", accuracy_score(y, r), "0.786")
log("one-line rule: kappa", cohen_kappa_score(y, r), "0.503")
log("one-line rule: ROC-AUC", roc_auc_score(y, r), "0.870")
log("model/rule agreement", np.mean((p >= .5).astype(int) == r), "81%")
yl, pl = y[land], p[land]
print(f"  validation rows {len(va):,}, land rows {land.sum():,}")
log("land only: accuracy", accuracy_score(yl, pl >= .5), "0.514")
log("land only: kappa", cohen_kappa_score(yl, (pl >= .5).astype(int)), "0.032")
log("land only: ROC-AUC", roc_auc_score(yl, pl), "0.515")
log("land only: always 'not flooded' accuracy", 1 - yl.mean(), "0.546")
log("land only: share of samples flagged (p >= 0.5)", (pl >= .5).mean(), "52%")
log("land only: precision of flagged samples", precision_score(yl, pl >= .5), "47%")
log("land only: base rate", yl.mean(), "45%")
log("land only: recall", recall_score(yl, pl >= .5), "")

rule("Chronological hold-outs")
yt = te.flood.values
pt = model.predict_proba(te[F].values)[:, 1]
log("test 2023: kappa", cohen_kappa_score(yt, (pt >= .5).astype(int)), "0.389")
log("validation 2024-2026: kappa", cohen_kappa_score(y, (p >= .5).astype(int)), "0.383")


def weights(v):
    c0, c1 = (v == 0).sum(), (v == 1).sum()
    m = max(c0, c1)
    return np.where(v == 0, m / c0, m / c1)


rule("Four classifier families (validation ROC-AUC)")
Xtr, ytr = tr[F].values, tr.flood.values
fams = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, class_weight="balanced")),
    "random forest": RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                            random_state=SEED, n_jobs=-1),
    "gradient boosting (XGBoost, as reported)": None,
    "LightGBM": LGBMClassifier(n_estimators=300, class_weight="balanced",
                               random_state=SEED, verbose=-1),
}
aucs = []
for name, m in fams.items():
    if m is None:
        a = roc_auc_score(y, p)
    else:
        m.fit(Xtr, ytr)
        a = roc_auc_score(y, m.predict_proba(va[F].values)[:, 1])
    aucs.append(a)
    log(f"{name}: ROC-AUC", a, "")
log("spread across families", max(aucs) - min(aucs), "0.03 (0.84-0.87)")

rule("Calibration (isotonic on out-of-fold training predictions)")


def make_xgb():
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                         objective="binary:logistic", random_state=SEED,
                         eval_metric="logloss", verbosity=0,
                         subsample=0.8, colsample_bytree=0.8)


oof = np.zeros(len(ytr))
for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr):
    m = make_xgb()
    m.fit(Xtr[a_], ytr[a_], sample_weight=weights(ytr[a_]))
    oof[b_] = m.predict_proba(Xtr[b_])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(oof, ytr)
final = make_xgb().fit(Xtr, ytr, sample_weight=weights(ytr))
pc = iso.predict(final.predict_proba(va[F].values)[:, 1])
b_iso = brier_score_loss(y, pc)
b_clim = brier_score_loss(y, np.full(len(y), y.mean()))
log("Brier, isotonic", b_iso, "")
log("Brier, climatology", b_clim, "")
log("Brier skill score", 1 - b_iso / b_clim, "0.33")

pd.DataFrame(rows).to_csv(RESULTS / "01_leak_table1.csv", index=False)
print(f"\n  wrote {RESULTS / '01_leak_table1.csv'}")
