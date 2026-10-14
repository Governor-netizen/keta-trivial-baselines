#!/usr/bin/env python3
"""Appendix A, Table 3: the 28-feature model against a one-line rule under three label-quality gates.

Land only (permanent water excluded), leave-one-event-out. The one-line rule's
feature, direction and threshold are chosen on the training events of each fold
(keta_evaluation.fit_threshold_rule). Intervals resample events with the held-out
predictions fixed. Adapted from keta_gate_sensitivity.py in the project repository.

    python scripts/05_table3_gates.py            # ~10-20 min
"""
import warnings

import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from common import DATA, RESULTS, rule
from keta_data import prepare_event_samples
from keta_evaluation import event_folds, out_of_event_baselines
from keta_loeo import DYNAMIC, TERRAIN, boot_ci, leave_one_event_out

warnings.filterwarnings("ignore")
GATES = [("all events", None), ("sign-only (<= 0 dB)", 0.0), ("strict (<= -0.25 dB)", -0.25)]
PAPER = {"all events": "+0.035 / +0.027 / 0.541",
         "sign-only (<= 0 dB)": "+0.036 / +0.033 / 0.518",
         "strict (<= -0.25 dB)": "+0.131 / +0.021 / 0.543"}

samples = pd.read_csv(DATA / "keta_samples_v5_1_all_splits.csv.gz")
rows, rules = [], []
rule("Table 3")
for label, thr in GATES:
    data = prepare_event_samples(samples, min_drop_db=thr if thr is not None else -.25,
                                 diagnostic=thr is None)
    feats = [f for f in TERRAIN + DYNAMIC if f in data]
    y = data.flood.astype(int).to_numpy()
    folds = event_folds(data)
    base = out_of_event_baselines(data, feats, y, folds)
    prob = leave_one_event_out(data, feats, y, folds=folds)
    kappa = lambda a, b: cohen_kappa_score(a, (b >= .5).astype(int))
    rk, mk = kappa(y, base["threshold"]), kappa(y, prob)
    rlo, rhi = boot_ci(data, y, base["threshold"], kappa, n=1000)
    mlo, mhi = boot_ci(data, y, prob, kappa, n=1000)
    auc = roc_auc_score(y, prob)
    rows.append(dict(gate=label, events=data.event_id.nunique(), land_rows=len(data),
                     n_features=len(feats), rule_kappa=rk, rule_lo=rlo, rule_hi=rhi,
                     model_kappa=mk, model_lo=mlo, model_hi=mhi, model_auc=auc))
    for r in base["fold_rules"]:
        rules.append(dict(gate=label, **r))
    used = pd.Series([r["feature"] for r in base["fold_rules"]]).value_counts().to_dict()
    print(f"  {label:<22} {data.event_id.nunique():>2} events {len(data):>6,} rows | "
          f"rule {rk:+.3f} [{rlo:+.3f}, {rhi:+.3f}]  model {mk:+.3f} [{mlo:+.3f}, {mhi:+.3f}]  "
          f"AUC {auc:.3f}   [paper: {PAPER[label]}]")
    print(f"  {'':<22} rule features chosen across folds: {used}")

pd.DataFrame(rows).to_csv(RESULTS / "05_table3_gates.csv", index=False)
pd.DataFrame(rules).to_csv(RESULTS / "05_table3_fold_rules.csv", index=False)
print(f"\n  wrote {RESULTS / '05_table3_gates.csv'} and 05_table3_fold_rules.csv")
