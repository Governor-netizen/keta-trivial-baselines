"""Event-held-out XGBoost and event bootstrap, extracted unchanged from keta_model_v5.py."""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from keta_evaluation import event_folds

SEED = 42

TERRAIN = ["elevation", "slope", "aspect", "dist_water", "lt3", "lt1", "lt0"]
DYNAMIC = ["imerg_24h", "imerg_peak_intensity", "imerg_3d", "chirps_7d",
           "chirps_30d", "solar_rad", "u_wind_10m", "v_wind_10m", "wind_speed",
           "wind_dir", "surface_pressure", "onshore_wind", "soil_moisture",
           "tide_max_fes", "tide_range_fes", "spring_flag_fes", "tide_anomaly",
           "freeboard_fes", "compound_risk", "season_wet", "rain_intensity_ratio"]


def model(**kw):
    p = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=3.0,
             eval_metric="logloss", random_state=SEED, verbosity=0)
    p.update(kw)
    return XGBClassifier(**p)


def leave_one_event_out(df, feats, y, extra=None, folds=None):
    """Out-of-fold probabilities, holding out one whole event at a time."""
    prob = np.zeros(len(y))
    folds = event_folds(df) if folds is None else folds
    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not feats:
        raise ValueError("No usable predictors were supplied.")
    for e, train_idx, test_idx in folds:
        tr, te = np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)
        tr[train_idx], te[test_idx] = True, True
        if len(np.unique(y[tr])) < 2:
            prob[te] = y[tr].mean()
            continue
        fold_X = X.copy()
        if extra is not None:
            fold_X = fold_X.assign(**{k: v for k, v in extra(df, tr).items()})
        m = model().fit(fold_X.loc[tr], y[tr])
        prob[te] = m.predict_proba(fold_X.loc[te])[:, 1]
    return prob


def boot_ci(df, y, prob, stat, n=2000, seed=SEED):
    """Bootstrap over EVENTS. Pixels within an event are not independent."""
    rng = np.random.default_rng(seed)
    ev = df.event_id.values
    uniq = np.unique(ev)
    idx = {e: np.where(ev == e)[0] for e in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[e] for e in pick])
        if len(np.unique(y[sel])) < 2:
            continue
        try:
            out.append(stat(y[sel], prob[sel]))
        except ValueError:
            continue
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan, np.nan)
