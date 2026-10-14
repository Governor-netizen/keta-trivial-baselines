"""Training-only baselines and shared event folds for retrospective evaluation."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def event_folds(df):
    if "event_id" not in df or df.event_id.isna().any():
        raise ValueError("Nonmissing event_id is required for grouped evaluation.")
    events = df.event_id.astype(str).to_numpy()
    unique = np.unique(events)
    if len(unique) < 2:
        raise ValueError("Leave-one-event-out evaluation requires at least two events.")
    return [(event, np.flatnonzero(events != event), np.flatnonzero(events == event))
            for event in unique]


def validate_event_folds(df, folds):
    """Reject caller-supplied partitions that expose a held-out event to training."""
    expected = {e: (tr, te) for e, tr, te in event_folds(df)}
    if len(folds) != len(expected) or len({str(e) for e, _, _ in folds}) != len(expected):
        raise ValueError("Each event must be held out exactly once.")
    for event, train, test in folds:
        if str(event) not in expected:
            raise ValueError("Unknown held-out event.")
        correct_train, correct_test = expected[str(event)]
        if not np.array_equal(np.sort(train), correct_train) or not np.array_equal(np.sort(test), correct_test):
            raise ValueError("Folds must hold out every row of one event with no train/test overlap.")
    return folds


def _binary_labels(y, n):
    values = np.asarray(y)
    if values.ndim != 1 or len(values) != n or not np.isin(values, [0, 1]).all():
        raise ValueError("Labels must be a length-matched binary vector.")
    return values.astype(int)


def _kappa(y, pred):
    observed = np.mean(y == pred)
    expected = y.mean() * pred.mean() + (1 - y.mean()) * (1 - pred.mean())
    return float((observed - expected) / (1 - expected)) if expected < 1 else 0.0


@dataclass(frozen=True)
class ThresholdRule:
    feature: str | None
    direction: str
    threshold: float | None
    fill_value: float | None
    constant: int
    training_kappa: float


def fit_threshold_rule(train, features, y):
    """Choose feature, direction, threshold and missing-value median using train only.

    Candidate quantiles are fixed in advance. The returned training kappa is a
    fit diagnostic only; the reported comparison must use held-out predictions.
    """
    y = _binary_labels(y, len(train))
    if not len(y):
        raise ValueError("A threshold rule requires nonempty training data.")
    majority = int(y.mean() >= 0.5)
    best = ThresholdRule(None, "constant", None, None, majority, 0.0)
    for feature in features:
        if feature not in train:
            raise ValueError(f"Missing predictor: {feature}")
        raw = pd.to_numeric(train[feature], errors="coerce").to_numpy(dtype=float)
        finite = raw[np.isfinite(raw)]
        if len(finite) == 0 or np.unique(finite).size < 2:
            continue
        median = float(np.median(finite))
        values = np.where(np.isfinite(raw), raw, median)
        for threshold in np.unique(np.percentile(finite, np.arange(5, 100, 5))):
            for direction in ("<=", ">"):
                pred = (values <= threshold if direction == "<=" else values > threshold).astype(int)
                score = _kappa(y, pred)
                if score > best.training_kappa:
                    best = ThresholdRule(feature, direction, float(threshold), median,
                                         majority, score)
    return best


def predict_threshold_rule(rule, data):
    if rule.feature is None:
        return np.full(len(data), rule.constant, dtype=float)
    raw = pd.to_numeric(data[rule.feature], errors="coerce").to_numpy(dtype=float)
    values = np.where(np.isfinite(raw), raw, rule.fill_value)
    return (values <= rule.threshold if rule.direction == "<="
            else values > rule.threshold).astype(float)


def out_of_event_baselines(df, features, y, folds=None):
    """All labels used for fitting are sliced by train indices before selection."""
    y = _binary_labels(y, len(df))
    folds = event_folds(df) if folds is None else validate_event_folds(df, folds)
    majority, prevalence, threshold = (np.full(len(df), np.nan) for _ in range(3))
    records = []
    for event, train_idx, test_idx in folds:
        train_y = y[train_idx]
        majority[test_idx] = float(train_y.mean() >= .5)
        prevalence[test_idx] = train_y.mean()
        rule = fit_threshold_rule(df.iloc[train_idx], features, train_y)
        threshold[test_idx] = predict_threshold_rule(rule, df.iloc[test_idx])
        records.append({"held_out_event": event, "train_n": len(train_idx),
                        "test_n": len(test_idx), **asdict(rule)})
    return {"majority": majority, "prevalence": prevalence,
            "threshold": threshold, "fold_rules": records}
