"""Validated inputs for retrospective Keta event-label experiments.

Passing these checks establishes usable export structure, not independent flood
truth, causal attribution, spatial transfer skill, or operational forecast skill.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
QUALITY_COLUMNS = ("qa_s1_ok", "qa_label_ok", "qa_otsu_threshold_db")


def find_file(filename, base=BASE):
    """Prefer the supported data directory over historical root copies."""
    base = Path(base)
    for directory in (base / "data", base, base / "models"):
        path = directory / filename
        if path.is_file():
            return str(path.resolve())
    return None


def resolve_samples(explicit=None, *, driver=False, base=BASE):
    """Select v5.1 then v5.0; never fall back to the withdrawn v4 exports."""
    if explicit is not None:
        candidate = Path(explicit)
        if re.search(r"_v[34](?:_|\.)", candidate.name, flags=re.IGNORECASE):
            raise ValueError("Withdrawn v3/v4 samples are not accepted for model evaluation.")
        if candidate.is_absolute():
            found = candidate if candidate.is_file() else None
        else:
            found = find_file(candidate, base=base)
        if found is None:
            raise FileNotFoundError(f"Samples file does not exist: {explicit}")
        return str(Path(found).resolve())
    stem = "keta_samples_driver" if driver else "keta_samples"
    for version in ("v5_1", "v5_0"):
        found = find_file(f"{stem}_{version}_all_splits.csv", base=base)
        if found:
            return found
    raise FileNotFoundError(
        f"No corrected {stem} v5.1/v5.0 samples found in {Path(base) / 'data'} "
        "or the project root. Export corrected samples first; legacy fallback is disabled."
    )


def coordinate_columns(df):
    for pair in (("longitude", "latitude"), ("lon", "lat")):
        if set(pair).issubset(df.columns):
            return list(pair)
    raise ValueError("Samples require longitude/latitude or lon/lat coordinates.")


def location_keys(df):
    """Use exported coordinates, never a possibly colliding terrain signature."""
    return df[coordinate_columns(df)].round(7).astype(str).agg("|".join, axis=1)


def prepare_event_samples(df, *, min_drop_db=-0.25, diagnostic=False):
    """Validate the schema and keep land with usable SAR labels.

    The default Otsu threshold range is [-25, -0.25] dB. Changing that range is
    a label sensitivity experiment. It does not validate the flood labels. The
    diagnostic switch bypasses only the physical threshold filter; failed SAR
    coverage and labels, missing flags, and invalid land masks are never kept.
    """
    required = {"event_id", "is_water", "flood", *QUALITY_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Invalid event samples schema; missing: " + ", ".join(missing))
    if not np.isfinite(min_drop_db) or not -25 <= min_drop_db <= 0:
        raise ValueError("min_drop_db must be finite and between -25 and 0 dB.")
    out = df.copy()
    if out.event_id.isna().any() or out.event_id.astype(str).str.strip().eq("").any():
        raise ValueError("Every sample must have a nonempty event_id.")
    out["event_id"] = out.event_id.astype(str)
    xy = coordinate_columns(out)
    numeric = ["is_water", "flood", *QUALITY_COLUMNS, *xy]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if not np.isfinite(out[col]).all():
            raise ValueError(f"{col} contains missing or nonfinite values; repair the export.")
    for col in ("is_water", "flood", "qa_s1_ok", "qa_label_ok"):
        if not out[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only explicit binary 0/1 values.")
    if not out[xy[0]].between(-180, 180).all() or not out[xy[1]].between(-90, 90).all():
        raise ValueError("Exported coordinates are outside geographic longitude/latitude bounds.")
    for col in QUALITY_COLUMNS:
        if out.groupby("event_id")[col].nunique().gt(1).any():
            raise ValueError(f"{col} must be constant within an event; export versions may be mixed.")
    out["_location_key"] = location_keys(out)
    key = ["event_id", "_location_key"]
    for col in ("flood", "is_water", "flood_driver", "split"):
        if col in out and out.groupby(key)[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting {col} values at the same event and coordinates.")
    # Exact repeated observations should not increase the apparent sample size.
    duplicates = int(out.duplicated(key).sum())
    out = out.drop_duplicates(key).drop(columns="_location_key")
    valid = (out.is_water == 0) & (out.qa_s1_ok == 1) & (out.qa_label_ok == 1)
    if not diagnostic:
        valid &= out.qa_otsu_threshold_db.between(-25, min_drop_db)
    out = out.loc[valid].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("No usable land samples remain after the label-quality gate.")
    if out.event_id.nunique() < 2:
        raise ValueError("Evaluation needs at least two independent event groups.")
    if out.flood.nunique() < 2:
        raise ValueError("Retained SAR labels have only one class; discrimination cannot be evaluated.")
    out.attrs["quality_scope"] = (
        "diagnostic SAR labels; physical threshold gate bypassed" if diagnostic
        else f"SAR structural QA and Otsu threshold [-25, {min_drop_db}] dB"
    )
    out.attrs["duplicates_removed"] = duplicates
    return out
