"""UCI Heart Disease (Cleveland subset) loader.

Uses the official UCI Machine Learning Repository source (dataset id=45,
via the `ucimlrepo` package), Cleveland records only. Hungary/Switzerland/
VA Long Beach records are never merged in.

Design constraints from the research protocol:
- explicit, non-silent binary target construction (raw 'num' is 0-4)
- missing values ('ca', 'thal') are left as NaN here; imputation must be
  fit on the TRAINING split only, inside the downstream pipeline, not here
- no global imputation before splitting

Network note: fetching live data requires reaching archive.ics.uci.edu
(via the ucimlrepo package). If that network path is unavailable in your
environment, this loader raises a clear RuntimeError with a manual-download
fallback path documented below, rather than silently returning fabricated
or cached-elsewhere data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_PATH = CACHE_DIR / "heart_disease_cleveland_raw.csv"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
RAW_TARGET_COLUMN = "num"
BINARY_TARGET_COLUMN = "binary_target"

# Manual fallback if `ucimlrepo` cannot reach the network in this environment:
# 1. Download processed.cleveland.data from:
#    https://archive.ics.uci.edu/dataset/45/heart+disease
#    (or the mirrored raw file at the UCI ML repository's data directory)
# 2. Save it, comma-separated, with a header row matching FEATURE_COLUMNS + ["num"],
#    to: ml/data/cache/heart_disease_cleveland_raw.csv
# 3. Missing values in the source file are encoded as "?" — leave them as-is;
#    this loader converts "?" to NaN automatically.


def _fetch_via_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo  # imported lazily; optional dependency

    repo = fetch_ucirepo(id=45)
    X = repo.data.features
    y = repo.data.targets
    df = pd.concat([X, y], axis=1)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _load_raw_dataframe() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH, na_values="?")
    try:
        df = _fetch_via_ucimlrepo()
    except Exception as exc:  # network unavailable, package missing, etc.
        raise RuntimeError(
            "Could not fetch UCI Heart Disease data via ucimlrepo, and no local "
            f"cache exists at {CACHE_PATH}. Install `ucimlrepo` and ensure network "
            "access to archive.ics.uci.edu, OR follow the manual fallback documented "
            "at the top of heart_disease_loader.py to populate the cache file."
        ) from exc
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def load_heart_disease() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load the Cleveland Heart Disease subset with explicit binary target construction.

    Returns
    -------
    X : (n_samples, 13) float array, may contain NaN in 'ca'/'thal' columns —
        impute inside the training pipeline only, never here.
    y : (n_samples,) int array in {0, 1}: 0 = no_disease, 1 = disease_present
        (raw 'num' values 1-4 are all mapped to 1; this is explicit, not silent)
    feature_names : list[str], the 13 standard Cleveland attributes
    class_names : ["no_disease", "disease_present"]
    """
    df = _load_raw_dataframe()

    missing_cols = [c for c in FEATURE_COLUMNS + [RAW_TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"Heart Disease source is missing expected columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[FEATURE_COLUMNS + [RAW_TARGET_COLUMN]].copy()
    raw_target = pd.to_numeric(df[RAW_TARGET_COLUMN], errors="coerce")
    if raw_target.isna().any():
        raise RuntimeError("Raw 'num' target contains non-numeric or missing values.")
    if not set(raw_target.unique()).issubset({0, 1, 2, 3, 4}):
        raise RuntimeError(
            f"Raw 'num' target has unexpected values: {sorted(raw_target.unique())}; "
            "expected only 0-4 per the standard Cleveland formulation."
        )

    # Explicit binary construction: 0 stays 0 (no disease); 1-4 -> 1 (disease present).
    df[BINARY_TARGET_COLUMN] = (raw_target > 0).astype(int)

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = df[BINARY_TARGET_COLUMN].to_numpy(dtype=int)
    class_names = ["no_disease", "disease_present"]

    return X, y, list(FEATURE_COLUMNS), class_names
