"""Dataset-agnostic 60/20/20 stratified split + training-only preprocessing.

Generalizes the existing data_loader.get_train_val_test_splits() to also
handle datasets with missing values (heart_disease) via a SimpleImputer
fit strictly on the training split, chained before the StandardScaler.
For WDBC and synthetic (no missing values), behavior is identical to the
original function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .unified_interface import DatasetBundle

MIN_CLASS_COUNT_PER_SPLIT = 2  # below this, a split is statistically unusable for that class


@dataclass
class PreparedSplits:
    dataset_id: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    preprocessing_pipeline: Pipeline
    split_warnings: list[str]


def _build_preprocessing_pipeline(has_missing_values: bool) -> Pipeline:
    steps = []
    if has_missing_values:
        # Median imputation for numeric clinical features; fit on training only
        # because this Pipeline is fit exclusively on X_train below.
        steps.append(("imputer", SimpleImputer(strategy="median")))
    steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _check_class_counts(y_split: np.ndarray, split_name: str, class_names: list[str]) -> list[str]:
    warnings = []
    counts = np.bincount(y_split, minlength=len(class_names))
    for class_idx, count in enumerate(counts):
        if count < MIN_CLASS_COUNT_PER_SPLIT:
            warnings.append(
                f"{split_name} split has only {count} sample(s) of class "
                f"'{class_names[class_idx] if class_idx < len(class_names) else class_idx}' "
                f"(< {MIN_CLASS_COUNT_PER_SPLIT}); metrics on this split/class may be unstable."
            )
    return warnings


def get_prepared_splits(bundle: DatasetBundle, seed: int) -> PreparedSplits:
    """Create 60/20/20 stratified splits and fit preprocessing on training only.

    Raises no exception on small-class warnings (by design — the caller decides
    whether to proceed); warnings are collected in `split_warnings` so nothing
    is silently swallowed, per the "do not change the protocol silently" rule.
    """
    X, y = np.asarray(bundle.X, dtype=float), np.asarray(bundle.y, dtype=int)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2-D, y must be 1-D, and X/y lengths must match.")

    all_indices = np.arange(len(X))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        all_indices, y, test_size=0.40, stratify=y, random_state=seed
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]

    train_set, val_set, test_set = set(idx_train), set(idx_val), set(idx_test)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Split index overlap detected — this must never happen.")
    if len(train_set | val_set | test_set) != len(all_indices):
        raise RuntimeError("Splits do not partition the full dataset — rows lost or duplicated.")

    pipeline = _build_preprocessing_pipeline(bundle.has_missing_values)
    X_train_prepared = pipeline.fit_transform(X_train, y_train)
    X_val_prepared = pipeline.transform(X_val)
    X_test_prepared = pipeline.transform(X_test)

    warnings: list[str] = []
    for split_name, y_split in (("train", y_train), ("validation", y_val), ("test", y_test)):
        warnings.extend(_check_class_counts(y_split, split_name, bundle.class_names))

    return PreparedSplits(
        dataset_id=bundle.dataset_id,
        X_train=X_train_prepared,
        X_val=X_val_prepared,
        X_test=X_test_prepared,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_indices=idx_train,
        val_indices=idx_val,
        test_indices=idx_test,
        feature_names=bundle.feature_names,
        class_names=bundle.class_names,
        preprocessing_pipeline=pipeline,
        split_warnings=warnings,
    )
