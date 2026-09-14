"""Dataset loading, stratified splitting, and leakage-safe preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSplits:
    """Arrays and metadata used by the experiment pipeline."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    scaler: StandardScaler


def load_wisconsin_dataset() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load the native scikit-learn Wisconsin Diagnostic Breast Cancer dataset."""
    dataset = load_breast_cancer()
    return (
        np.asarray(dataset.data, dtype=float),
        np.asarray(dataset.target, dtype=int),
        [str(name) for name in dataset.feature_names],
        [str(name) for name in dataset.target_names],
    )


def get_train_val_test_splits(
    X: np.ndarray, y: np.ndarray, seed: int
) -> DatasetSplits:
    """Create 60/20/20 stratified splits and fit StandardScaler on training only."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2-D, y must be 1-D, and X/y lengths must match.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    feature_names = load_breast_cancer().feature_names.tolist()
    target_names = load_breast_cancer().target_names.tolist()
    return DatasetSplits(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        feature_names,
        target_names,
        scaler,
    )


def load_and_split(seed: int) -> DatasetSplits:
    """Convenience function used by experiments.py."""
    X, y, _, _ = load_wisconsin_dataset()
    return get_train_val_test_splits(X, y, seed)
