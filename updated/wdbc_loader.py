"""WDBC loader — thin, verified wrapper around sklearn's packaged dataset.

Preserves the existing project's semantics exactly: class 0 = malignant,
class 1 = benign, 30 features, no ID field. This does not replace
data_loader.load_wisconsin_dataset(); it wraps the same call behind the
unified interface and asserts the declared shape/semantics hold at runtime.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer


def load_wdbc() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load WDBC and verify it matches the documented registry metadata.

    Raises
    ------
    RuntimeError if the loaded data does not match the documented shape or
    class semantics (569x30, class 0 = malignant, class 1 = benign). This
    is a fail-loud check, not a silent adjustment.
    """
    dataset = load_breast_cancer()
    X = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target, dtype=int)
    feature_names = [str(name) for name in dataset.feature_names]
    class_names = [str(name) for name in dataset.target_names]

    if X.shape != (569, 30):
        raise RuntimeError(
            f"WDBC shape mismatch: expected (569, 30), got {X.shape}. "
            "Registry metadata assumes the standard sklearn-packaged version."
        )
    if class_names != ["malignant", "benign"]:
        raise RuntimeError(
            f"WDBC class order changed: expected ['malignant', 'benign'], got {class_names}. "
            "Downstream code assumes class 0 = malignant, class 1 = benign."
        )
    if "id" in [f.lower() for f in feature_names]:
        raise RuntimeError("Unexpected ID-like feature found in WDBC feature names.")

    return X, y, feature_names, class_names
