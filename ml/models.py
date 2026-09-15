"""Model pipelines, validation calibration, and system-level metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

try:
    from .config import CONFIG
except ImportError:  # Support execution from the ml directory.
    from config import CONFIG

LOGISTIC_REGRESSION_CONFIG = {
    "C": CONFIG["logistic_regression_C"],
    "max_iter": CONFIG["logistic_regression_max_iter"],
    "random_state": CONFIG["model_random_state"],
}
RANDOM_FOREST_CONFIG = {
    "n_estimators": CONFIG["random_forest_trees"],
    "n_jobs": -1,
    "random_state": CONFIG["model_random_state"],
}
RANDOM_FOREST_IMBALANCE_RATIO_THRESHOLD = CONFIG[
    "random_forest_imbalance_ratio_threshold"
]


def requires_feature_scaling(model_name: str) -> bool:
    if model_name == "logistic_regression":
        return True
    if model_name == "random_forest":
        return False
    raise ValueError("model_name must be 'logistic_regression' or 'random_forest'.")


def random_forest_class_weight(y_train: np.ndarray) -> str | None:
    """Use balanced weighting only when the training split is meaningfully imbalanced."""
    labels, counts = np.unique(np.asarray(y_train, dtype=int), return_counts=True)
    if len(labels) < 2 or np.min(counts) == 0:
        raise ValueError("y_train must contain both classes for binary classification.")
    imbalance_ratio = float(np.max(counts) / np.min(counts))
    return "balanced" if imbalance_ratio >= RANDOM_FOREST_IMBALANCE_RATIO_THRESHOLD else None


def build_base_pipeline(
    model_name: str = "logistic_regression", y_train: np.ndarray | None = None
) -> Pipeline:
    """Build a classifier pipeline for already-prepared feature arrays.

    Random Forest class weighting is selected from training labels only. If
    labels are omitted, no weighting is applied so callers cannot accidentally
    tune it using validation or test data.
    """
    if model_name == "logistic_regression":
        classifier = LogisticRegression(**LOGISTIC_REGRESSION_CONFIG)
    elif model_name == "random_forest":
        class_weight = random_forest_class_weight(y_train) if y_train is not None else None
        classifier = RandomForestClassifier(
            **RANDOM_FOREST_CONFIG,
            class_weight=class_weight,
        )
    else:
        raise ValueError("model_name must be 'logistic_regression' or 'random_forest'.")
    return Pipeline([("classifier", classifier)])


def fit_calibrated_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> CalibratedClassifierCV:
    """Fit a base model on train and calibrate its probabilities on validation.

    ``cv='prefit'`` is used intentionally: validation is the calibration set,
    and therefore must not be mixed into base-model fitting.
    """
    base = build_base_pipeline(model_name, y_train=y_train)
    base.fit(X_train, y_train)
    # scikit-learn 1.6 replaced the deprecated ``cv='prefit'`` spelling with
    # FrozenEstimator. Keep a compatibility branch so the same code also runs
    # on older Colab images.
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    except ImportError:
        calibrated = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
    calibrated.fit(X_val, y_val)
    return calibrated


def gate_g0_metrics(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Calculate AUROC, AUPRC, and Brier score for a binary classifier."""
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "auprc": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
    }


def gate_g5_subgroup_check(
    model: Any, X: np.ndarray, y: np.ndarray, feature_index: int = 0
) -> dict[str, Any]:
    """Compare performance for groups split at the first feature's median.

    This is an engineered proxy subgroup, not a real demographic attribute and
    must not be interpreted as a protected-class fairness audit.
    """
    if len(X) == 0:
        raise ValueError("X cannot be empty.")
    threshold = float(np.median(X[:, feature_index]))
    groups = X[:, feature_index] >= threshold
    output: dict[str, Any] = {"proxy_feature_index": feature_index, "threshold": threshold}
    for group_name, mask in (("below_median", ~groups), ("at_or_above_median", groups)):
        if mask.sum() == 0:
            output[group_name] = {"n": 0}
            continue
        probs = model.predict_proba(X[mask])[:, 1]
        predictions = (probs >= 0.5).astype(int)
        output[group_name] = {
            "n": int(mask.sum()),
            "accuracy": float(np.mean(predictions == y[mask])),
            "positive_rate": float(np.mean(predictions)),
        }
    return output
