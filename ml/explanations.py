"""Primary and diagnostic explanation methods for fitted EFTA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


@dataclass
class ShapExplanation:
    """Callable SHAP adapter with stable class-1 and feature-shape semantics."""

    explainer: Any
    explainer_type: str
    feature_names: list[str]
    target_class: int = 1

    def __call__(self, X: np.ndarray) -> np.ndarray:
        values = self.explainer(np.asarray(X, dtype=float))
        raw = values.values if hasattr(values, "values") else values
        if isinstance(raw, list):
            raw = raw[self.target_class]
        raw = np.asarray(raw, dtype=float)
        if raw.ndim == 3:
            raw = raw[:, :, self.target_class]
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        expected_shape = (len(np.asarray(X)), len(self.feature_names))
        if raw.shape != expected_shape:
            raise ValueError(
                f"SHAP output shape mismatch: expected {expected_shape}, got {raw.shape}."
            )
        return raw


def _unwrap_classifier(model: Any) -> Any:
    """Extract the fitted classifier from calibrated and preprocessing wrappers."""
    candidate = model
    calibrated = getattr(candidate, "calibrated_classifiers_", None)
    if calibrated:
        calibrated_model = calibrated[0]
        candidate = getattr(calibrated_model, "estimator", None)
        if candidate is None:
            candidate = getattr(calibrated_model, "base_estimator", None)
    while candidate is not None and hasattr(candidate, "estimator"):
        candidate = candidate.estimator
    if hasattr(candidate, "named_steps"):
        candidate = candidate.named_steps["classifier"]
    return candidate


def _load_shap() -> Any:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is required for the primary explanation methods. "
            "Install the project requirements before requesting explanations."
        ) from exc
    return shap


def build_shap_explainer(
    model: Any,
    background: np.ndarray,
    feature_names: list[str] | None = None,
) -> ShapExplanation:
    """Build the primary SHAP explainer for the fitted model family.

    The returned values explain class 1, matching the existing G2 methodology.
    Inputs must already be transformed by the dataset preparation pipeline.
    """
    shap = _load_shap()
    background_array = np.asarray(background, dtype=float)
    if background_array.ndim != 2 or len(background_array) == 0:
        raise ValueError("background must be a non-empty 2-D feature array.")

    classifier = _unwrap_classifier(model)
    names = feature_names or [f"feature_{i}" for i in range(background_array.shape[1])]
    if len(names) != background_array.shape[1]:
        raise ValueError("feature_names must align with the background feature count.")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if isinstance(classifier, LogisticRegression):
        explainer = shap.LinearExplainer(classifier, background_array)
        explainer_type = "LinearExplainer"
    elif isinstance(classifier, RandomForestClassifier):
        explainer = shap.TreeExplainer(
            classifier, data=background_array, feature_perturbation="interventional"
        )
        explainer_type = "TreeExplainer"
    else:
        raise TypeError(
            f"Unsupported classifier for primary SHAP explanations: {type(classifier).__name__}."
        )

    return ShapExplanation(explainer, explainer_type, list(names))


def global_shap_summary(
    explainer: ShapExplanation, X: np.ndarray
) -> pd.DataFrame:
    """Return mean absolute class-1 SHAP importance for every feature."""
    values = explainer(X)
    return pd.DataFrame(
        {
            "feature": explainer.feature_names,
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def permutation_importance_summary(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    random_state: int = 0,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Compute global permutation importance as a diagnostic sanity check only."""
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=int)
    if X_array.ndim != 2 or X_array.shape[1] != len(feature_names):
        raise ValueError("X columns must align with feature_names.")
    result = permutation_importance(
        model,
        X_array,
        y_array,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
