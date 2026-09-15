"""Case-level EFTA safety gates and explanation utilities."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    from .config import CONFIG
except ImportError:  # Support execution from the ml directory.
    from config import CONFIG


def _target_probability(model: Any, X: np.ndarray, target_class: int = 1) -> float:
    """Return the probability for the EFTA explanation target class.

    The validated research implementation defines ``positive`` as dataset
    class 1. For this dataset, class 1 is benign; this is intentionally not
    silently changed to malignant merely because class 0 is the clinical
    condition of interest.
    """
    return float(model.predict_proba(np.asarray(X).reshape(1, -1))[0, target_class])


def gate_g1_local_uncertainty(
    model: Any, patient_profile: np.ndarray, confidence_threshold: float
) -> int:
    """Return 1 when the model's highest class probability meets the threshold."""
    probabilities = model.predict_proba(np.asarray(patient_profile).reshape(1, -1))[0]
    return int(float(np.max(probabilities)) >= confidence_threshold)


def top_k_indices(shap_values: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k largest absolute SHAP contributions."""
    values = np.asarray(shap_values, dtype=float).reshape(-1)
    return np.argsort(np.abs(values))[-k:][::-1]


def gate_g2_faithfulness(
    model: Any,
    patient_profile: np.ndarray,
    shap_values: np.ndarray,
    dataset_means: np.ndarray,
    top_k: int,
    drop_threshold: float = 0.20,
    target_class: int = 1,
) -> tuple[int, float]:
    """Mask top-k features and test the class-1 (benign) target probability drop."""
    patient = np.asarray(patient_profile, dtype=float).reshape(-1)
    means = np.asarray(dataset_means, dtype=float).reshape(-1)
    original = _target_probability(model, patient, target_class)
    masked = patient.copy()
    masked[top_k_indices(shap_values, top_k)] = means[top_k_indices(shap_values, top_k)]
    masked_probability = _target_probability(model, masked, target_class)
    drop = original - masked_probability
    return int(drop >= drop_threshold), float(drop)


def default_shap_explainer(
    model: Any,
    background: np.ndarray,
    feature_names: list[str] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the model-specific primary SHAP explainer for class 1."""
    try:
        from .explanations import build_shap_explainer
    except ImportError:  # Support execution from the ml directory.
        from explanations import build_shap_explainer

    return build_shap_explainer(model, background, feature_names=feature_names)


def gate_g3_stability(
    model: Any,
    patient_profile: np.ndarray,
    shap_explainer: Callable[[np.ndarray], np.ndarray],
    top_k: int,
    perturbation_noise: float = CONFIG["perturbation_noise"],
    num_perturbations: int = CONFIG["num_perturbations"],
    stability_threshold: float = CONFIG["stability_threshold"],
    random_state: int = 0,
) -> tuple[int, float]:
    """Perturb a patient profile and compute mean pairwise top-k Jaccard overlap."""
    del model  # The explainer already closes over the fitted model.
    rng = np.random.default_rng(random_state)
    patient = np.asarray(patient_profile, dtype=float).reshape(-1)
    copies = patient + rng.normal(0.0, perturbation_noise, size=(num_perturbations, patient.size))
    shap_matrix = np.asarray(shap_explainer(copies), dtype=float)
    sets = [set(top_k_indices(row, top_k).tolist()) for row in shap_matrix]
    overlaps = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            overlaps.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    jaccard = float(np.mean(overlaps)) if overlaps else 1.0
    return int(jaccard >= stability_threshold), jaccard


def efta_decision(g1: int, g2: int, g3: int) -> str:
    """Apply the non-compensatory conjunction of the first three gates."""
    return "ACCEPT_FOR_REVIEW" if (g1 == 1 and g2 == 1 and g3 == 1) else "ABSTAIN_ESCALATE"
