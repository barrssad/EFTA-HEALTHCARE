from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[3] / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from config import CONFIG
from gates import (
    efta_decision,
    gate_g1_local_uncertainty,
    gate_g2_faithfulness,
    gate_g3_stability,
    top_k_indices,
)
from .model_service import ModelService


def run_efta(
    service: ModelService,
    scaled_features: np.ndarray,
    confidence_threshold: float | None = None,
    faithfulness_drop_threshold: float | None = None,
    stability_threshold: float | None = None,
) -> dict[str, Any]:
    """Run prediction plus G1/G2/G3 using the existing gate functions."""
    patient = np.asarray(scaled_features, dtype=float).reshape(-1)
    thresholds = {
        "confidence": confidence_threshold if confidence_threshold is not None else CONFIG["confidence_threshold"],
        "faithfulness": faithfulness_drop_threshold if faithfulness_drop_threshold is not None else CONFIG["faithfulness_drop_threshold"],
        "stability": stability_threshold if stability_threshold is not None else CONFIG["stability_threshold"],
    }
    probabilities = service.predict_proba_scaled(patient)
    malignant_probability = float(probabilities[0])
    benign_probability = float(probabilities[1])
    confidence = float(np.max(probabilities))
    predicted_class = int(np.argmax(probabilities))
    predicted_class_probability = float(probabilities[predicted_class])

    g1 = gate_g1_local_uncertainty(service.model, patient, thresholds["confidence"])
    explainer = service.get_shap_explainer()
    shap_values = np.asarray(explainer(patient.reshape(1, -1)))[0]
    important_features = top_k_indices(shap_values, CONFIG["top_k_features"])
    g2, probability_drop = gate_g2_faithfulness(
        service.model,
        patient,
        shap_values,
        np.zeros_like(patient),
        CONFIG["top_k_features"],
        thresholds["faithfulness"],
        target_class=1,
    )
    g3, jaccard = gate_g3_stability(
        service.model,
        patient,
        explainer,
        CONFIG["top_k_features"],
        CONFIG["perturbation_noise"],
        CONFIG["num_perturbations"],
        thresholds["stability"],
        random_state=service.seed,
    )
    decision = efta_decision(g1, g2, g3)
    names = service.feature_names
    return {
        "prediction": predicted_class,
        "prediction_label": service.class_names[predicted_class],
        "probability": predicted_class_probability,
        "probability_class": predicted_class,
        "probability_class_label": service.class_names[predicted_class],
        "malignant_probability": malignant_probability,
        "benign_probability": benign_probability,
        "confidence": confidence,
        "gates": {
            "G1": {
                "passed": bool(g1),
                "score": confidence,
                "threshold": thresholds["confidence"],
                "reason": "Highest class probability meets the local confidence threshold." if g1 else "Highest class probability is below the local confidence threshold.",
            },
            "G2": {
                "passed": bool(g2),
                "score": probability_drop,
                "threshold": thresholds["faithfulness"],
                "reason": "Masking top-k explained features caused the required class-1 (benign) probability drop." if g2 else "Masking top-k explained features did not cause the required class-1 (benign) probability drop.",
            },
            "G3": {
                "passed": bool(g3),
                "score": jaccard,
                "threshold": thresholds["stability"],
                "reason": "Perturbation explanations meet the Jaccard stability threshold." if g3 else "Perturbation explanations fall below the Jaccard stability threshold.",
            },
        },
        "overall_decision": decision,
        "explanation": {
            "method": "Existing EFTA top-k masking and perturbation stability implementation",
            "faithfulness_target": {"class": 1, "label": service.class_names[1], "definition": "The preserved research methodology evaluates the class-1 probability for G2."},
            "top_features": [
                {"index": int(i), "name": names[int(i)], "shap_value": float(shap_values[int(i)])}
                for i in important_features
            ],
            "note": "Feature values are accepted in raw dataset units; the backend applies the training-only scaler before model and gate execution. Probability is the predicted-class probability; malignant and benign probabilities are returned separately.",
        },
    }
