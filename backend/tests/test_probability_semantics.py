from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
ML = ROOT / "ml"
for path in (BACKEND, ML):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.schemas.prediction import PredictionRequest  # noqa: E402
from app.services.efta_service import run_efta  # noqa: E402
from gates import gate_g1_local_uncertainty, gate_g2_faithfulness  # noqa: E402


class FakeModel:
    """Two-class model where feature 0 controls class-1/benign probability."""

    def predict_proba(self, X):
        rows = np.asarray(X)
        p_benign = 1.0 / (1.0 + np.exp(-2.0 * rows[:, 0]))
        return np.column_stack([1.0 - p_benign, p_benign])


class FakeService:
    model = FakeModel()
    seed = 40
    feature_names = [f"feature_{i}" for i in range(30)]
    class_names = ["malignant", "benign"]

    def predict_proba_scaled(self, features):
        return self.model.predict_proba(np.asarray(features).reshape(1, -1))[0]

    def get_shap_explainer(self):
        def explain(X):
            values = np.zeros((len(X), 30), dtype=float)
            values[:, :3] = 1.0
            return values

        return explain


def test_malignant_prediction_has_consistent_probabilities_and_g1():
    result = run_efta(FakeService(), np.array([-2.0] + [0.0] * 29))
    assert result["prediction"] == 0
    assert result["prediction_label"] == "malignant"
    assert result["malignant_probability"] > result["benign_probability"]
    assert result["probability"] == result["malignant_probability"]
    assert result["probability_class"] == 0
    assert result["confidence"] == result["malignant_probability"]
    assert result["gates"]["G1"]["passed"] is True


def test_benign_prediction_has_consistent_probabilities_and_g1():
    result = run_efta(FakeService(), np.array([2.0] + [0.0] * 29))
    assert result["prediction"] == 1
    assert result["prediction_label"] == "benign"
    assert result["benign_probability"] > result["malignant_probability"]
    assert result["probability"] == result["benign_probability"]
    assert result["probability_class"] == 1
    assert result["confidence"] == result["benign_probability"]
    assert result["gates"]["G1"]["passed"] is True


def test_g2_uses_predicted_class_by_default_and_supports_explicit_target():
    model = FakeModel()
    patient = np.array([2.0] + [0.0] * 29)
    shap_values = np.zeros(30)
    shap_values[:3] = 1.0
    gate, drop = gate_g2_faithfulness(model, patient, shap_values, np.zeros(30), 3, 0.2)
    assert gate == 1
    assert drop >= 0.2

    malignant_patient = np.array([-2.0] + [0.0] * 29)
    malignant_gate, malignant_drop = gate_g2_faithfulness(
        model, malignant_patient, shap_values, np.zeros(30), 3, 0.2
    )
    assert malignant_gate == 1
    assert malignant_drop >= 0.2

    explicit_gate, explicit_drop = gate_g2_faithfulness(
        model, malignant_patient, shap_values, np.zeros(30), 3, 0.2, target_class=1
    )
    assert explicit_gate == 0
    assert explicit_drop < 0.0


def test_request_rejects_non_30_feature_vectors():
    try:
        PredictionRequest(features=[1.0, 2.0, 3.0])
    except Exception as exc:
        assert "30" in str(exc) or "at least 30" in str(exc)
    else:
        raise AssertionError("A non-30-feature request must be rejected.")
