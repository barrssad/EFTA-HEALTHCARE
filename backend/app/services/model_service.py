from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_ROOT = PROJECT_ROOT / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from data_loader import load_and_split, load_wisconsin_dataset  # noqa: E402
from models import fit_calibrated_model  # noqa: E402


class ModelService:
    """Own one fitted EFTA model and the raw-to-model preprocessing path."""

    def __init__(self, model_name: str = "logistic_regression", seed: int = 40) -> None:
        if model_name not in {"logistic_regression", "random_forest"}:
            raise ValueError("Unsupported model. Use logistic_regression or random_forest.")
        self.model_name = model_name
        self.seed = seed
        self.X_raw, self.y_raw, self.feature_names, self.class_names = load_wisconsin_dataset()
        self.splits = load_and_split(seed)
        self.model = fit_calibrated_model(
            model_name,
            self.splits.X_train,
            self.splits.y_train,
            self.splits.X_val,
            self.splits.y_val,
        )
        self.shap_explainer = None

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def transform_raw(self, features: list[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float).reshape(1, -1)
        if values.shape[1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} features, received {values.shape[1]}.")
        return self.splits.scaler.transform(values)[0]

    def predict_proba_scaled(self, scaled_features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(scaled_features).reshape(1, -1))[0]

    def get_shap_explainer(self) -> Any:
        if self.shap_explainer is None:
            from gates import default_shap_explainer

            self.shap_explainer = default_shap_explainer(self.model, self.splits.X_train[:80])
        return self.shap_explainer

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset": "scikit-learn Wisconsin Diagnostic Breast Cancer",
            "feature_count": self.feature_count,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "training_seed": self.seed,
            "preprocessing": "StandardScaler fitted on the training split only, then applied to raw inference input.",
        }


_model_services: dict[str, ModelService] = {}


def get_model_service(model_name: str = "logistic_regression") -> ModelService:
    if model_name not in _model_services:
        _model_services[model_name] = ModelService(model_name=model_name)
    return _model_services[model_name]
