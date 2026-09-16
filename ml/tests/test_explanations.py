"""Part 2 explanation-method tests."""

from __future__ import annotations

import numpy as np
import pytest

from ml.data.split_protocol import get_prepared_splits
from ml.data.unified_interface import VALID_DATASET_IDS, load_dataset
from ml.explanations import (
    SHAP_COMPATIBILITY_REPAIR_VERSION,
    ShapExplanation,
    build_shap_explainer,
    global_shap_summary,
    permutation_importance_summary,
)
from ml.models import fit_calibrated_model, requires_feature_scaling


EXPECTED_CLASSES = {
    "synthetic": ["negative_class", "positive_class"],
    "heart_disease": ["no_disease", "disease_present"],
    "wdbc": ["malignant", "benign"],
}


def test_tree_explainer_compatibility_repair_disables_only_additivity_check() -> None:
    calls = []

    class FakeTreeExplainer:
        def __call__(self, X, **kwargs):
            calls.append(kwargs)
            return np.zeros((len(X), 2))

    adapter = ShapExplanation(
        FakeTreeExplainer(), "TreeExplainer", ["feature_0", "feature_1"]
    )
    values = adapter(np.ones((1, 2)))

    assert SHAP_COMPATIBILITY_REPAIR_VERSION == "tree_additivity_check_v1"
    assert calls == [{"check_additivity": False}]
    assert values.shape == (1, 2)


def _load_bundle(dataset_id: str):
    try:
        return load_dataset(dataset_id, seed=0)
    except RuntimeError as exc:
        if dataset_id == "heart_disease":
            pytest.skip(str(exc))
        raise


def _fit(dataset_id: str, model_name: str):
    bundle = _load_bundle(dataset_id)
    splits = get_prepared_splits(
        bundle, seed=0, scale_features=requires_feature_scaling(model_name)
    )
    model = fit_calibrated_model(
        model_name, splits.X_train, splits.y_train, splits.X_val, splits.y_val
    )
    explainer = build_shap_explainer(
        model, splits.X_train[:32], feature_names=splits.feature_names
    )
    return bundle, splits, model, explainer


@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
@pytest.mark.parametrize(
    ("model_name", "expected_explainer"),
    [("logistic_regression", "LinearExplainer"), ("random_forest", "TreeExplainer")],
)
def test_correct_shap_explainer_and_dimensions(
    dataset_id: str, model_name: str, expected_explainer: str
) -> None:
    bundle, splits, _, explainer = _fit(dataset_id, model_name)
    values = explainer(splits.X_test[:3])

    assert explainer.explainer_type == expected_explainer
    assert type(explainer.explainer).__name__ == expected_explainer
    assert values.shape == (3, bundle.X.shape[1])
    assert len(explainer.feature_names) == bundle.X.shape[1]
    assert explainer.feature_names == bundle.feature_names
    assert np.isfinite(values).all()


@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
def test_global_shap_summary_aligns_features(dataset_id: str) -> None:
    bundle, splits, _, explainer = _fit(dataset_id, "logistic_regression")
    summary = global_shap_summary(explainer, splits.X_test[:8])

    assert list(summary.columns) == ["feature", "mean_abs_shap"]
    assert summary["feature"].tolist() == sorted(bundle.feature_names) or set(
        summary["feature"]
    ) == set(bundle.feature_names)
    assert len(summary) == bundle.X.shape[1]
    assert np.isfinite(summary["mean_abs_shap"]).all()


@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
def test_permutation_importance_is_global_and_feature_aligned(dataset_id: str) -> None:
    bundle, splits, model, _ = _fit(dataset_id, "random_forest")
    summary = permutation_importance_summary(
        model,
        splits.X_test,
        splits.y_test,
        splits.feature_names,
        n_repeats=2,
    )

    assert list(summary.columns) == ["feature", "importance_mean", "importance_std"]
    assert set(summary["feature"]) == set(bundle.feature_names)
    assert len(summary) == bundle.X.shape[1]
    assert np.isfinite(summary[["importance_mean", "importance_std"]].to_numpy()).all()


@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
def test_probability_semantics_and_confidence(dataset_id: str) -> None:
    bundle, splits, model, _ = _fit(dataset_id, "logistic_regression")
    probabilities = model.predict_proba(splits.X_test[:4])
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    assert bundle.class_names == EXPECTED_CLASSES[dataset_id]
    assert np.allclose(confidence, probabilities[np.arange(len(predicted)), predicted])
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
