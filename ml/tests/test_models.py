"""Part 1 model and preprocessing tests."""

from __future__ import annotations

import numpy as np
import pytest

from ml.data.split_protocol import get_prepared_splits
from ml.data.unified_interface import VALID_DATASET_IDS, load_dataset
from ml.models import (
    build_base_pipeline,
    fit_calibrated_model,
    random_forest_class_weight,
    requires_feature_scaling,
)


def _load_bundle(dataset_id: str):
    try:
        return load_dataset(dataset_id, seed=0)
    except RuntimeError as exc:
        if dataset_id == "heart_disease":
            pytest.skip(str(exc))
        raise


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest"])
@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
def test_both_models_fit_all_datasets(model_name: str, dataset_id: str) -> None:
    bundle = _load_bundle(dataset_id)
    splits = get_prepared_splits(
        bundle, seed=0, scale_features=requires_feature_scaling(model_name)
    )
    model = fit_calibrated_model(
        model_name, splits.X_train, splits.y_train, splits.X_val, splits.y_val
    )

    probabilities = model.predict_proba(splits.X_test)
    assert probabilities.shape == (len(splits.y_test), 2)
    assert np.isfinite(probabilities).all()
    assert splits.feature_names == bundle.feature_names
    assert len(splits.feature_names) == splits.X_train.shape[1]


@pytest.mark.parametrize(
    ("model_name", "expected_steps"),
    [
        ("logistic_regression", ["scaler"]),
        ("random_forest", ["identity"]),
    ],
)
def test_model_preprocessing_has_no_double_scaling(
    model_name: str, expected_steps: list[str]
) -> None:
    bundle = load_dataset("synthetic", seed=0)
    splits = get_prepared_splits(
        bundle, seed=0, scale_features=requires_feature_scaling(model_name)
    )
    model = build_base_pipeline(model_name, y_train=splits.y_train)

    assert list(splits.preprocessing_pipeline.named_steps) == expected_steps
    assert "scaler" not in model.named_steps
    assert list(model.named_steps) == ["classifier"]


def test_preprocessing_statistics_are_training_only() -> None:
    bundle = _load_bundle("heart_disease")
    splits = get_prepared_splits(bundle, seed=0, scale_features=True)
    imputer = splits.preprocessing_pipeline.named_steps["imputer"]
    scaler = splits.preprocessing_pipeline.named_steps["scaler"]

    raw_train = bundle.X[splits.train_indices]
    np.testing.assert_allclose(imputer.statistics_, np.nanmedian(raw_train, axis=0))
    assert scaler.n_samples_seen_ == len(splits.y_train)

    original_mean = scaler.mean_.copy()
    altered_test = bundle.X[splits.test_indices].copy()
    altered_test[:] = 1_000_000.0
    splits.preprocessing_pipeline.transform(altered_test)
    np.testing.assert_array_equal(scaler.mean_, original_mean)


def test_random_forest_weighting_uses_training_distribution_only() -> None:
    assert random_forest_class_weight(np.array([0, 0, 0, 1])) == "balanced"
    assert random_forest_class_weight(np.array([0, 0, 1, 1])) is None

    bundle = load_dataset("wdbc")
    splits = get_prepared_splits(bundle, seed=0, scale_features=False)
    pipeline = build_base_pipeline("random_forest", y_train=splits.y_train)
    classifier = pipeline.named_steps["classifier"]
    expected_weight = random_forest_class_weight(splits.y_train)
    assert classifier.class_weight == expected_weight


def test_model_reproducibility() -> None:
    bundle = load_dataset("wdbc")
    first_splits = get_prepared_splits(bundle, seed=40, scale_features=False)
    second_splits = get_prepared_splits(bundle, seed=40, scale_features=False)
    first_model = fit_calibrated_model(
        "random_forest",
        first_splits.X_train,
        first_splits.y_train,
        first_splits.X_val,
        first_splits.y_val,
    )
    second_model = fit_calibrated_model(
        "random_forest",
        second_splits.X_train,
        second_splits.y_train,
        second_splits.X_val,
        second_splits.y_val,
    )

    np.testing.assert_array_equal(first_splits.train_indices, second_splits.train_indices)
    np.testing.assert_allclose(
        first_model.predict_proba(first_splits.X_test),
        second_model.predict_proba(second_splits.X_test),
    )
