"""Distribution-shift foundation tests."""

from __future__ import annotations

import numpy as np
import pytest

from ml.data.unified_interface import load_dataset
from ml.models import requires_feature_scaling
from ml.shifts import (
    age_subgroup,
    measurement_noise,
    missingness,
    permissible_continuous_noise_indices,
    permissible_missingness_indices,
    prevalence_shift,
    shift_ready_splits,
)


def _bundle(dataset_id: str):
    try:
        return load_dataset(dataset_id, seed=0)
    except RuntimeError as exc:
        if dataset_id == "heart_disease":
            pytest.skip(str(exc))
        raise


def test_measurement_noise_is_reproducible_and_uses_permissible_features() -> None:
    bundle = _bundle("heart_disease")
    splits = shift_ready_splits(bundle, seed=0, scale_features=True)
    original_train = bundle.X[splits.train_indices].copy()
    first = measurement_noise(bundle, splits, 0.1, seed=123)
    second = measurement_noise(bundle, splits, 0.1, seed=123)

    np.testing.assert_array_equal(first.X_raw, second.X_raw)
    np.testing.assert_array_equal(first.X_prepared, second.X_prepared)
    np.testing.assert_array_equal(bundle.X[splits.train_indices], original_train)
    assert set(first.changed_feature_indices) == {
        bundle.feature_names.index(name)
        for name in ("trestbps", "chol", "thalach", "oldpeak")
    }
    assert bundle.feature_names.index("age") not in first.changed_feature_indices
    assert bundle.feature_names.index("sex") not in first.changed_feature_indices
    assert first.shift_name == "gaussian_noise_0.1"


def test_measurement_noise_does_not_change_immutable_values() -> None:
    bundle = _bundle("heart_disease")
    splits = shift_ready_splits(bundle, seed=1, scale_features=False)
    shifted = measurement_noise(bundle, splits, 0.2, seed=5)
    age = bundle.feature_names.index("age")
    sex = bundle.feature_names.index("sex")
    np.testing.assert_array_equal(shifted.X_raw[:, age], bundle.X[splits.test_indices, age])
    np.testing.assert_array_equal(shifted.X_raw[:, sex], bundle.X[splits.test_indices, sex])


def test_missingness_excludes_protected_features_and_uses_frozen_imputer() -> None:
    bundle = _bundle("heart_disease")
    splits = shift_ready_splits(bundle, seed=0, scale_features=True)
    shifted = missingness(bundle, splits, 0.2, seed=99)

    assert np.isfinite(shifted.X_prepared).all()
    assert np.isnan(shifted.X_raw).any()
    age = bundle.feature_names.index("age")
    sex = bundle.feature_names.index("sex")
    assert not np.isnan(shifted.X_raw[:, age]).any()
    assert not np.isnan(shifted.X_raw[:, sex]).any()
    assert set(shifted.changed_feature_indices) == set(permissible_missingness_indices(bundle))


def test_missingness_requires_a_frozen_imputer() -> None:
    bundle = _bundle("wdbc")
    from ml.data.split_protocol import get_prepared_splits

    splits = get_prepared_splits(bundle, seed=0, scale_features=False)
    with pytest.raises(ValueError, match="force_imputation=True"):
        missingness(bundle, splits, 0.1, seed=0)


def test_prevalence_shift_is_deterministic_and_preserves_test_class_conditionals() -> None:
    bundle = _bundle("synthetic")
    splits = shift_ready_splits(
        bundle, seed=0, scale_features=requires_feature_scaling("logistic_regression")
    )
    first = prevalence_shift(bundle, splits, 0.25, seed=7)
    second = prevalence_shift(bundle, splits, 0.25, seed=7)

    np.testing.assert_array_equal(first.source_indices, second.source_indices)
    np.testing.assert_array_equal(first.y, second.y)
    assert np.mean(first.y == 1) == pytest.approx(0.25, abs=1 / len(first.y))
    assert set(first.source_indices).issubset(set(splits.test_indices))
    assert first.X_prepared.shape[0] == splits.X_test.shape[0]


def test_age_subgroup_uses_training_median_and_handles_missing_age_schema() -> None:
    heart = _bundle("heart_disease")
    heart_splits = shift_ready_splits(heart, seed=0, scale_features=False)
    subgroup = age_subgroup(heart, heart_splits, min_samples=2)
    assert subgroup is not None
    age_index = heart.feature_names.index("age")
    threshold = np.nanmedian(heart.X[heart_splits.train_indices, age_index])
    assert np.all(subgroup.X_raw[:, age_index] >= threshold)
    assert len(subgroup.source_indices) >= 2

    synthetic = _bundle("synthetic")
    synthetic_splits = shift_ready_splits(synthetic, seed=0, scale_features=False)
    assert age_subgroup(synthetic, synthetic_splits) is None


def test_permissible_noise_sets_are_explicit() -> None:
    heart = _bundle("heart_disease")
    names = [heart.feature_names[i] for i in permissible_continuous_noise_indices(heart)]
    assert names == ["trestbps", "chol", "thalach", "oldpeak"]

    wdbc = _bundle("wdbc")
    assert len(permissible_continuous_noise_indices(wdbc)) == 30
    assert len(permissible_missingness_indices(wdbc)) == 30
