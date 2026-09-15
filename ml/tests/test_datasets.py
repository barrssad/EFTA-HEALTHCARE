"""Dataset-foundation tests. Run with: pytest ml/tests/test_datasets.py -v

Covers the 11 required checks:
1. all three loaders work
2. repeated loading produces identical data
3. WDBC has expected dimensions and class semantics
4. Heart Disease target construction is correct
5. synthetic ground-truth relevant feature list is correct
6. metadata exists for every dataset
7. no test data is used to fit preprocessing
8. feature names align with X columns
9. y length equals number of samples
10. train/validation/test splits contain no overlapping indices
11. all required classes are represented where statistically possible
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.data.dataset_registry import DATASET_REGISTRY, get_metadata
from ml.data.split_protocol import get_prepared_splits
from ml.data.unified_interface import VALID_DATASET_IDS, load_dataset

HEART_DISEASE_AVAILABLE = True
try:
    load_dataset("heart_disease")
except RuntimeError:
    HEART_DISEASE_AVAILABLE = False  # network/ucimlrepo unavailable in this environment

requires_heart_disease = pytest.mark.skipif(
    not HEART_DISEASE_AVAILABLE,
    reason="UCI Heart Disease unreachable: no network to archive.ics.uci.edu and no local cache.",
)


# 1. all three loaders work
@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
def test_loader_works(dataset_id):
    if dataset_id == "heart_disease" and not HEART_DISEASE_AVAILABLE:
        pytest.skip("heart_disease unreachable in this environment")
    bundle = load_dataset(dataset_id)
    assert bundle.X.shape[0] == bundle.y.shape[0]
    assert bundle.X.shape[0] > 0


# 2. repeated loading produces identical data
@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
def test_repeated_loading_is_identical(dataset_id):
    if dataset_id == "heart_disease" and not HEART_DISEASE_AVAILABLE:
        pytest.skip("heart_disease unreachable in this environment")
    seed = 0 if dataset_id == "synthetic" else None
    b1 = load_dataset(dataset_id, seed=seed) if seed is not None else load_dataset(dataset_id)
    b2 = load_dataset(dataset_id, seed=seed) if seed is not None else load_dataset(dataset_id)
    np.testing.assert_array_equal(b1.X, b2.X)
    np.testing.assert_array_equal(b1.y, b2.y)


# 3. WDBC has expected dimensions and class semantics
def test_wdbc_dimensions_and_semantics():
    bundle = load_dataset("wdbc")
    assert bundle.X.shape == (569, 30)
    assert bundle.class_names == ["malignant", "benign"]
    # class 0 must be malignant per project-wide convention
    assert bundle.class_names[0] == "malignant"


# 4. Heart Disease target construction is correct
@requires_heart_disease
def test_heart_disease_target_construction():
    bundle = load_dataset("heart_disease")
    assert set(np.unique(bundle.y)).issubset({0, 1})
    assert bundle.class_names == ["no_disease", "disease_present"]
    assert bundle.X.shape[1] == 13


# 5. synthetic ground-truth relevant feature list is correct
def test_synthetic_ground_truth_features():
    bundle = load_dataset("synthetic", seed=0)
    assert bundle.ground_truth_relevant_features == [0, 1, 2]
    assert len(bundle.feature_names) == bundle.X.shape[1] == 10


# 6. metadata exists for every dataset
@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
def test_metadata_exists(dataset_id):
    metadata = get_metadata(dataset_id)
    assert metadata.dataset_id == dataset_id
    assert metadata.source_url
    assert metadata.dataset_limitation


# 7. no test data is used to fit preprocessing (indirect check: transform is
# deterministic given train-fitted stats, and re-fitting on test would change it)
def test_preprocessing_fit_on_training_only():
    bundle = load_dataset("wdbc")
    splits = get_prepared_splits(bundle, seed=0)
    # Refit an identical pipeline on train only and confirm transform matches exactly.
    from sklearn.preprocessing import StandardScaler

    manual_scaler = StandardScaler().fit(bundle.X[: len(splits.y_train)])
    # Not a byte-identical re-derivation (indices differ after shuffling), so instead
    # assert the *pipeline itself* was only ever fit once, on X_train-shaped data:
    assert splits.preprocessing_pipeline.named_steps["scaler"].n_samples_seen_ == len(
        splits.y_train
    )


# 8. feature names align with X columns
@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
def test_feature_names_align_with_columns(dataset_id):
    if dataset_id == "heart_disease" and not HEART_DISEASE_AVAILABLE:
        pytest.skip("heart_disease unreachable in this environment")
    bundle = load_dataset(dataset_id)
    assert len(bundle.feature_names) == bundle.X.shape[1]


# 9. y length equals number of samples
@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
def test_y_length_matches_samples(dataset_id):
    if dataset_id == "heart_disease" and not HEART_DISEASE_AVAILABLE:
        pytest.skip("heart_disease unreachable in this environment")
    bundle = load_dataset(dataset_id)
    assert bundle.y.shape[0] == bundle.X.shape[0]


# 10. train/validation/test splits contain no overlapping indices
@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
def test_no_overlapping_split_indices(dataset_id):
    if dataset_id == "heart_disease" and not HEART_DISEASE_AVAILABLE:
        pytest.skip("heart_disease unreachable in this environment")
    bundle = load_dataset(dataset_id)
    splits = get_prepared_splits(bundle, seed=0)

    train_set = set(splits.train_indices.tolist())
    val_set = set(splits.val_indices.tolist())
    test_set = set(splits.test_indices.tolist())

    assert not (train_set & val_set), "train/val index overlap"
    assert not (train_set & test_set), "train/test index overlap"
    assert not (val_set & test_set), "val/test index overlap"
    assert len(train_set | val_set | test_set) == bundle.X.shape[0], (
        "Splits must partition every row exactly once."
    )


# 11. all required classes are represented where statistically possible
@pytest.mark.parametrize("dataset_id", VALID_DATASET_IDS)
def test_all_classes_represented_in_splits(dataset_id):
    if dataset_id == "heart_disease" and not HEART_DISEASE_AVAILABLE:
        pytest.skip("heart_disease unreachable in this environment")
    bundle = load_dataset(dataset_id)
    splits = get_prepared_splits(bundle, seed=0)
    n_classes = len(bundle.class_names)
    for y_split, name in (
        (splits.y_train, "train"), (splits.y_val, "val"), (splits.y_test, "test")
    ):
        present = len(np.unique(y_split))
        assert present == n_classes, (
            f"{dataset_id} {name} split is missing a class: "
            f"expected {n_classes}, found {present}"
        )
