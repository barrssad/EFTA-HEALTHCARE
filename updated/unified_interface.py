"""Unified dataset interface consumed by all downstream ML code.

Every dataset (synthetic, heart_disease, wdbc) is returned as the same
DatasetBundle shape, so the split/train/gate pipeline never needs to know
which dataset it's operating on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset_registry import DatasetMetadata, get_metadata
from .heart_disease_loader import load_heart_disease
from .synthetic_dataset import generate_synthetic_dataset
from .wdbc_loader import load_wdbc

VALID_DATASET_IDS = ("synthetic", "heart_disease", "wdbc")


@dataclass
class DatasetBundle:
    dataset_id: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    class_names: list[str]
    metadata: DatasetMetadata
    ground_truth_relevant_features: list[int] | None  # synthetic only, else None
    has_missing_values: bool


def load_dataset(dataset_id: str, seed: int = 0) -> DatasetBundle:
    """Load any registered dataset behind one consistent interface.

    `seed` only affects the synthetic generator (it is deterministic-per-seed);
    heart_disease and wdbc are fixed real data and ignore it.
    """
    if dataset_id not in VALID_DATASET_IDS:
        raise ValueError(f"dataset_id must be one of {VALID_DATASET_IDS}, got {dataset_id!r}")

    metadata = get_metadata(dataset_id)

    if dataset_id == "synthetic":
        X, y, true_idx, feature_names = generate_synthetic_dataset(seed=seed)
        return DatasetBundle(
            dataset_id=dataset_id,
            X=X,
            y=y,
            feature_names=feature_names,
            target_name="y",
            class_names=["negative_class", "positive_class"],
            metadata=metadata,
            ground_truth_relevant_features=true_idx,
            has_missing_values=False,
        )

    if dataset_id == "heart_disease":
        X, y, feature_names, class_names = load_heart_disease()
        return DatasetBundle(
            dataset_id=dataset_id,
            X=X,
            y=y,
            feature_names=feature_names,
            target_name="binary_target",
            class_names=class_names,
            metadata=metadata,
            ground_truth_relevant_features=None,
            has_missing_values=bool(np.isnan(X).any()),
        )

    # wdbc
    X, y, feature_names, class_names = load_wdbc()
    return DatasetBundle(
        dataset_id=dataset_id,
        X=X,
        y=y,
        feature_names=feature_names,
        target_name="target",
        class_names=class_names,
        metadata=metadata,
        ground_truth_relevant_features=None,
        has_missing_values=False,
    )
