"""Runtime dataset audit — checks real loaded data, never trusts declared metadata blindly."""

from __future__ import annotations

from typing import Any

import numpy as np

from .split_protocol import PreparedSplits, get_prepared_splits
from .unified_interface import DatasetBundle, load_dataset


def audit_dataset(bundle: DatasetBundle, seed: int = 0) -> dict[str, Any]:
    X, y = bundle.X, bundle.y
    n_samples, n_features = X.shape

    missing_per_feature = np.isnan(X).sum(axis=0) if bundle.has_missing_values else np.zeros(n_features)
    duplicate_rows = int(n_samples - len(np.unique(X, axis=0)))
    constant_features = [
        bundle.feature_names[i] for i in range(n_features)
        if np.nanstd(X[:, i]) == 0.0
    ]
    class_counts = {
        bundle.class_names[c] if c < len(bundle.class_names) else str(c): int(count)
        for c, count in zip(*np.unique(y, return_counts=True))
    }

    splits: PreparedSplits = get_prepared_splits(bundle, seed=seed)

    report: dict[str, Any] = {
        "dataset_id": bundle.dataset_id,
        "shape": (n_samples, n_features),
        "feature_names": bundle.feature_names,
        "dtype": str(X.dtype),
        "target_distribution": class_counts,
        "class_imbalance_ratio": (
            max(class_counts.values()) / min(class_counts.values())
            if min(class_counts.values()) > 0 else float("inf")
        ),
        "missing_values_per_feature": (
            dict(zip(bundle.feature_names, missing_per_feature.tolist()))
            if bundle.has_missing_values else "none"
        ),
        "duplicated_rows": duplicate_rows,
        "constant_features": constant_features,
        "train_size": len(splits.y_train),
        "val_size": len(splits.y_val),
        "test_size": len(splits.y_test),
        "split_warnings": splits.split_warnings,
        "ground_truth_relevant_features": bundle.ground_truth_relevant_features,
    }
    return report


def audit_all_datasets(seed: int = 0) -> dict[str, dict[str, Any]]:
    reports = {}
    for dataset_id in ("synthetic", "heart_disease", "wdbc"):
        try:
            bundle = load_dataset(dataset_id, seed=seed)
            reports[dataset_id] = audit_dataset(bundle, seed=seed)
        except Exception as exc:  # surface, don't hide, a failing dataset
            reports[dataset_id] = {"error": str(exc)}
    return reports


if __name__ == "__main__":
    import json

    print(json.dumps(audit_all_datasets(), indent=2, default=str))
