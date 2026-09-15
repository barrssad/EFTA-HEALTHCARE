"""Deterministic distribution-shift foundations for held-out evaluation.

These utilities transform or resample only the held-out test rows. They never
fit preprocessing on shifted data and never mutate the source dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data.split_protocol import PreparedSplits
from .data.unified_interface import DatasetBundle

NoiseLevel = Literal[0.1, 0.2]
MissingnessLevel = Literal[0.1, 0.2]

# Explicit protocol choices. Age and sex are immutable/protected for local
# perturbations; Heart's coded categorical variables are not continuous noise targets.
PROTECTED_FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "synthetic": (),
    "heart_disease": ("age", "sex"),
    "wdbc": (),
}
CONTINUOUS_NOISE_FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "synthetic": tuple(f"feature_{i}" for i in range(10)),
    "heart_disease": ("trestbps", "chol", "thalach", "oldpeak"),
    "wdbc": (
        "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
        "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
        "mean fractal dimension", "radius error", "texture error", "perimeter error",
        "area error", "smoothness error", "compactness error", "concavity error",
        "concave points error", "symmetry error", "fractal dimension error", "worst radius",
        "worst texture", "worst perimeter", "worst area", "worst smoothness",
        "worst compactness", "worst concavity", "worst concave points", "worst symmetry",
        "worst fractal dimension",
    ),
}


@dataclass(frozen=True)
class ShiftedTestData:
    """Raw and prepared held-out data produced by one deterministic shift."""

    dataset_id: str
    shift_name: str
    X_raw: np.ndarray
    X_prepared: np.ndarray
    y: np.ndarray
    source_indices: np.ndarray
    changed_feature_indices: tuple[int, ...] = ()
    changed_row_mask: np.ndarray | None = None


def _feature_indices(bundle: DatasetBundle, names: tuple[str, ...]) -> tuple[int, ...]:
    missing = [name for name in names if name not in bundle.feature_names]
    if missing:
        raise ValueError(f"Dataset {bundle.dataset_id!r} is missing declared features: {missing}")
    return tuple(bundle.feature_names.index(name) for name in names)


def permissible_continuous_noise_indices(bundle: DatasetBundle) -> tuple[int, ...]:
    """Return the explicit continuous, non-protected noise feature set."""
    protected = set(_feature_indices(bundle, PROTECTED_FEATURE_NAMES[bundle.dataset_id]))
    continuous = _feature_indices(
        bundle, CONTINUOUS_NOISE_FEATURE_NAMES[bundle.dataset_id]
    )
    if protected.intersection(continuous):
        raise RuntimeError("Protected features must never be in the noise feature set.")
    return tuple(index for index in continuous if index not in protected)


def permissible_missingness_indices(bundle: DatasetBundle) -> tuple[int, ...]:
    """Return every mutable feature eligible for missingness masking."""
    protected = set(_feature_indices(bundle, PROTECTED_FEATURE_NAMES[bundle.dataset_id]))
    return tuple(index for index in range(len(bundle.feature_names)) if index not in protected)


def _prepare_shifted(
    bundle: DatasetBundle,
    splits: PreparedSplits,
    raw_X: np.ndarray,
    y: np.ndarray,
    source_indices: np.ndarray,
    shift_name: str,
    changed_feature_indices: tuple[int, ...] = (),
    changed_row_mask: np.ndarray | None = None,
) -> ShiftedTestData:
    pipeline = splits.preprocessing_pipeline
    if np.isnan(raw_X).any() and "imputer" not in pipeline.named_steps:
        raise ValueError(
            "Missingness shifts require splits prepared with force_imputation=True; "
            "the frozen training imputer must handle shifted test values."
        )
    return ShiftedTestData(
        dataset_id=bundle.dataset_id,
        shift_name=shift_name,
        X_raw=np.asarray(raw_X, dtype=float),
        X_prepared=np.asarray(pipeline.transform(raw_X), dtype=float),
        y=np.asarray(y, dtype=int),
        source_indices=np.asarray(source_indices, dtype=int),
        changed_feature_indices=changed_feature_indices,
        changed_row_mask=changed_row_mask,
    )


def measurement_noise(
    bundle: DatasetBundle,
    splits: PreparedSplits,
    noise_scale: NoiseLevel,
    seed: int,
) -> ShiftedTestData:
    """Add Gaussian noise scaled by raw training SD to permissible features only."""
    if noise_scale not in (0.1, 0.2):
        raise ValueError("noise_scale must be exactly 0.1 or 0.2.")
    indices = permissible_continuous_noise_indices(bundle)
    raw_test = np.asarray(bundle.X[splits.test_indices], dtype=float).copy()
    raw_train = np.asarray(bundle.X[splits.train_indices], dtype=float)
    training_std = np.nanstd(raw_train[:, indices], axis=0)
    rng = np.random.default_rng(seed)
    raw_test[:, indices] += rng.normal(
        loc=0.0, scale=float(noise_scale) * training_std, size=(len(raw_test), len(indices))
    )
    return _prepare_shifted(
        bundle,
        splits,
        raw_test,
        bundle.y[splits.test_indices],
        splits.test_indices,
        f"gaussian_noise_{noise_scale:.1f}",
        changed_feature_indices=indices,
    )


def missingness(
    bundle: DatasetBundle,
    splits: PreparedSplits,
    missing_fraction: MissingnessLevel,
    seed: int,
) -> ShiftedTestData:
    """Mask mutable held-out inputs and transform with the frozen train imputer."""
    if missing_fraction not in (0.1, 0.2):
        raise ValueError("missing_fraction must be exactly 0.1 or 0.2.")
    indices = permissible_missingness_indices(bundle)
    raw_test = np.asarray(bundle.X[splits.test_indices], dtype=float).copy()
    rng = np.random.default_rng(seed)
    changed = rng.random((len(raw_test), len(indices))) < float(missing_fraction)
    raw_test[:, indices] = np.where(changed, np.nan, raw_test[:, indices])
    row_mask = changed.any(axis=1)
    return _prepare_shifted(
        bundle,
        splits,
        raw_test,
        bundle.y[splits.test_indices],
        splits.test_indices,
        f"random_missingness_{missing_fraction:.0%}",
        changed_feature_indices=indices,
        changed_row_mask=row_mask,
    )


def prevalence_shift(
    bundle: DatasetBundle,
    splits: PreparedSplits,
    target_positive_prevalence: float,
    seed: int,
) -> ShiftedTestData:
    """Resample held-out rows by class to alter prevalence, preserving class conditionals."""
    if not 0.0 < target_positive_prevalence < 1.0:
        raise ValueError("target_positive_prevalence must be strictly between 0 and 1.")
    rng = np.random.default_rng(seed)
    test_indices = np.asarray(splits.test_indices, dtype=int)
    test_y = np.asarray(bundle.y[test_indices], dtype=int)
    negative = test_indices[test_y == 0]
    positive = test_indices[test_y == 1]
    if len(negative) == 0 or len(positive) == 0:
        raise ValueError("Prevalence shift requires both classes in the test split.")
    n_positive = int(np.floor(len(test_indices) * target_positive_prevalence + 0.5))
    n_positive = min(max(n_positive, 1), len(test_indices) - 1)
    sampled_negative = rng.choice(negative, size=len(test_indices) - n_positive, replace=True)
    sampled_positive = rng.choice(positive, size=n_positive, replace=True)
    source_indices = np.concatenate((sampled_negative, sampled_positive))
    rng.shuffle(source_indices)
    return _prepare_shifted(
        bundle,
        splits,
        bundle.X[source_indices],
        bundle.y[source_indices],
        source_indices,
        f"prevalence_{target_positive_prevalence:.0%}",
    )


def age_subgroup(
    bundle: DatasetBundle,
    splits: PreparedSplits,
    min_samples: int = 20,
) -> ShiftedTestData | None:
    """Return the held-out at-or-above-training-median age subgroup if sufficient."""
    if min_samples < 1:
        raise ValueError("min_samples must be positive.")
    if "age" not in bundle.feature_names:
        return None
    age_index = bundle.feature_names.index("age")
    training_age = np.asarray(bundle.X[splits.train_indices, age_index], dtype=float)
    test_ages = np.asarray(bundle.X[splits.test_indices, age_index], dtype=float)
    threshold = float(np.nanmedian(training_age))
    selected = np.flatnonzero(test_ages >= threshold)
    if len(selected) < min_samples:
        return None
    source_indices = np.asarray(splits.test_indices[selected], dtype=int)
    return _prepare_shifted(
        bundle,
        splits,
        bundle.X[source_indices],
        bundle.y[source_indices],
        source_indices,
        f"age_at_or_above_training_median_{threshold:.2f}",
    )


def shift_ready_splits(
    bundle: DatasetBundle, seed: int, scale_features: bool
) -> PreparedSplits:
    """Build ordinary splits with a frozen train imputer for missingness shifts."""
    from .data.split_protocol import get_prepared_splits

    return get_prepared_splits(
        bundle,
        seed=seed,
        scale_features=scale_features,
        force_imputation=True,
    )
