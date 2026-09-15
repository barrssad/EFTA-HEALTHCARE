"""Deterministic synthetic dataset for construct-validity / explanation-recovery tests.

Design goals (per research protocol):
- exact ground-truth relevant features are known and stored as metadata
- non-trivial: uses a nonlinear (interaction) mechanism, not a simple threshold
- includes both informative and pure-noise features
- fully reproducible from (n_samples, seed) alone
"""

from __future__ import annotations

import numpy as np

TRUE_FEATURE_INDICES: list[int] = [0, 1, 2]
TRUE_FEATURE_NAMES: list[str] = ["feature_0", "feature_1", "feature_2"]
N_NOISE_FEATURES: int = 7
N_FEATURES: int = len(TRUE_FEATURE_INDICES) + N_NOISE_FEATURES


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_synthetic_dataset(
    n_samples: int = 900, seed: int = 0, label_noise: float = 0.05
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Generate a synthetic binary classification dataset with known-relevant features.

    Mechanism (deliberately nonlinear/non-trivial):
        z = 1.5*x0 - 1.2*x1 + 0.9*(x0 * x2) + noise
        y = Bernoulli(sigmoid(z)), then `label_noise` fraction of labels are flipped

    Only features at TRUE_FEATURE_INDICES (0, 1, 2) causally influence y.
    Features 3..9 are independent standard-normal noise with no effect on y,
    so a faithful explainer should never rank them highly.

    Returns
    -------
    X : (n_samples, N_FEATURES) float array
    y : (n_samples,) int array, values in {0, 1}
    true_feature_indices : list[int], copy of TRUE_FEATURE_INDICES
    feature_names : list[str], length N_FEATURES
    """
    if n_samples < 100:
        raise ValueError("n_samples must be >= 100 for stable train/val/test splitting.")

    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, N_FEATURES))

    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    z = 1.5 * x0 - 1.2 * x1 + 0.9 * (x0 * x2) + rng.normal(0.0, 0.3, size=n_samples)
    p = _sigmoid(z)
    y = (rng.random(n_samples) < p).astype(int)

    if label_noise > 0:
        flip_mask = rng.random(n_samples) < label_noise
        y[flip_mask] = 1 - y[flip_mask]

    if y.mean() < 0.05 or y.mean() > 0.95:
        raise RuntimeError(
            "Synthetic label distribution degenerated (near-constant y); "
            "adjust the coefficients in generate_synthetic_dataset before use."
        )

    feature_names = TRUE_FEATURE_NAMES + [f"feature_{i}" for i in range(3, N_FEATURES)]
    return X, y, list(TRUE_FEATURE_INDICES), feature_names
