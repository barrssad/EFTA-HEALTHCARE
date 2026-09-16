"""Reusable descriptive metrics for confirmatory EFTA result generation."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


def _binary_arrays(y_true: Sequence[int], probabilities: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("y_true and probabilities must have equal length.")
    if len(y) and not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if len(p) and (not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("probabilities must be finite values in [0, 1].")
    return y, p


def _bootstrap_binary_metric(y: np.ndarray, p: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float], *, iterations: int, random_state: int) -> tuple[float, float]:
    if len(y) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    estimates: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y), len(y))
        try:
            estimate = metric(y[indices], p[indices])
        except ValueError:
            continue
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return (float("nan"), float("nan"))
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(y) < 2 or len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return float("nan"), float("nan")
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    fit.fit(logits, y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def expected_calibration_error(y_true: Sequence[int], probabilities: Sequence[float], *, n_bins: int = 10) -> dict[str, Any]:
    """Calculate descriptive equal-width ECE and return its explicit bins."""
    y, p = _binary_arrays(y_true, probabilities)
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    weighted_error = 0.0
    for index in range(n_bins):
        mask = (p >= edges[index]) & (p <= edges[index + 1] if index == n_bins - 1 else p < edges[index + 1])
        count = int(mask.sum())
        mean_probability = float(np.mean(p[mask])) if count else float("nan")
        event_rate = float(np.mean(y[mask])) if count else float("nan")
        weighted_error += count / len(y) * abs(mean_probability - event_rate) if count and len(y) else 0.0
        rows.append({"bin": index, "lower": float(edges[index]), "upper": float(edges[index + 1]), "count": count, "mean_probability": mean_probability, "event_rate": event_rate})
    return {"ece": float(weighted_error) if len(y) else float("nan"), "n_bins": n_bins, "binning": "equal_width_[0,1]", "bins": rows}


def predictive_metrics(y_true: Sequence[int], probabilities: Sequence[float], threshold: float, *, threshold_split: str = "validation", bootstrap_iterations: int = 1000, random_state: int = 0, ece_bins: int = 10) -> dict[str, Any]:
    """Return predictive metrics with deterministic 95% percentile CIs."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if threshold_split.lower() == "test":
        raise ValueError("thresholds must be selected on validation data, never test data.")
    y, p = _binary_arrays(y_true, probabilities)
    predictions = (p >= threshold).astype(int)
    sensitivity = float(np.mean(predictions[y == 1] == 1)) if np.any(y == 1) else float("nan")
    specificity = float(np.mean(predictions[y == 0] == 0)) if np.any(y == 0) else float("nan")
    auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    auprc = float(average_precision_score(y, p)) if len(y) else float("nan")
    brier = float(brier_score_loss(y, p)) if len(y) else float("nan")
    ece = expected_calibration_error(y, p, n_bins=ece_bins)
    intercept, slope = _calibration(y, p)
    return {"auroc": auroc, "auroc_ci95": _bootstrap_binary_metric(y, p, roc_auc_score, iterations=bootstrap_iterations, random_state=random_state), "auprc": auprc, "auprc_ci95": _bootstrap_binary_metric(y, p, average_precision_score, iterations=bootstrap_iterations, random_state=random_state + 1), "positive_prevalence": float(np.mean(y)) if len(y) else float("nan"), "sensitivity": sensitivity, "specificity": specificity, "brier_score": brier, "brier_score_ci95": _bootstrap_binary_metric(y, p, brier_score_loss, iterations=bootstrap_iterations, random_state=random_state + 2), "calibration_intercept": intercept, "calibration_slope": slope, "ece": ece["ece"], "ece_detail": ece, "threshold": float(threshold), "threshold_split": threshold_split, "n": len(y)}


def risk_coverage_curve(y_true: Sequence[int], probabilities: Sequence[float]) -> list[dict[str, float]]:
    """Return descending-confidence risk/coverage points."""
    y, p = _binary_arrays(y_true, probabilities)
    if len(y) == 0:
        return []
    confidence = np.maximum(p, 1.0 - p)
    errors = ((p >= 0.5).astype(int) != y).astype(float)
    order = np.argsort(-confidence, kind="stable")
    cumulative = np.cumsum(errors[order])
    return [{"coverage": float(count / len(y)), "risk": float(error_count / count)} for count, error_count in enumerate(cumulative, start=1)]


def selective_metrics(y_true: Sequence[int], probabilities: Sequence[float], threshold: float, *, threshold_split: str = "validation") -> dict[str, Any]:
    """Calculate selective prediction quantities at a frozen confidence threshold."""
    if threshold_split.lower() == "test":
        raise ValueError("thresholds must be selected on validation data, never test data.")
    y, p = _binary_arrays(y_true, probabilities)
    confidence = np.maximum(p, 1.0 - p)
    accepted = confidence >= threshold
    errors = (p >= 0.5).astype(int) != y
    accepted_count = int(accepted.sum())
    rejected_count = int((~accepted).sum())
    curve = risk_coverage_curve(y, p)
    x = np.asarray([point["coverage"] for point in curve])
    risk = np.asarray([point["risk"] for point in curve])
    aurc = float(np.trapz(risk, x)) if len(curve) else float("nan")
    return {"realized_coverage": float(accepted_count / len(y)) if len(y) else float("nan"), "selective_error": float(np.mean(errors[accepted])) if accepted_count else float("nan"), "abstention_rate": float(rejected_count / len(y)) if len(y) else float("nan"), "rejected_case_error": float(np.mean(errors[~accepted])) if rejected_count else float("nan"), "aurc": aurc, "risk_coverage_curve": curve, "accepted_count": accepted_count, "rejected_count": rejected_count, "denominator": len(y), "threshold": float(threshold), "threshold_split": threshold_split}


def explanation_faithfulness(model: Any, X: np.ndarray, attributions: np.ndarray, neutralize: Callable[[np.ndarray, np.ndarray], np.ndarray], *, top_k: int = 3, target_class: int = 1) -> dict[str, Any]:
    """Measure predicted-class probability change after neutralising top-k features."""
    features = np.asarray(X, dtype=float)
    values = np.asarray(attributions, dtype=float)
    if features.ndim != 2 or values.shape != features.shape:
        raise ValueError("X and attributions must have the same 2-D shape.")
    changes: list[float] = []
    for row, attribution in zip(features, values):
        predicted = int(np.argmax(model.predict_proba(row.reshape(1, -1))[0]))
        indices = np.argsort(np.abs(attribution))[-top_k:][::-1]
        neutralized = neutralize(row.copy(), indices)
        before = float(model.predict_proba(row.reshape(1, -1))[0, predicted])
        after = float(model.predict_proba(np.asarray(neutralized).reshape(1, -1))[0, predicted])
        changes.append(before - after)
    return {"top_k": top_k, "target_class": target_class, "mean_probability_change": float(np.mean(changes)) if changes else float("nan"), "changes": changes}


def explanation_stability(original: np.ndarray, perturbed: np.ndarray, *, top_k: int = 3) -> dict[str, float]:
    """Return top-k Jaccard overlap and Spearman rank correlation."""
    baseline = np.asarray(original, dtype=float).reshape(-1)
    matrix = np.asarray(perturbed, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(baseline):
        raise ValueError("perturbed attributions must be 2-D and feature-aligned.")
    base_order = np.argsort(-np.abs(baseline), kind="stable")
    base_top = set(base_order[:top_k].tolist())
    jaccards: list[float] = []
    correlations: list[float] = []
    base_ranks = np.empty(len(base_order), dtype=float)
    base_ranks[base_order] = np.arange(len(base_order))
    for row in matrix:
        row_order = np.argsort(-np.abs(row), kind="stable")
        row_top = set(row_order[:top_k].tolist())
        union = base_top | row_top
        jaccards.append(len(base_top & row_top) / len(union) if union else 1.0)
        row_ranks = np.empty(len(row_order), dtype=float)
        row_ranks[row_order] = np.arange(len(row_order))
        correlations.append(float(np.corrcoef(base_ranks, row_ranks)[0, 1]) if len(row) > 1 else float("nan"))
    return {"jaccard_overlap": float(np.mean(jaccards)) if jaccards else float("nan"), "spearman_rank_correlation": float(np.nanmean(correlations)) if correlations else float("nan"), "top_k": top_k, "n_perturbations": len(matrix)}


def synthetic_recovery(attributions: Sequence[float], known_effects: Sequence[float], *, top_k: int) -> dict[str, float]:
    """Compare top-k attributed features with known synthetic effects."""
    values = np.asarray(attributions, dtype=float).reshape(-1)
    effects = np.asarray(known_effects, dtype=float).reshape(-1)
    if values.shape != effects.shape:
        raise ValueError("attributions and known_effects must have equal length.")
    attributed = set(np.argsort(-np.abs(values), kind="stable")[:top_k].tolist())
    known = set(np.argsort(-np.abs(effects), kind="stable")[:top_k].tolist())
    overlap = len(attributed & known)
    value_order = np.argsort(np.argsort(-np.abs(values), kind="stable"))
    effect_order = np.argsort(np.argsort(-np.abs(effects), kind="stable"))
    correlation = float(np.corrcoef(value_order, effect_order)[0, 1]) if len(values) > 1 else float("nan")
    return {"precision_at_k": float(overlap / top_k) if top_k else float("nan"), "recall_at_k": float(overlap / len(known)) if known else float("nan"), "rank_correlation": float(correlation), "top_k": top_k}


def unsupported_release_rate(accepted: Sequence[bool], faithfulness: Sequence[float], stability: Sequence[float], *, faithfulness_threshold: float, stability_threshold: float) -> float:
    """Rate accepted cases failing either frozen explanation criterion."""
    release = np.asarray(accepted, dtype=bool)
    faithful = np.asarray(faithfulness, dtype=float)
    stable = np.asarray(stability, dtype=float)
    if not (len(release) == len(faithful) == len(stable)):
        raise ValueError("accepted, faithfulness, and stability must have equal length.")
    denominator = int(release.sum())
    failures = release & ((faithful < faithfulness_threshold) | (stable < stability_threshold))
    return float(failures.sum() / denominator) if denominator else float("nan")


def subgroup_metrics(y_true: Sequence[int], probabilities: Sequence[float], subgroup: Iterable[Any], threshold: float, *, min_size: int = 20) -> list[dict[str, Any]]:
    """Audit adequately sized subgroups; flag and suppress very small groups."""
    y, p = _binary_arrays(y_true, probabilities)
    groups = np.asarray(list(subgroup), dtype=object)
    if len(groups) != len(y):
        raise ValueError("subgroup must have one value per observation.")
    overall = selective_metrics(y, p, threshold)
    results = []
    for label in sorted(set(groups.tolist()), key=str):
        mask = groups == label
        count = int(mask.sum())
        row: dict[str, Any] = {"subgroup": label, "sample_size": count, "event_count": int(y[mask].sum()), "adequate": count >= min_size, "status": "reported" if count >= min_size else "suppressed_small_subgroup"}
        if count < min_size:
            row.update({"auroc": None, "balanced_accuracy": None, "brier_score": None, "selective_error": None, "realized_coverage": None, "absolute_coverage_gap": None, "absolute_error_gap": None})
        else:
            selective = selective_metrics(y[mask], p[mask], threshold)
            row.update({"auroc": float(roc_auc_score(y[mask], p[mask])) if len(np.unique(y[mask])) == 2 else None, "balanced_accuracy": float(balanced_accuracy_score(y[mask], p[mask] >= 0.5)), "brier_score": float(brier_score_loss(y[mask], p[mask])), "selective_error": selective["selective_error"], "realized_coverage": selective["realized_coverage"], "absolute_coverage_gap": abs(selective["realized_coverage"] - overall["realized_coverage"]), "absolute_error_gap": abs(selective["selective_error"] - overall["selective_error"]) if np.isfinite(overall["selective_error"]) and np.isfinite(selective["selective_error"]) else None})
        results.append(row)
    return results