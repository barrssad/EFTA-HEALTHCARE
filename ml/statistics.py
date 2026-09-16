from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover - keep the safer fallback for lean installs.
    stats = None

PAIRING_KEY_FIELDS = (
    "dataset",
    "model",
    "scenario",
    "shift_type",
    "shift_level",
    "seed",
)


def _policy_label(policy: str) -> str:
    map_ = {
        "efta": "EFTA",
        "b1_confidence_only": "B1",
        "b2_weighted_index": "B2",
    }
    return map_.get(str(policy).lower(), str(policy))


def _pairing_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, Any] | None:
    """Return the canonical seed-pairing key.

    The exact grouping key is:
    (dataset, model, scenario, shift_type, shift_level, seed)
    where missing shift_type or shift_level are recorded as empty strings,
    and rows without dataset/model/scenario/seed are skipped.
    """
    dataset = row.get("dataset")
    model = row.get("model")
    scenario = row.get("scenario")
    if scenario is None:
        scenario = row.get("shift_condition")
    seed = row.get("seed")
    if dataset is None or model is None or scenario is None or seed is None:
        return None
    shift_type = row.get("shift_type")
    shift_level = row.get("shift_level")
    return (
        str(dataset),
        str(model),
        str(scenario),
        "" if shift_type is None else str(shift_type),
        "" if shift_level is None else str(shift_level),
        seed,
    )


def compute_coverage(accepted: Sequence[bool] | np.ndarray) -> float:
    """Return accepted proportion among all eligible cases."""
    values = np.asarray(accepted, dtype=bool).reshape(-1)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def compute_abstention_rate(accepted: Sequence[bool] | np.ndarray) -> float:
    """Return 1 - coverage."""
    coverage = compute_coverage(accepted)
    if np.isnan(coverage):
        return float("nan")
    return float(1.0 - coverage)


def compute_selective_error(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    accepted: Sequence[bool] | np.ndarray,
) -> float:
    """Return error among accepted cases, with NaN for zero accepted cases."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(y_pred, dtype=int).reshape(-1)
    a = np.asarray(accepted, dtype=bool).reshape(-1)
    if y.size == 0 or p.size == 0 or a.size == 0:
        return float("nan")
    if y.size != p.size or y.size != a.size:
        raise ValueError("y_true, y_pred, and accepted must have the same length.")
    accepted_mask = a
    if not accepted_mask.any():
        return float("nan")
    errors = (p[accepted_mask] != y[accepted_mask]).astype(float)
    return float(np.mean(errors))


def matched_coverage_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    target_coverage: float = 0.70,
    tolerance: float = 0.05,
    reference_policy: str = "efta",
    comparator_policy: str | Sequence[str] | None = None,
    comparator_policies: Sequence[str] | None = None,
    analysis_type: str = "secondary",
    hypothesis: str = "null",
    metric: str = "selective_error",
) -> list[dict[str, Any]]:
    """Match seed-level EFTA/comparator rows near the target coverage.

    Pairing is keyed by:
    (dataset, model, scenario, shift_type, shift_level, seed)
    Rows that do not have complete pairing metadata are skipped. A single call
    must only compare one comparator policy to EFTA; mixing B1 and B2 in the
    same pairing call is rejected.
    """
    if comparator_policy is not None and comparator_policies is not None:
        raise ValueError("Pass either comparator_policy or comparator_policies, not both.")
    if comparator_policies is not None:
        comparator_policy = comparator_policies
    if comparator_policy is None:
        raise ValueError("Exactly one comparator policy must be supplied for each pairing call.")
    if isinstance(comparator_policy, (list, tuple, set, np.ndarray)):
        values = [str(item) for item in comparator_policy]
        unique = {value.lower() for value in values}
        if len(unique) != 1:
            raise ValueError("A single matched_coverage_pairs call must not mix B1 and B2 comparator policies.")
        comparator_policy = values[0]

    grouped: dict[tuple[str, str, str, str, str, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _pairing_key(row)
        if key is None:
            continue
        grouped[key].append(row)

    matched: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        entries = grouped[key]
        reference_candidates = [
            row for row in entries if str(row.get("policy", "")).lower() == str(reference_policy).lower()
        ]
        if len(reference_candidates) != 1:
            continue
        reference = reference_candidates[0]
        reference_coverage = float(reference.get("achieved_coverage", np.nan))
        if not np.isfinite(reference_coverage):
            continue
        comparator_candidates = [
            row for row in entries if str(row.get("policy", "")).lower() == str(comparator_policy).lower()
        ]
        if not comparator_candidates:
            raise ValueError(f"No comparator rows available for key {key!r} and policy {comparator_policy!r}.")
        valid = [
            row for row in comparator_candidates
            if np.isfinite(float(row.get("achieved_coverage", np.nan)))
            and abs(float(row.get("achieved_coverage", np.nan)) - target_coverage) <= tolerance
        ]
        if not valid:
            raise ValueError(
                f"No paired coverage match for key {key!r} under policy {comparator_policy!r} within tolerance {tolerance}; "
                f"reference coverage was {reference_coverage}."
            )
        comparator = min(valid, key=lambda row: abs(float(row.get("achieved_coverage", np.nan)) - target_coverage))
        dataset, model, scenario, shift_type, shift_level, seed = key
        matched.append(
            {
                "dataset": dataset,
                "model": model,
                "scenario": scenario,
                "shift_type": shift_type,
                "shift_level": shift_level,
                "seed": seed,
                "seed_ids": [seed],
                "paired_seed_ids": [seed],
                "reference_policy": reference_policy,
                "comparator_policy": comparator_policy,
                "comparison": f"{_policy_label(reference_policy)}_vs_{_policy_label(comparator_policy)}",
                "analysis_type": analysis_type,
                "hypothesis": hypothesis,
                "metric": metric,
                "target_coverage": float(target_coverage),
                "achieved_coverage": float(comparator.get("achieved_coverage", np.nan)),
                "coverage_gap": abs(float(comparator.get("achieved_coverage", np.nan)) - target_coverage),
                "coverage_tolerance": float(tolerance),
                "reference_row": dict(reference),
                "comparator_row": dict(comparator),
                "status": "estimated",
                "reason": "paired_seed_coverage_match",
            }
        )
    if not matched:
        raise ValueError("No valid paired observations remain after enforcing complete pairing metadata and comparator policy separation.")
    return matched


def paired_seed_differences(
    reference: Sequence[float] | np.ndarray,
    comparator: Sequence[float] | np.ndarray,
    *,
    reference_name: str = "efta",
    comparator_name: str = "comparator",
) -> np.ndarray:
    """Return paired differences with a consistent sign convention.

    The default convention is EFTA minus comparator, so a negative value means
    the reference policy performs better on the metric.
    """
    left = np.asarray(reference, dtype=float).reshape(-1)
    right = np.asarray(comparator, dtype=float).reshape(-1)
    if left.size != right.size:
        raise ValueError("reference and comparator must have the same length.")
    if left.size == 0:
        return np.asarray([], dtype=float)
    del reference_name, comparator_name
    return left - right


def wilcoxon_signed_rank(
    reference: Sequence[float] | np.ndarray,
    comparator: Sequence[float] | np.ndarray,
    *,
    alternative: str = "less",
) -> dict[str, Any]:
    """Perform a paired Wilcoxon signed-rank test on seed-level differences."""
    left = np.asarray(reference, dtype=float).reshape(-1)
    right = np.asarray(comparator, dtype=float).reshape(-1)
    if left.size != right.size:
        raise ValueError("reference and comparator must have same length.")
    if left.size == 0:
        return {"n_pairs": 0, "statistic": np.nan, "p_value": np.nan, "alternative": alternative, "status": "not_estimable", "reason": "No paired observations."}
    diff = left - right
    finite = np.isfinite(diff)
    if finite.sum() < 2:
        return {"n_pairs": int(finite.sum()), "statistic": np.nan, "p_value": np.nan, "alternative": alternative, "status": "not_estimable", "reason": "Fewer than two usable paired seeds."}
    signed = diff[finite]
    if np.allclose(signed, 0.0):
        return {"n_pairs": int(finite.sum()), "statistic": 0.0, "p_value": 1.0, "alternative": alternative, "status": "degenerate", "reason": "No paired variation."}
    if stats is None:
        return {"n_pairs": int(finite.sum()), "statistic": np.nan, "p_value": np.nan, "alternative": alternative, "status": "not_estimable", "reason": "scipy unavailable."}
    try:
        statistic, p_value = stats.wilcoxon(signed, alternative=alternative, zero_method="wilcox")
    except ValueError:
        return {"n_pairs": int(finite.sum()), "statistic": np.nan, "p_value": np.nan, "alternative": alternative, "status": "not_estimable", "reason": "Wilcoxon statistic not defined for this paired sample."}
    return {"n_pairs": int(finite.sum()), "statistic": float(statistic), "p_value": float(p_value), "alternative": alternative, "status": "estimated", "reason": "paired_seed_wilcoxon"}


def paired_bootstrap_interval(
    reference: Sequence[float] | np.ndarray,
    comparator: Sequence[float] | np.ndarray,
    *,
    iterations: int = 200,
    random_state: int = 0,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap the seed-level paired mean difference with deterministic resampling."""
    left = np.asarray(reference, dtype=float).reshape(-1)
    right = np.asarray(comparator, dtype=float).reshape(-1)
    if left.size != right.size:
        raise ValueError("reference and comparator must have the same length.")
    if left.size < 2 or iterations < 1:
        return {
            "lower": np.nan,
            "upper": np.nan,
            "mean_difference": np.nan,
            "method": "paired_seed_bootstrap",
            "iterations": int(iterations),
            "random_state": int(random_state),
            "confidence_level": float(confidence_level),
            "status": "not_estimable",
            "reason": "Fewer than two paired seeds or insufficient bootstrap iterations.",
        }
    diff = left - right
    finite = np.isfinite(diff)
    diff = diff[finite]
    if diff.size < 2:
        return {
            "lower": np.nan,
            "upper": np.nan,
            "mean_difference": np.nan,
            "method": "paired_seed_bootstrap",
            "iterations": int(iterations),
            "random_state": int(random_state),
            "confidence_level": float(confidence_level),
            "status": "not_estimable",
            "reason": "Too few finite paired differences for bootstrap.",
        }
    rng = np.random.default_rng(random_state)
    estimates = []
    for _ in range(int(iterations)):
        idx = rng.integers(0, diff.size, diff.size)
        estimates.append(float(np.mean(diff[idx])))
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "mean_difference": float(np.mean(diff)),
        "method": "paired_seed_bootstrap",
        "iterations": int(iterations),
        "random_state": int(random_state),
        "confidence_level": float(confidence_level),
        "status": "estimated",
        "reason": "paired_seed_bootstrap",
    }


def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR adjustment to a family of p-values.

    Finite p-values are corrected in the original order. NaN and infinite values
    remain undefined and yield NaN in the output.
    """
    values = np.asarray(p_values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return adjusted
    finite_values = values[finite_mask]
    order = np.argsort(finite_values, kind="stable")
    sorted_values = finite_values[order]
    m = sorted_values.size
    sorted_adjusted = np.empty(m, dtype=float)
    running = 1.0
    for index in range(m - 1, -1, -1):
        rank = index + 1
        running = min(running, sorted_values[index] * m / rank)
        sorted_adjusted[index] = running
    mapped = np.empty_like(sorted_adjusted)
    mapped[order] = np.clip(sorted_adjusted, 0.0, 1.0)
    adjusted[finite_mask] = mapped
    return adjusted


def deterioration(clean: Sequence[float] | np.ndarray, shifted: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return shifted - clean as a descriptive robustness delta."""
    clean_values = np.asarray(clean, dtype=float).reshape(-1)
    shifted_values = np.asarray(shifted, dtype=float).reshape(-1)
    if clean_values.size != shifted_values.size:
        raise ValueError("clean and shifted must have the same length.")
    if clean_values.size == 0:
        return np.asarray([], dtype=float)
    return shifted_values - clean_values


def summarize_seed_comparisons(
    pairings: Sequence[dict[str, Any]],
    *,
    metric: str,
    comparison: str,
    analysis_type: str = "secondary",
    hypothesis: str = "null",
    target_coverage: float | None = None,
    coverage_tolerance: float | None = None,
) -> list[dict[str, Any]]:
    """Summarize per-seed paired observations into machine-readable comparison rows.

    Each summary preserves the original paired seed IDs and the metadata needed to
    trace the aggregate statistic back to the exact raw pairings.
    """
    if not pairings:
        return []

    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairings:
        dataset = str(pair.get("dataset", ""))
        model = str(pair.get("model", ""))
        scenario = str(pair.get("scenario", ""))
        shift_type = str(pair.get("shift_type", ""))
        shift_level = str(pair.get("shift_level", ""))
        grouped[(dataset, model, scenario, shift_type, shift_level, str(comparison))].append(pair)

    for (dataset, model, scenario, shift_type, shift_level, comparison_name), group in sorted(
        grouped.items(), key=lambda item: item[0]
    ):
        reference_values = []
        comparator_values = []
        paired_seed_ids = []
        for entry in sorted(group, key=lambda row: str(row.get("seed", ""))):
            reference = entry.get("reference_row", {})
            comparator = entry.get("comparator_row", {})
            ref_metric = reference.get(metric)
            comp_metric = comparator.get(metric)
            if np.isfinite(ref_metric) and np.isfinite(comp_metric):
                reference_values.append(float(ref_metric))
                comparator_values.append(float(comp_metric))
                paired_seed_ids.append(int(entry.get("seed", -1)))
        if not reference_values:
            summaries.append(
                {
                    "comparison": comparison_name,
                    "metric": metric,
                    "dataset": dataset,
                    "model": model,
                    "scenario": scenario,
                    "shift_type": shift_type,
                    "shift_level": shift_level,
                    "paired_seed_ids": [],
                    "n_pairs": 0,
                    "paired_mean_difference": np.nan,
                    "paired_median_difference": np.nan,
                    "confidence_interval": {"lower": np.nan, "upper": np.nan},
                    "test_name": "wilcoxon_signed_rank",
                    "test_statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size": np.nan,
                    "analysis_type": analysis_type,
                    "hypothesis": hypothesis,
                    "target_coverage": target_coverage,
                    "coverage_tolerance": coverage_tolerance,
                    "status": "not_estimable",
                    "reason": "No finite paired seed metrics available.",
                }
            )
            continue

        diffs = paired_seed_differences(reference_values, comparator_values)
        ci = paired_bootstrap_interval(reference_values, comparator_values)
        test = wilcoxon_signed_rank(reference_values, comparator_values, alternative="less")
        summaries.append(
            {
                "comparison": comparison_name,
                "metric": metric,
                "dataset": dataset,
                "model": model,
                "scenario": scenario,
                "shift_type": shift_type,
                "shift_level": shift_level,
                "paired_seed_ids": sorted(paired_seed_ids),
                "n_pairs": int(len(diffs)),
                "paired_mean_difference": float(np.mean(diffs)) if len(diffs) else np.nan,
                "paired_median_difference": float(np.median(diffs)) if len(diffs) else np.nan,
                "confidence_interval": {
                    "lower": ci.get("lower", np.nan),
                    "upper": ci.get("upper", np.nan),
                },
                "test_name": "wilcoxon_signed_rank",
                "test_statistic": test.get("statistic", np.nan),
                "p_value": test.get("p_value", np.nan),
                "effect_size": float(np.mean(diffs)) if len(diffs) else np.nan,
                "analysis_type": analysis_type,
                "hypothesis": hypothesis,
                "target_coverage": target_coverage,
                "coverage_tolerance": coverage_tolerance,
                "status": test.get("status", "estimated"),
                "reason": test.get("reason", "paired_seed_summary"),
            }
        )
    return summaries


def deterioration_summary(
    clean_rows: Sequence[dict[str, Any]],
    shifted_rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    policy: str,
    dataset: str,
    model: str,
    scenario: str,
    shift_type: str | None = None,
    shift_level: str | None = None,
    analysis_type: str = "secondary",
    hypothesis: str = "H3",
) -> list[dict[str, Any]]:
    """Return a minimal scenario-matched deterioration summary for a single policy."""
    clean_map = {int(row.get("seed")): row for row in clean_rows if row.get("seed") is not None}
    shifted_map = {int(row.get("seed")): row for row in shifted_rows if row.get("seed") is not None}
    paired_ids = sorted(set(clean_map) & set(shifted_map))
    rows: list[dict[str, Any]] = []
    for seed in paired_ids:
        clean_metric = clean_map[seed].get(metric)
        shifted_metric = shifted_map[seed].get(metric)
        if np.isfinite(clean_metric) and np.isfinite(shifted_metric):
            rows.append({"seed": seed, "clean": float(clean_metric), "shifted": float(shifted_metric)})
    if not rows:
        return [{
            "comparison": f"{_policy_label(policy)}_deterioration",
            "metric": metric,
            "dataset": dataset,
            "model": model,
            "scenario": scenario,
            "shift_type": shift_type,
            "shift_level": shift_level,
            "paired_seed_ids": [],
            "n_pairs": 0,
            "paired_mean_difference": np.nan,
            "paired_median_difference": np.nan,
            "confidence_interval": {"lower": np.nan, "upper": np.nan},
            "test_name": "paired_seed_deterioration",
            "test_statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "analysis_type": analysis_type,
            "hypothesis": hypothesis,
            "status": "not_estimable",
            "reason": "No finite paired seed deterioration values available.",
        }]
    deltas = deterioration([row["clean"] for row in rows], [row["shifted"] for row in rows])
    ci = paired_bootstrap_interval(np.asarray([row["clean"] for row in rows], dtype=float), np.asarray([row["shifted"] for row in rows], dtype=float))
    return [{
        "comparison": f"{_policy_label(policy)}_deterioration",
        "metric": metric,
        "dataset": dataset,
        "model": model,
        "scenario": scenario,
        "shift_type": shift_type,
        "shift_level": shift_level,
        "paired_seed_ids": sorted([row["seed"] for row in rows]),
        "n_pairs": len(rows),
        "paired_mean_difference": float(np.mean(deltas)),
        "paired_median_difference": float(np.median(deltas)),
        "confidence_interval": {"lower": ci.get("lower", np.nan), "upper": ci.get("upper", np.nan)},
        "test_name": "paired_seed_deterioration",
        "test_statistic": np.nan,
        "p_value": np.nan,
        "effect_size": float(np.mean(deltas)),
        "analysis_type": analysis_type,
        "hypothesis": hypothesis,
        "status": "estimated",
        "reason": "shifted_minus_clean_seed_deterioration",
    }]
