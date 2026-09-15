"""Release-policy baselines over shared model and explanation evidence.

Thresholds are fitted from validation evidence and then reused unchanged on test
or deployment evidence. This module does not run model or explanation methods;
callers compute those inputs once and pass them to every policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

try:
    from .gates import efta_decision
except ImportError:  # Support execution from the ml directory.
    from gates import efta_decision

PolicyName = Literal["b0_accuracy_only", "b1_confidence_only", "b2_weighted_index", "efta"]


@dataclass(frozen=True)
class CaseEvidence:
    """Shared per-case evidence consumed by all release policies."""

    confidence: float
    faithfulness: float
    stability: float
    g1: int = 0
    g2: int = 0
    g3: int = 0
    model_eligible: bool = True

    def __post_init__(self) -> None:
        for name in ("confidence", "faithfulness", "stability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        for name in ("g1", "g2", "g3"):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1.")


@dataclass(frozen=True)
class PolicyCalibration:
    """Validation-fitted policy settings frozen for later evaluation."""

    target_coverage: float = 0.70
    coverage_tolerance: float = 0.05
    confidence_threshold: float | None = None
    weighted_index_threshold: float | None = None
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    weights_provisional: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.target_coverage <= 1.0:
            raise ValueError("target_coverage must be in (0, 1].")
        if self.coverage_tolerance < 0.0:
            raise ValueError("coverage_tolerance must be non-negative.")
        if len(self.weights) != 3 or any(weight < 0.0 for weight in self.weights):
            raise ValueError("weights must contain three non-negative values.")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("weights must sum to 1.")


@dataclass(frozen=True)
class PolicyDecision:
    policy: PolicyName
    accepted: bool
    score: float | None
    threshold: float | None
    reason: str


def normalized_scores(evidence: Sequence[CaseEvidence]) -> np.ndarray:
    """Return [confidence, faithfulness, stability] scores in [0, 1].

    The uncertainty component is represented as the normalized uncertainty
    support score ``1 - uncertainty = confidence`` so that larger is better.
    """
    return np.asarray(
        [[item.confidence, item.faithfulness, item.stability] for item in evidence],
        dtype=float,
    )


def weighted_index(
    evidence: CaseEvidence | Sequence[CaseEvidence],
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> float | np.ndarray:
    """Compute the compensatory weighted index without gate conjunction."""
    rows = [evidence] if isinstance(evidence, CaseEvidence) else list(evidence)
    if len(weights) != 3 or any(weight < 0.0 for weight in weights):
        raise ValueError("weights must contain three non-negative values.")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("weights must sum to 1.")
    scores = normalized_scores(rows)
    result = scores @ np.asarray(weights, dtype=float)
    return float(result[0]) if isinstance(evidence, CaseEvidence) else result


def _select_threshold(scores: np.ndarray, calibration: PolicyCalibration) -> float:
    """Choose a threshold using validation scores only, targeting coverage."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) == 0:
        raise ValueError("validation scores cannot be empty.")
    candidates = np.unique(np.concatenate(([0.0], values, [1.0])))
    records = []
    for threshold in candidates:
        coverage = float(np.mean(values >= threshold))
        records.append((threshold, coverage))
    in_tolerance = [
        item for item in records
        if abs(item[1] - calibration.target_coverage) <= calibration.coverage_tolerance
    ]
    eligible = in_tolerance or records
    return float(min(eligible, key=lambda item: (abs(item[1] - calibration.target_coverage), item[0]))[0])


def fit_b1_calibration(
    validation_confidence: Sequence[float],
    target_coverage: float = 0.70,
    coverage_tolerance: float = 0.05,
) -> PolicyCalibration:
    """Fit B1's confidence threshold from validation confidence only."""
    calibration = PolicyCalibration(
        target_coverage=target_coverage,
        coverage_tolerance=coverage_tolerance,
    )
    threshold = _select_threshold(np.asarray(validation_confidence), calibration)
    return PolicyCalibration(**{**calibration.__dict__, "confidence_threshold": threshold})


def fit_b2_calibration(
    validation_evidence: Sequence[CaseEvidence],
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    target_coverage: float = 0.70,
    coverage_tolerance: float = 0.05,
) -> PolicyCalibration:
    """Fit B2's threshold from validation index scores; weights stay fixed."""
    calibration = PolicyCalibration(
        target_coverage=target_coverage,
        coverage_tolerance=coverage_tolerance,
        weights=weights,
    )
    threshold = _select_threshold(weighted_index(validation_evidence, weights), calibration)
    return PolicyCalibration(**{**calibration.__dict__, "weighted_index_threshold": threshold})


def apply_policy(
    policy: PolicyName,
    evidence: CaseEvidence,
    calibration: PolicyCalibration | None = None,
) -> PolicyDecision:
    """Apply one policy to shared evidence and frozen calibration settings."""
    settings = calibration or PolicyCalibration()
    if not evidence.model_eligible:
        return PolicyDecision(policy, False, None, None, "model_not_eligible")
    if policy == "b0_accuracy_only":
        return PolicyDecision(policy, True, None, None, "model_eligible")
    if policy == "b1_confidence_only":
        if settings.confidence_threshold is None:
            raise ValueError("B1 requires a validation-fitted confidence threshold.")
        accepted = evidence.confidence >= settings.confidence_threshold
        return PolicyDecision(
            policy, accepted, evidence.confidence, settings.confidence_threshold,
            "confidence_pass" if accepted else "confidence_below_threshold",
        )
    if policy == "b2_weighted_index":
        if settings.weighted_index_threshold is None:
            raise ValueError("B2 requires a validation-fitted weighted-index threshold.")
        score = float(weighted_index(evidence, settings.weights))
        accepted = score >= settings.weighted_index_threshold
        return PolicyDecision(
            policy, accepted, score, settings.weighted_index_threshold,
            "weighted_index_pass" if accepted else "weighted_index_below_threshold",
        )
    if policy == "efta":
        accepted = bool(efta_decision(evidence.g1, evidence.g2, evidence.g3) == "ACCEPT_FOR_REVIEW")
        return PolicyDecision(
            policy, accepted, None, None,
            "all_case_gates_pass" if accepted else "mandatory_case_gate_failed",
        )
    raise ValueError(f"Unknown policy: {policy!r}")
