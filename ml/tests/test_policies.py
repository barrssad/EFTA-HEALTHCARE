"""Part 3 release-policy baseline tests."""

from __future__ import annotations

import numpy as np
import pytest

from ml.policies import (
    CaseEvidence,
    PolicyCalibration,
    apply_policy,
    fit_b1_calibration,
    fit_b2_calibration,
    normalized_scores,
    weighted_index,
)


def evidence(
    confidence: float = 0.9,
    faithfulness: float = 0.8,
    stability: float = 0.7,
    g1: int = 1,
    g2: int = 1,
    g3: int = 1,
    model_eligible: bool = True,
) -> CaseEvidence:
    return CaseEvidence(
        confidence, faithfulness, stability, g1, g2, g3, model_eligible
    )


def test_b0_releases_all_model_eligible_cases_without_case_gates() -> None:
    decision = apply_policy(
        "b0_accuracy_only", evidence(confidence=0.51, g1=0, g2=0, g3=0)
    )
    assert decision.accepted is True
    assert decision.reason == "model_eligible"

    blocked = apply_policy("b0_accuracy_only", evidence(model_eligible=False))
    assert blocked.accepted is False


def test_b1_uses_confidence_only() -> None:
    calibration = PolicyCalibration(confidence_threshold=0.8)
    accepted = apply_policy(
        "b1_confidence_only",
        evidence(confidence=0.81, faithfulness=0.0, stability=0.0, g1=0, g2=0, g3=0),
        calibration,
    )
    rejected = apply_policy(
        "b1_confidence_only",
        evidence(confidence=0.79, faithfulness=1.0, stability=1.0, g1=1, g2=1, g3=1),
        calibration,
    )
    assert accepted.accepted is True
    assert rejected.accepted is False


def test_b1_threshold_is_fitted_from_validation_only() -> None:
    validation = [0.55, 0.65, 0.75, 0.85, 0.95]
    calibration = fit_b1_calibration(validation, target_coverage=0.6, coverage_tolerance=0.0)
    assert calibration.confidence_threshold == 0.75
    assert apply_policy("b1_confidence_only", evidence(confidence=0.75), calibration).accepted
    assert apply_policy("b1_confidence_only", evidence(confidence=0.74), calibration).accepted is False


def test_b2_normalization_and_configurable_weights() -> None:
    items = [evidence(0.2, 0.4, 0.6), evidence(1.0, 1.0, 1.0)]
    np.testing.assert_allclose(normalized_scores(items), [[0.2, 0.4, 0.6], [1.0, 1.0, 1.0]])
    assert weighted_index(items[0], (0.2, 0.3, 0.5)) == pytest.approx(0.46)

    with pytest.raises(ValueError):
        weighted_index(items[0], (0.2, 0.2, 0.2))


def test_b2_is_compensatory_not_and_logic() -> None:
    calibration = PolicyCalibration(
        weighted_index_threshold=0.6,
        weights=(0.5, 0.25, 0.25),
    )
    item = evidence(confidence=1.0, faithfulness=0.2, stability=0.2, g1=1, g2=0, g3=0)
    decision = apply_policy("b2_weighted_index", item, calibration)
    assert decision.score == pytest.approx(0.6)
    assert decision.accepted is True


def test_b2_threshold_and_weights_are_validation_fitted_and_frozen() -> None:
    validation = [evidence(0.4, 0.4, 0.4), evidence(0.8, 0.8, 0.8)]
    calibration = fit_b2_calibration(
        validation,
        weights=(0.5, 0.25, 0.25),
        target_coverage=0.5,
        coverage_tolerance=0.0,
    )
    assert calibration.weights == (0.5, 0.25, 0.25)
    assert calibration.weights_provisional is True
    assert calibration.weighted_index_threshold == pytest.approx(0.8)
    assert apply_policy("b2_weighted_index", validation[1], calibration).accepted is True
    assert apply_policy("b2_weighted_index", validation[0], calibration).accepted is False


def test_efta_uses_and_logic_without_compensation() -> None:
    calibration = PolicyCalibration(confidence_threshold=0.1, weighted_index_threshold=0.1)
    for failed_gate in ("g1", "g2", "g3"):
        values = {"g1": 1, "g2": 1, "g3": 1}
        values[failed_gate] = 0
        decision = apply_policy("efta", evidence(**values), calibration)
        assert decision.accepted is False

    assert apply_policy("efta", evidence(), calibration).accepted is True


def test_policy_decisions_are_deterministic() -> None:
    item = evidence()
    calibration = PolicyCalibration(confidence_threshold=0.8)
    assert apply_policy("b1_confidence_only", item, calibration) == apply_policy(
        "b1_confidence_only", item, calibration
    )
