from __future__ import annotations

import numpy as np
import pytest

from ml.statistics import (
    benjamini_hochberg,
    compute_abstention_rate,
    compute_coverage,
    compute_selective_error,
    deterioration,
    matched_coverage_pairs,
    paired_bootstrap_interval,
    paired_seed_differences,
    summarize_seed_comparisons,
    wilcoxon_signed_rank,
)


def test_selective_error_coverage_and_abstention_are_correct() -> None:
    accepted = np.array([True, True, False, False])
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])

    assert compute_selective_error(y_true, y_pred, accepted) == pytest.approx(0.5)
    assert compute_coverage(accepted) == pytest.approx(0.5)
    assert compute_abstention_rate(accepted) == pytest.approx(0.5)


def test_matched_coverage_pairs_require_seed_alignment_and_tolerance() -> None:
    rows = [
        {"dataset": "heart_disease", "model": "logistic_regression", "scenario": "clean_test", "seed": 1, "policy": "efta", "achieved_coverage": 0.69, "selective_error": 0.12},
        {"dataset": "heart_disease", "model": "logistic_regression", "scenario": "clean_test", "seed": 1, "policy": "b1_confidence_only", "achieved_coverage": 0.71, "selective_error": 0.15},
        {"dataset": "heart_disease", "model": "logistic_regression", "scenario": "clean_test", "seed": 2, "policy": "efta", "achieved_coverage": 0.68, "selective_error": 0.10},
        {"dataset": "heart_disease", "model": "logistic_regression", "scenario": "clean_test", "seed": 2, "policy": "b1_confidence_only", "achieved_coverage": 0.72, "selective_error": 0.14},
    ]
    paired = matched_coverage_pairs(rows, comparator_policy="b1_confidence_only", target_coverage=0.70, tolerance=0.05)
    assert paired[0]["seed"] == 1
    assert paired[1]["seed"] == 2
    assert len(paired) == 2

    with pytest.raises(ValueError):
        matched_coverage_pairs(
            [
                {"dataset": "heart_disease", "model": "logistic_regression", "scenario": "clean_test", "seed": 1, "policy": "efta", "achieved_coverage": 0.69},
                {"dataset": "wdbc", "model": "logistic_regression", "scenario": "clean_test", "seed": 2, "policy": "b1_confidence_only", "achieved_coverage": 0.71},
            ],
            comparator_policy="b1_confidence_only",
            target_coverage=0.70,
            tolerance=0.05,
        )


def test_seed_level_differences_and_wilcoxon_are_paired() -> None:
    efta = np.array([0.10, 0.20, 0.15, 0.25, 0.18])
    b1 = np.array([0.18, 0.24, 0.22, 0.30, 0.29])
    diff = paired_seed_differences(efta, b1, reference_name="efta_minus_comparator")
    assert np.all(diff < 0)
    assert diff.shape == (5,)

    result = wilcoxon_signed_rank(efta, b1, alternative="less")
    assert result["n_pairs"] == 5
    assert result["p_value"] >= 0.0
    assert result["statistic"] >= 0.0


def test_bootstrap_ci_and_sign_for_differences_are_reproducible() -> None:
    efta = np.array([0.30, 0.40])
    b1 = np.array([0.50, 0.60])
    diff = paired_seed_differences(efta, b1, reference_name="efta_minus_comparator")
    ci = paired_bootstrap_interval(efta, b1, iterations=200, random_state=7)
    assert np.all(diff < 0)
    assert ci["lower"] < 0.0 < ci["upper"] or ci["lower"] < ci["upper"]
    assert ci["method"] == "paired_seed_bootstrap"

    second = paired_bootstrap_interval(efta, b1, iterations=200, random_state=7)
    assert ci == second


def test_bh_correction_and_shift_deterioration_are_descriptive() -> None:
    p_values = np.array([0.01, 0.04, 0.20, 0.50])
    corrected = benjamini_hochberg(p_values)
    assert corrected[0] <= corrected[1]
    assert corrected[-1] >= 0.0

    clean = np.array([0.10, 0.20, 0.18])
    shifted = np.array([0.14, 0.24, 0.18])
    delta = deterioration(clean, shifted)
    assert np.all(delta >= 0.0)

    efta_deterioration = np.array([0.02, 0.03])
    b1_deterioration = np.array([0.05, 0.08])
    diff = paired_seed_differences(efta_deterioration, b1_deterioration)
    assert np.all(diff < 0)


def test_missing_and_zero_variance_are_handled_safely() -> None:
    assert np.isnan(compute_selective_error([], [], []))
    assert np.isnan(compute_coverage(np.array([], dtype=bool)))
    assert np.isnan(compute_abstention_rate(np.array([], dtype=bool)))
    assert np.isnan(wilcoxon_signed_rank([], [], alternative="less")["p_value"])
    assert np.isnan(paired_bootstrap_interval(np.array([0.5]), np.array([0.5]), iterations=50, random_state=0)["lower"])


def test_pairing_rejects_cross_dataset_model_and_scenario_mismatches() -> None:
    base = {"dataset": "heart_disease", "model": "logistic_regression", "scenario": "clean_test", "seed": 40, "achieved_coverage": 0.70}
    with pytest.raises(ValueError):
        matched_coverage_pairs(
            [
                {**base, "policy": "efta", "dataset": "heart_disease"},
                {**base, "policy": "b1_confidence_only", "dataset": "wdbc"},
            ],
            comparator_policy="b1_confidence_only",
            target_coverage=0.70,
            tolerance=0.05,
        )

    with pytest.raises(ValueError):
        matched_coverage_pairs(
            [
                {**base, "policy": "efta", "model": "logistic_regression"},
                {**base, "policy": "b1_confidence_only", "model": "random_forest"},
            ],
            comparator_policy="b1_confidence_only",
            target_coverage=0.70,
            tolerance=0.05,
        )

    with pytest.raises(ValueError):
        matched_coverage_pairs(
            [
                {**base, "policy": "efta", "scenario": "clean_test"},
                {**base, "policy": "b1_confidence_only", "scenario": "measurement_noise_0_1"},
            ],
            comparator_policy="b1_confidence_only",
            target_coverage=0.70,
            tolerance=0.05,
        )


def test_efta_vs_b1_and_efta_vs_b2_are_kept_separate() -> None:
    rows = [
        {"dataset": "wdbc", "model": "logistic_regression", "scenario": "clean_test", "seed": 40, "policy": "efta", "achieved_coverage": 0.70, "selective_error": 0.12},
        {"dataset": "wdbc", "model": "logistic_regression", "scenario": "clean_test", "seed": 40, "policy": "b1_confidence_only", "achieved_coverage": 0.70, "selective_error": 0.18},
        {"dataset": "wdbc", "model": "logistic_regression", "scenario": "clean_test", "seed": 40, "policy": "b2_weighted_index", "achieved_coverage": 0.70, "unsupported_release_rate": 0.26},
    ]

    b1_pairing = matched_coverage_pairs(rows, comparator_policy="b1_confidence_only", target_coverage=0.70, tolerance=0.05)
    b2_pairing = matched_coverage_pairs(rows, comparator_policy="b2_weighted_index", target_coverage=0.70, tolerance=0.05)

    assert len(b1_pairing) == 1
    assert b1_pairing[0]["comparison"] == "EFTA_vs_B1"
    assert len(b2_pairing) == 1
    assert b2_pairing[0]["comparison"] == "EFTA_vs_B2"

    with pytest.raises(ValueError):
        matched_coverage_pairs(rows, comparator_policy=["b1_confidence_only", "b2_weighted_index"], target_coverage=0.70, tolerance=0.05)


def test_primary_secondary_and_exploratory_metadata_are_machine_readable() -> None:
    rows = [{
        "dataset": "wdbc",
        "model": "logistic_regression",
        "scenario": "clean_test",
        "shift_type": "",
        "shift_level": "",
        "seed": 40,
        "reference_policy": "efta",
        "reference_row": {"selective_error": 0.12},
        "comparator_policy": "b1_confidence_only",
        "comparator_row": {"selective_error": 0.18},
        "target_coverage": 0.70,
        "coverage_tolerance": 0.05,
    }]

    summary = summarize_seed_comparisons(
        rows,
        metric="selective_error",
        comparison="EFTA_vs_B1",
        analysis_type="primary",
        hypothesis="H1",
        target_coverage=0.70,
        coverage_tolerance=0.05,
    )
    assert summary[0]["analysis_type"] == "primary"
    assert summary[0]["hypothesis"] == "H1"
    assert summary[0]["comparison"] == "EFTA_vs_B1"
    assert summary[0]["target_coverage"] == 0.70
    assert summary[0]["coverage_tolerance"] == 0.05
    assert summary[0]["paired_seed_ids"] == [40]

    rows2 = [
        {"dataset": "wdbc", "model": "logistic_regression", "scenario": "measurement_noise_0_1", "seed": 41, "reference_row": {"selective_error": 0.17}, "comparator_row": {"selective_error": 0.23}},
        {"dataset": "wdbc", "model": "logistic_regression", "scenario": "measurement_noise_0_2", "seed": 42, "reference_row": {"selective_error": 0.15}, "comparator_row": {"selective_error": 0.19}},
    ]

    summary2 = summarize_seed_comparisons(rows2, metric="selective_error", comparison="EFTA_vs_B1", analysis_type="secondary", hypothesis="H3")
    assert summary2[0]["analysis_type"] == "secondary"
    assert summary2[0]["hypothesis"] == "H3"


def test_bh_handles_undefined_values_and_preserves_order() -> None:
    p_values = np.array([0.01, np.nan, 0.04, np.inf, 0.20])
    adjusted = benjamini_hochberg(p_values)
    assert adjusted.shape == p_values.shape
    assert np.isfinite(adjusted[0])
    assert np.isnan(adjusted[1])
    assert np.isinf(p_values[3])
    assert np.isnan(adjusted[3])
    assert np.all(np.isfinite(benjamini_hochberg(np.array([0.01, 0.04, 0.20]))))
    assert benjamini_hochberg(np.array([], dtype=float)).size == 0
    assert np.all(np.isnan(benjamini_hochberg(np.array([np.nan, np.nan]))))

    original = np.array([0.5, 0.1, 0.2])
    transformed = benjamini_hochberg(original)
    assert np.allclose(transformed, np.array([0.5, 0.3, 0.3]), atol=1e-8)
    assert np.allclose(benjamini_hochberg(np.array([0.01, 0.02, 0.03])), np.array([0.03, 0.03, 0.03]), atol=1e-8)


def test_summary_is_deterministic_and_traceable_to_seed_ids() -> None:
    rows = [
        {
            "dataset": "heart_disease",
            "model": "random_forest",
            "scenario": "clean_test",
            "shift_type": "",
            "shift_level": "",
            "seed": 42,
            "reference_row": {"selective_error": 0.25},
            "comparator_row": {"selective_error": 0.30},
        },
        {
            "dataset": "heart_disease",
            "model": "random_forest",
            "scenario": "clean_test",
            "shift_type": "",
            "shift_level": "",
            "seed": 40,
            "reference_row": {"selective_error": 0.20},
            "comparator_row": {"selective_error": 0.28},
        },
    ]
    summary = summarize_seed_comparisons(
        rows,
        metric="selective_error",
        comparison="EFTA_vs_B1",
        analysis_type="primary",
        hypothesis="H1",
        target_coverage=0.70,
        coverage_tolerance=0.05,
    )
    assert [entry["paired_seed_ids"] for entry in summary] == [[40, 42]]
    assert summary[0]["paired_seed_ids"] == [40, 42]
    assert summary[0]["n_pairs"] == 2


def test_missing_metadata_is_skipped_and_fewer_than_two_pairs_are_not_estimable() -> None:
    rows = [
        {"seed": 40, "policy": "efta", "achieved_coverage": 0.70},
        {"seed": 40, "policy": "b1_confidence_only", "achieved_coverage": 0.70},
    ]
    with pytest.raises(ValueError):
        matched_coverage_pairs(rows, comparator_policy="b1_confidence_only", target_coverage=0.70, tolerance=0.05)

    result = wilcoxon_signed_rank(np.array([0.10]), np.array([0.12]), alternative="less")
    assert result["status"] == "not_estimable"
    assert np.isnan(result["p_value"])


def test_zero_accepted_cases_are_nan() -> None:
    assert np.isnan(compute_selective_error(np.array([1, 0]), np.array([1, 1]), np.array([False, False])))
