from __future__ import annotations

import numpy as np
import pytest

from ml.metrics import (
    explanation_stability,
    predictive_metrics,
    risk_coverage_curve,
    selective_metrics,
    subgroup_metrics,
    synthetic_recovery,
    unsupported_release_rate,
)
from ml.result_tables import ResultContext, risk_coverage_data, table2_metrics


def test_predictive_metrics_and_explicit_ece_bins() -> None:
    result = predictive_metrics([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], 0.5, bootstrap_iterations=30)
    assert result["auroc"] == 1.0
    assert result["positive_prevalence"] == 0.5
    assert result["ece_detail"]["n_bins"] == 10
    assert len(result["ece_detail"]["bins"]) == 10


def test_bootstrap_is_reproducible_and_test_threshold_is_rejected() -> None:
    first = predictive_metrics([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6], 0.5, bootstrap_iterations=50, random_state=9)
    second = predictive_metrics([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6], 0.5, bootstrap_iterations=50, random_state=9)
    assert first["auroc_ci95"] == second["auroc_ci95"]
    with pytest.raises(ValueError, match="validation"):
        predictive_metrics([0, 1], [0.2, 0.8], 0.5, threshold_split="test")


def test_selective_empty_acceptance_and_zero_denominators() -> None:
    result = selective_metrics([0, 1], [0.5, 0.5], 1.0)
    assert np.isnan(result["selective_error"])
    assert result["abstention_rate"] == 1.0
    assert np.isfinite(result["rejected_case_error"])
    assert selective_metrics([], [], 0.5)["denominator"] == 0
    assert risk_coverage_curve([], []) == []


def test_explanation_and_subgroup_edge_cases() -> None:
    stability = explanation_stability(np.array([3.0, 1.0, 0.1]), np.array([[3.0, 1.0, 0.2]]), top_k=2)
    assert stability["jaccard_overlap"] == 1.0
    recovery = synthetic_recovery([3, 1, 0], [2, 1, 0], top_k=2)
    assert recovery["precision_at_k"] == 1.0
    assert unsupported_release_rate([True, False], [0.1, 0.9], [0.9, 0.9], faithfulness_threshold=0.2, stability_threshold=0.7) == 1.0
    assert np.isnan(unsupported_release_rate([False], [0.1], [0.1], faithfulness_threshold=0.2, stability_threshold=0.7))
    rows = subgroup_metrics([0, 1, 0], [0.2, 0.8, 0.3], ["tiny", "tiny", "large"], 0.5, min_size=2)
    statuses = {row["subgroup"]: row["status"] for row in rows}
    assert statuses["tiny"] == "reported"
    assert statuses["large"] == "suppressed_small_subgroup"


def test_result_context_is_retained_for_metrics_and_figure_data() -> None:
    context = ResultContext("synthetic", "logistic_regression", 40, "test", "frozen", "clean", 0.7, 10, "validation_target_70", "process7-v1")
    table = table2_metrics({"auroc": 0.9, "ece_detail": {"bins": []}}, context)
    assert set(context.as_dict()).issubset(table.columns)
    figure = risk_coverage_data([{"coverage": 0.5, "risk": 0.1}], context)
    assert figure.iloc[0]["split"] == "test"