from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.experiment_service import get_experiment_results


def test_experiment_results_are_generated_and_scoped() -> None:
    result = get_experiment_results()
    assert result["available"] is True
    assert result["generated_by"] == "ml/experiments.py"
    assert result["raw_result_count"] == 100
    assert len(result["summary"]) == 10
    assert result["methodology"]["models"] == ["logistic_regression", "random_forest"]
    assert "dataset-level G1/G2/G3" in result["methodology"]["gate_scope"]
