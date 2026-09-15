from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_ROOT = PROJECT_ROOT / "ml"
RESULTS_DIR = ML_ROOT / "results"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from config import CONFIG  # noqa: E402


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def get_experiment_results() -> dict[str, Any]:
    """Load the exact files produced by ml/experiments.py, if present."""
    raw_path = RESULTS_DIR / "raw_results.csv"
    comparison_path = RESULTS_DIR / "comparison_table.md"
    if not raw_path.exists() or not comparison_path.exists():
        return {
            "available": False,
            "source": "ml/results/raw_results.csv and ml/results/comparison_table.md",
            "generated_by": "ml/experiments.py",
            "methodology": {},
            "summary": [],
            "raw_results": [],
            "raw_result_count": 0,
            "message": "Experiment results are not currently available. Run `cd ml && python3 experiments.py`.",
        }

    raw = pd.read_csv(raw_path)
    summary = (
        raw.groupby(["model", "condition"], as_index=False)[
            ["coverage", "selective_error", "accuracy", "unsupported_release_rate"]
        ].mean()
    )
    return {
        "available": True,
        "source": "ml/results/raw_results.csv and ml/results/comparison_table.md",
        "generated_by": "ml/experiments.py",
        "methodology": {
            "dataset": "Wisconsin Diagnostic Breast Cancer (WDBC)",
            "models": ["logistic_regression", "random_forest"],
            "seeds": CONFIG["seeds"],
            "split": "60% train / 20% validation / 20% held-out test, stratified",
            "calibration": "Sigmoid calibration fitted on the validation split",
            "threshold_target_coverage": CONFIG["target_coverage"],
            "coverage_tolerance": CONFIG["coverage_tolerance"],
            "shift_conditions": ["clean_test", "gaussian_noise_0.1", "gaussian_noise_0.2", "random_missingness_10%", "random_missingness_20%"],
            "metrics": ["coverage", "selective_error", "accuracy", "unsupported_release_rate"],
            "gate_scope": "The generated batch output contains selective prediction and shift metrics; it does not contain dataset-level G1/G2/G3 gate outcomes.",
        },
        "summary": _json_records(summary),
        "raw_results": _json_records(raw),
        "raw_result_count": int(len(raw)),
        "message": None,
    }
