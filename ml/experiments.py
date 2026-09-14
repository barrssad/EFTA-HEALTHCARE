"""Reproducible EFTA experiments across seeds and simulated shifts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from config import CONFIG
from data_loader import load_and_split
from gates import default_shap_explainer, efta_decision, gate_g1_local_uncertainty
from models import fit_calibrated_model, gate_g0_metrics, gate_g5_subgroup_check

OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def _candidate_thresholds() -> list[float]:
    return [round(float(x), 2) for x in np.arange(0.50, 1.00, 0.01)]


def lock_gate_threshold(
    model: Any, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Choose the confidence threshold whose validation coverage is closest to 70%."""
    records = []
    for threshold in _candidate_thresholds():
        confidence = np.max(model.predict_proba(X_val), axis=1)
        accepted = confidence >= threshold
        coverage = float(np.mean(accepted))
        records.append({"threshold": threshold, "coverage": coverage})
    target = CONFIG["target_coverage"]
    tolerance = CONFIG["coverage_tolerance"]
    in_tolerance = [r for r in records if abs(r["coverage"] - target) <= tolerance]
    candidates = in_tolerance or records
    best = min(candidates, key=lambda r: (abs(r["coverage"] - target), r["threshold"]))
    best["within_tolerance"] = float(abs(best["coverage"] - target) <= tolerance)
    return float(best["threshold"]), best


def _evaluate_predictions(
    probabilities: np.ndarray, y: np.ndarray, threshold: float, condition: str, seed: int, model_name: str
) -> dict[str, Any]:
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    accepted = confidence >= threshold
    predictions = (probabilities >= 0.5).astype(int)
    n_accepted = int(accepted.sum())
    errors = predictions != y
    selective_error = float(errors[accepted].mean()) if n_accepted else 0.0
    accuracy = float(accuracy_score(y, predictions))
    unsupported_release_rate = float(np.mean(accepted & errors))
    return {
        "model": model_name,
        "seed": seed,
        "condition": condition,
        "coverage": float(np.mean(accepted)),
        "selective_error": selective_error,
        "accuracy": accuracy,
        "unsupported_release_rate": unsupported_release_rate,
        "n": len(y),
        "n_accepted": n_accepted,
    }


def run_one_seed(seed: int, model_name: str) -> list[dict[str, Any]]:
    splits = load_and_split(seed)
    model = fit_calibrated_model(
        model_name, splits.X_train, splits.y_train, splits.X_val, splits.y_val
    )
    threshold, lock = lock_gate_threshold(model, splits.X_val, splits.y_val)
    rows = [
        _evaluate_predictions(
            model.predict_proba(splits.X_test)[:, 1], splits.y_test,
            threshold, "clean_test", seed, model_name
        )
    ]
    rows[0].update({"locked_threshold": threshold, "validation_coverage": lock["coverage"]})
    rng = np.random.default_rng(seed)
    for noise in (0.1, 0.2):
        X_shifted = splits.X_test + rng.normal(0.0, noise, size=splits.X_test.shape)
        rows.append(_evaluate_predictions(
            model.predict_proba(X_shifted)[:, 1], splits.y_test,
            threshold, f"gaussian_noise_{noise:.1f}", seed, model_name
        ))
    for missingness in (0.10, 0.20):
        X_masked = splits.X_test.copy()
        mask = rng.random(X_masked.shape) < missingness
        X_masked[mask] = 0.0  # zero is the training mean in standardized space
        rows.append(_evaluate_predictions(
            model.predict_proba(X_masked)[:, 1], splits.y_test,
            threshold, f"random_missingness_{missingness:.0%}", seed, model_name
        ))
    for row in rows:
        row.setdefault("locked_threshold", threshold)
        row.setdefault("validation_coverage", lock["coverage"])
    return rows


def run_experiments(model_names: tuple[str, ...] = ("logistic_regression", "random_forest")) -> pd.DataFrame:
    """Run all seeds/models sequentially and save raw and summary results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for seed in CONFIG["seeds"]:
            rows.extend(run_one_seed(seed, model_name))
    raw = pd.DataFrame(rows)
    raw_path = OUTPUT_DIR / "raw_results.csv"
    raw.to_csv(raw_path, index=False)
    summary = (
        raw.groupby(["model", "condition"], as_index=False)[
            ["coverage", "selective_error", "accuracy", "unsupported_release_rate"]
        ].mean()
    )
    summary_path = OUTPUT_DIR / "comparison_table.md"
    summary.to_markdown(summary_path, index=False, floatfmt=".4f")
    print(summary.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nRaw results: {raw_path}\nMarkdown table: {summary_path}")
    return raw


if __name__ == "__main__":
    run_experiments()
