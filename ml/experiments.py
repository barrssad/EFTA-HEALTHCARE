"""Reproducible EFTA experiments across datasets, policies, and shifts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from . import shifts
    from .config import CONFIG
    from .data.split_protocol import get_prepared_splits
    from .data.unified_interface import load_dataset
    from .gates import (
        default_shap_explainer,
        gate_g1_local_uncertainty,
        gate_g2_faithfulness,
        gate_g4_shift_robustness,
        top_k_indices,
    )
    from .metrics import (
        predictive_metrics,
        selective_metrics,
        subgroup_metrics,
        synthetic_recovery,
        unsupported_release_rate,
    )
    from .models import fit_calibrated_model, gate_g0_metrics, requires_feature_scaling
    from .policies import CaseEvidence, apply_policy, fit_b1_calibration, fit_b2_calibration
    from .result_tables import (
        ResultContext,
        architecture_decision_data,
        export_table,
        risk_coverage_data,
        table1_design,
        table2_metrics,
        table3_gate_ablation,
    )
except ImportError:  # Support execution from the ml directory.
    import shifts
    from config import CONFIG
    from data.split_protocol import get_prepared_splits
    from data.unified_interface import load_dataset
    from gates import default_shap_explainer, gate_g1_local_uncertainty, gate_g2_faithfulness, gate_g4_shift_robustness, top_k_indices
    from metrics import predictive_metrics, selective_metrics, subgroup_metrics, synthetic_recovery, unsupported_release_rate
    from models import fit_calibrated_model, gate_g0_metrics, requires_feature_scaling
    from policies import CaseEvidence, apply_policy, fit_b1_calibration, fit_b2_calibration
    from result_tables import ResultContext, architecture_decision_data, export_table, risk_coverage_data, table1_design, table2_metrics, table3_gate_ablation


LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
POLICIES = ("b0_accuracy_only", "b1_confidence_only", "b2_weighted_index", "efta")
DATASETS = ("wdbc", "heart_disease", "synthetic")


def _prediction_classes(model: Any, X: np.ndarray) -> np.ndarray:
    return np.argmax(model.predict_proba(X), axis=1).astype(int)


def _targeted_attributions(explainer: Any, X: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Batch SHAP calls by predicted class when the adapter supports class views."""
    output = np.empty_like(np.asarray(X, dtype=float))
    for target in np.unique(targets):
        mask = targets == target
        if hasattr(explainer, "target_class"):
            output[mask] = np.asarray(explainer(X[mask], target_class=int(target)), dtype=float)
        else:
            output[mask] = np.asarray(explainer(X[mask]), dtype=float)
    return output


def _batched_stability(
    X: np.ndarray,
    targets: np.ndarray,
    explainer: Any,
    top_k: int,
    perturbation_noise: float,
    num_perturbations: int,
    seed: int,
) -> np.ndarray:
    """Compute G3's pairwise Jaccard statistic with one SHAP batch per class."""
    patients = np.asarray(X, dtype=float)
    copies = np.stack(
        [
            patient
            + np.random.default_rng(seed + index).normal(
                0.0, perturbation_noise, size=(num_perturbations, patient.size)
            )
            for index, patient in enumerate(patients)
        ]
    )
    flattened = copies.reshape(-1, patients.shape[1])
    values = np.empty_like(flattened)
    repeated_targets = np.repeat(np.asarray(targets, dtype=int), num_perturbations)
    for target in np.unique(repeated_targets):
        mask = repeated_targets == target
        if hasattr(explainer, "target_class"):
            values[mask] = np.asarray(
                explainer(flattened[mask], target_class=int(target)), dtype=float
            )
        else:
            values[mask] = np.asarray(explainer(flattened[mask]), dtype=float)
    values = values.reshape(len(patients), num_perturbations, patients.shape[1])
    results = []
    for patient_values in values:
        sets = [set(top_k_indices(row, top_k).tolist()) for row in patient_values]
        overlaps = []
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                union = sets[left] | sets[right]
                overlaps.append(len(sets[left] & sets[right]) / len(union) if union else 1.0)
        results.append(float(np.mean(overlaps)) if overlaps else 1.0)
    return np.asarray(results, dtype=float)


def _case_evidence(
    model: Any,
    X: np.ndarray,
    attributions: np.ndarray,
    means: np.ndarray,
    explainer: Any,
    seed: int,
) -> tuple[list[CaseEvidence], list[dict[str, Any]]]:
    probabilities = model.predict_proba(X)
    confidence = np.max(probabilities, axis=1)
    targets = np.argmax(probabilities, axis=1).astype(int)
    stability_values = _batched_stability(
        X,
        targets,
        explainer,
        CONFIG["top_k_features"],
        CONFIG["perturbation_noise"],
        CONFIG["num_perturbations"],
        seed,
    )
    evidence: list[CaseEvidence] = []
    gate_rows: list[dict[str, Any]] = []
    for index, (patient, attribution, target) in enumerate(zip(X, attributions, targets)):
        g1 = gate_g1_local_uncertainty(model, patient, CONFIG["confidence_threshold"])
        g2, faithfulness_drop = gate_g2_faithfulness(
            model,
            patient,
            attribution,
            means,
            CONFIG["top_k_features"],
            CONFIG["faithfulness_drop_threshold"],
            target_class=int(target),
        )
        stability = float(stability_values[index])
        g3 = int(stability >= CONFIG["stability_threshold"])
        evidence.append(
            CaseEvidence(
                confidence=float(confidence[index]),
                faithfulness=float(np.clip(faithfulness_drop, 0.0, 1.0)),
                stability=float(np.clip(stability, 0.0, 1.0)),
                g1=g1,
                g2=g2,
                g3=g3,
            )
        )
        gate_rows.append(
            {
                "case_index": index,
                "predicted_class": int(target),
                "confidence": float(confidence[index]),
                "g1": g1,
                "g2": g2,
                "g3": g3,
                "faithfulness": float(faithfulness_drop),
                "stability": float(stability),
            }
        )
    return evidence, gate_rows


def _conditions(bundle: Any, splits: Any, seed: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    conditions = [("clean_test", splits.X_test, splits.y_test, splits.test_indices)]
    shifted_conditions = (
        shifts.measurement_noise(bundle, splits, 0.1, seed),
        shifts.measurement_noise(bundle, splits, 0.2, seed),
        shifts.missingness(bundle, splits, 0.1, seed),
        shifts.missingness(bundle, splits, 0.2, seed),
        shifts.prevalence_shift(bundle, splits, 0.70, seed),
    )
    for shifted in shifted_conditions:
        conditions.append((shifted.shift_name, shifted.X_prepared, shifted.y, shifted.source_indices))
    age = shifts.age_subgroup(bundle, splits)
    if age is not None:
        conditions.append((age.shift_name, age.X_prepared, age.y, age.source_indices))
    return conditions


def _blocked_rows(dataset_id: str, model_name: str, seed: int, reason: str) -> pd.DataFrame:
    rows = []
    for policy in POLICIES:
        rows.append(
            {
                "dataset": dataset_id,
                "model": model_name,
                "seed": seed,
                "split": "test",
                "policy": policy,
                "shift_condition": "clean_test",
                "coverage": np.nan,
                "denominator": 0,
                "threshold_id": "blocked",
                "config_id": "process7_v1",
                "model_status": "MODEL_BLOCKED",
                "decision": "MODEL_BLOCKED",
                "decision_reason": reason,
                "g1": 0,
                "g2": 0,
                "g3": 0,
            }
        )
    return pd.DataFrame(rows)


def run_one_seed(seed: int, model_name: str, dataset_id: str = "wdbc") -> pd.DataFrame:
    """Run one dataset/model/seed without tuning on held-out test data."""
    try:
        bundle = load_dataset(dataset_id, seed=seed)
    except Exception as exc:
        if dataset_id == "heart_disease":
            LOGGER.warning("Skipping unreachable heart_disease dataset: %s", exc)
            return pd.DataFrame()
        raise
    splits = get_prepared_splits(
        bundle,
        seed=seed,
        scale_features=requires_feature_scaling(model_name),
        force_imputation=True,
    )
    model = fit_calibrated_model(
        model_name, splits.X_train, splits.y_train, splits.X_val, splits.y_val
    )
    try:
        g0 = gate_g0_metrics(model, splits.X_val, splits.y_val)
    except ValueError as exc:
        return _blocked_rows(dataset_id, model_name, seed, f"G0 unavailable: {exc}")
    if not np.isfinite(g0["auroc"]) or g0["auroc"] < CONFIG["g0_min_auroc"]:
        return _blocked_rows(
            dataset_id,
            model_name,
            seed,
            f"G0 AUROC {g0['auroc']:.4f} below minimum",
        )

    explainer = default_shap_explainer(model, splits.X_train, feature_names=splits.feature_names)
    means = np.mean(splits.X_train, axis=0)
    val_targets = _prediction_classes(model, splits.X_val)
    val_attr = _targeted_attributions(explainer, splits.X_val, val_targets)
    val_evidence, _ = _case_evidence(model, splits.X_val, val_attr, means, explainer, seed)
    b1 = fit_b1_calibration(
        np.max(model.predict_proba(splits.X_val), axis=1),
        CONFIG["target_coverage"],
        CONFIG["coverage_tolerance"],
    )
    b2 = fit_b2_calibration(
        val_evidence,
        target_coverage=CONFIG["target_coverage"],
        coverage_tolerance=CONFIG["coverage_tolerance"],
    )
    calibrations = {
        "b0_accuracy_only": None,
        "b1_confidence_only": b1,
        "b2_weighted_index": b2,
        "efta": None,
    }
    threshold_ids = {
        "b0_accuracy_only": "none",
        "b1_confidence_only": "b1_validation_confidence",
        "b2_weighted_index": "b2_validation_index",
        "efta": "gates_frozen",
    }

    raw_rows: list[dict[str, Any]] = []
    metric_tables: list[pd.DataFrame] = []
    gate_rows: list[dict[str, Any]] = []
    risk_tables: list[pd.DataFrame] = []
    conditions = _conditions(bundle, splits, seed)
    b1_errors: dict[str, float] = {}
    for condition, X, y, source_indices in conditions:
        targets = _prediction_classes(model, X)
        attributions = _targeted_attributions(explainer, X, targets)
        evidence, condition_gates = _case_evidence(
            model, X, attributions, means, explainer, seed + len(condition)
        )
        probabilities = model.predict_proba(X)[:, 1]
        for policy in POLICIES:
            decisions = [apply_policy(policy, item, calibrations[policy]) for item in evidence]
            accepted = np.asarray([decision.accepted for decision in decisions], dtype=bool)
            accepted_errors = targets[accepted] != y[accepted]
            policy_coverage = float(np.mean(accepted)) if len(accepted) else np.nan
            policy_error = float(np.mean(accepted_errors)) if accepted.any() else np.nan
            standard_selective = selective_metrics(y, probabilities, 0.5)
            metric_values = predictive_metrics(
                y,
                probabilities,
                0.5,
                threshold_split="validation",
                bootstrap_iterations=200,
                random_state=seed,
            )
            metric_values.update(
                {
                    "realized_coverage": policy_coverage,
                    "selective_error": policy_error,
                    "policy_abstention_rate": float(1.0 - policy_coverage) if len(accepted) else np.nan,
                    "policy_rejected_case_error": float(np.mean(targets[~accepted] != y[~accepted])) if (~accepted).any() else np.nan,
                    "unsupported_release_rate": unsupported_release_rate(
                        accepted,
                        [item.faithfulness for item in evidence],
                        [item.stability for item in evidence],
                        faithfulness_threshold=CONFIG["faithfulness_drop_threshold"],
                        stability_threshold=CONFIG["stability_threshold"],
                    ),
                }
            )
            context = ResultContext(
                bundle.dataset_id,
                model_name,
                seed,
                "test",
                policy,
                condition,
                policy_coverage,
                len(y),
                threshold_ids[policy],
                "process7_v1",
            )
            metric_tables.append(table2_metrics(metric_values, context))
            risk_tables.append(risk_coverage_data(standard_selective["risk_coverage_curve"], context))
            if policy == "b1_confidence_only":
                b1_errors[condition] = policy_error
            for case_index, (item, decision, gate) in enumerate(
                zip(evidence, decisions, condition_gates)
            ):
                row = context.as_dict()
                row.update(
                    {
                        "case_index": case_index,
                        "source_index": int(source_indices[case_index]),
                        "y_true": int(y[case_index]),
                        "probability_positive": float(probabilities[case_index]),
                        "predicted_class": gate["predicted_class"],
                        "confidence": gate["confidence"],
                        "g1": item.g1,
                        "g2": item.g2,
                        "g3": item.g3,
                        "decision": "ACCEPT" if decision.accepted else "ABSTAIN",
                        "decision_reason": decision.reason,
                        "model_status": "ELIGIBLE",
                    }
                )
                raw_rows.append(row)
                gate_rows.append({**row, "gate_failure_reason": decision.reason})
            if "age" in bundle.feature_names or "sex" in bundle.feature_names:
                raw_groups = bundle.X[source_indices]
                for demographic in ("age", "sex"):
                    if demographic in bundle.feature_names:
                        subgroup = raw_groups[:, bundle.feature_names.index(demographic)]
                        for subgroup_row in subgroup_metrics(y, probabilities, subgroup, 0.5):
                            values = {
                                f"subgroup_{demographic}_{key}": value
                                for key, value in subgroup_row.items()
                                if key != "subgroup"
                            }
                            metric_tables.append(table2_metrics(values, context))
        if bundle.ground_truth_relevant_features is not None:
            known = np.zeros(X.shape[1], dtype=float)
            known[bundle.ground_truth_relevant_features] = 1.0
            recovery = synthetic_recovery(
                np.mean(np.abs(attributions), axis=0),
                known,
                top_k=CONFIG["top_k_features"],
            )
            recovery_context = ResultContext(
                bundle.dataset_id,
                model_name,
                seed,
                "test",
                "all",
                condition,
                None,
                len(y),
                "none",
                "process7_v1",
            )
            metric_tables.append(
                table2_metrics(
                    {f"synthetic_{key}": value for key, value in recovery.items()},
                    recovery_context,
                )
            )

    clean_error = b1_errors.get("clean_test", np.nan)
    for row in raw_rows:
        shifted_error = b1_errors.get(row["shift_condition"], np.nan)
        row["g4"] = (
            gate_g4_shift_robustness(
                clean_error,
                shifted_error,
                CONFIG["g4_degradation_threshold"],
            )
            if np.isfinite(clean_error) and np.isfinite(shifted_error)
            else np.nan
        )
    for row in gate_rows:
        shifted_error = b1_errors.get(row["shift_condition"], np.nan)
        row["g4"] = (
            gate_g4_shift_robustness(
                clean_error,
                shifted_error,
                CONFIG["g4_degradation_threshold"],
            )
            if np.isfinite(clean_error) and np.isfinite(shifted_error)
            else np.nan
        )
    gate_context = ResultContext(
        bundle.dataset_id,
        model_name,
        seed,
        "test",
        "all",
        "all",
        None,
        None,
        "gates_frozen",
        "process7_v1",
    )
    gate_table = table3_gate_ablation(gate_rows, gate_context)
    architecture = architecture_decision_data(
        [
            {"stage": "g0", "status": "passed", "auroc": g0["auroc"]},
            {"stage": "policies", "status": "evaluated"},
        ],
        gate_context,
    )
    raw = pd.DataFrame(raw_rows)
    raw.attrs["metric_tables"] = metric_tables
    raw.attrs["gate_table"] = gate_table
    raw.attrs["risk_tables"] = risk_tables
    raw.attrs["architecture_table"] = architecture
    return raw


def run_experiments(
    model_names: tuple[str, ...] = ("logistic_regression", "random_forest")
) -> pd.DataFrame:
    """Run configured seeds and export all Process 7 result artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    metric_tables: list[pd.DataFrame] = []
    gate_tables: list[pd.DataFrame] = []
    risk_tables: list[pd.DataFrame] = []
    architecture_tables: list[pd.DataFrame] = []
    design_rows = []
    for dataset_id in DATASETS:
        for model_name in model_names:
            design_rows.append(
                {
                    "dataset": dataset_id,
                    "model": model_name,
                    "explainer": "SHAP",
                    "split_design": "60/20/20",
                    "threshold_source": "validation",
                    "seed": -1,
                    "split": "frozen_design",
                    "policy": "all",
                    "shift_condition": "none",
                    "coverage": None,
                    "denominator": None,
                    "threshold_id": "design",
                    "config_id": "process7_v1",
                }
            )
            for seed in CONFIG["seeds"]:
                frame = run_one_seed(seed, model_name, dataset_id)
                if frame.empty:
                    continue
                frames.append(frame)
                metric_tables.extend(frame.attrs.get("metric_tables", []))
                gate_tables.append(frame.attrs["gate_table"])
                risk_tables.extend(frame.attrs.get("risk_tables", []))
                architecture_tables.append(frame.attrs["architecture_table"])
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    export_table(raw, OUTPUT_DIR / "raw_case_level.csv")
    export_table(table1_design(design_rows), OUTPUT_DIR / "table1_design.csv")
    export_table(
        pd.concat(metric_tables, ignore_index=True) if metric_tables else pd.DataFrame(),
        OUTPUT_DIR / "table2_metrics.csv",
    )
    export_table(
        pd.concat(gate_tables, ignore_index=True) if gate_tables else pd.DataFrame(),
        OUTPUT_DIR / "table3_gate_ablation.csv",
    )
    export_table(
        pd.concat(risk_tables, ignore_index=True) if risk_tables else pd.DataFrame(),
        OUTPUT_DIR / "risk_coverage_data.csv",
    )
    export_table(
        pd.concat(architecture_tables, ignore_index=True) if architecture_tables else pd.DataFrame(),
        OUTPUT_DIR / "architecture_decision_data.csv",
    )
    return raw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_experiments()
