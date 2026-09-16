from __future__ import annotations

import inspect
import json
from pathlib import Path

from ml.experiments import OUTPUT_DIR, run_experiments


def test_confirmatory_output_isolation_is_explicit_without_running_experiment() -> None:
    parameters = inspect.signature(run_experiments).parameters

    assert "output_dir" in parameters
    assert parameters["output_dir"].default is None
    assert parameters["execution_mode"].default == "full"
    assert OUTPUT_DIR.name == "results"


def test_confirmatory_v2_prerun_identity_is_explicit() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "ml/results/confirmatory_v2/pre_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["run_id"] == "confirmatory_v2"
    assert manifest["execution_mode"] == "confirmatory"
    assert manifest["repair_version"] == "tree_additivity_check_v1"
    assert manifest["confirmatory_run_started"] is False
    assert manifest["isolation_contract"]["checkpoint_execution_mode"] == "confirmatory"
    assert manifest["isolation_contract"]["split_manifest_execution_mode"] == "confirmatory"
