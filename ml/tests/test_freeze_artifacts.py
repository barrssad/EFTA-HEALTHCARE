from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "freeze"
SMOKE = ROOT / "ml" / "results" / "smoke"


def test_part10_freeze_artifacts_exist() -> None:
    required = [
        FREEZE / "README.md",
        FREEZE / "config.yaml",
        FREEZE / "preregistration.md",
        FREEZE / "environment.txt",
        FREEZE / "data_dictionary.md",
        FREEZE / "data_manifest.csv",
        FREEZE / "source_manifest.csv",
        FREEZE / "ai_use_log.md",
        FREEZE / "checksums.sha256",
        FREEZE / "results_raw" / "README.md",
        FREEZE / "results_tables" / "README.md",
        FREEZE / "figures" / "README.md",
        FREEZE / "manuscript" / "README.md",
        FREEZE / "splits" / "README.md",
    ]
    assert all(path.is_file() for path in required)


def test_frozen_smoke_artifacts_are_pilot_only_and_traceable() -> None:
    with (SMOKE / "raw_results_smoke.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["execution_mode"] for row in rows} == {"smoke"}
    assert {int(row["seed"]) for row in rows} == {40, 41}
    keys = [
        (row["dataset"], row["model"], row["seed"], row["shift_condition"], row["policy"], row["case_index"])
        for row in rows
    ]
    assert len(keys) == len(set(keys))

    manifest = json.loads((SMOKE / "run_manifest_smoke.json").read_text(encoding="utf-8"))
    assert manifest["execution_mode"] == "smoke"
    assert manifest["seed_list"] == [40, 41]


def test_frozen_split_manifests_are_disjoint() -> None:
    for path in sorted((SMOKE / "splits").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        train = set(manifest["train_indices"])
        validation = set(manifest["validation_indices"])
        test = set(manifest["test_indices"])
        assert manifest["execution_mode"] == "smoke"
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert train | validation | test


def test_freeze_checksum_manifest_covers_protocol_inputs() -> None:
    checksum_text = (FREEZE / "checksums.sha256").read_text(encoding="utf-8").replace("\\", "/")
    for relative_path in (
        "freeze/config.yaml",
        "freeze/data_manifest.csv",
        "ml/config.py",
        "ml/data/dataset_manifest.yaml",
        "ml/results/smoke/raw_results_smoke.csv",
    ):
        assert relative_path in checksum_text
