from __future__ import annotations

import json

import pandas as pd

from ml.reproducibility import (
    CheckpointStore,
    build_run_manifest,
    environment_snapshot,
    sha256_file,
    write_checksum_manifest,
)


def test_environment_and_manifest_capture_protocol_metadata() -> None:
    environment = environment_snapshot(execution_mode="smoke", seeds=[40, 41])
    manifest = build_run_manifest(
        dataset_ids=["wdbc"],
        model_names=["logistic_regression"],
        seeds=[40, 41],
        execution_mode="smoke",
        output_paths=["results/smoke/raw_results_smoke.csv"],
        environment=environment,
    )

    assert environment["execution_mode"] == "smoke"
    assert environment["configured_seeds"] == [40, 41]
    assert manifest["split_ratios"] == [0.6, 0.2, 0.2]
    assert manifest["target_coverage"] == 0.70
    assert manifest["coverage_tolerance"] == 0.05
    assert len(manifest["configuration_hash"]) == 64


def test_checkpoint_skips_only_verified_completed_units(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints", "config-hash")
    frame = pd.DataFrame({"seed": [40], "value": [1.0]})
    store.save_completed("wdbc", "logistic_regression", 40, frame)

    loaded = store.load_verified("wdbc", "logistic_regression", 40)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, frame)

    data_path = tmp_path / "checkpoints" / "wdbc__logistic_regression__seed_40.pkl"
    data_path.write_bytes(b"corrupt")
    assert store.load_verified("wdbc", "logistic_regression", 40) is None


def test_checksum_manifest_is_sha256_and_machine_readable(tmp_path) -> None:
    artifact = tmp_path / "raw.csv"
    artifact.write_text("seed,value\n40,1\n", encoding="utf-8")
    checksum_path = write_checksum_manifest([artifact], tmp_path / "checksums.sha256")

    line = checksum_path.read_text(encoding="utf-8").strip()
    digest, path = line.split("  ", maxsplit=1)
    assert digest == sha256_file(artifact)
    assert path == str(artifact)
    assert json.loads(json.dumps({"execution_mode": "smoke"}))["execution_mode"] == "smoke"
