"""Additive reproducibility and artifact-management utilities for EFTA runs."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "results" / "artifacts"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def hash_files(paths: Iterable[str | Path]) -> dict[str, str]:
    return {str(Path(path)): sha256_file(path) for path in sorted(paths, key=lambda item: str(item)) if Path(path).is_file()}


def _package_version(package: str, import_name: str | None = None) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        if import_name is None:
            return None
        try:
            return str(getattr(importlib.import_module(import_name), "__version__", "unknown"))
        except (ImportError, AttributeError):
            return None


def _git_commit_hash(root: Path = PROJECT_ROOT) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def environment_snapshot(
    *,
    execution_mode: str,
    seeds: Iterable[int],
    timestamp: str | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the execution environment."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": _package_version("numpy", "numpy"),
        "pandas_version": _package_version("pandas", "pandas"),
        "scikit_learn_version": _package_version("scikit-learn", "sklearn"),
        "scipy_version": _package_version("scipy", "scipy"),
        "matplotlib_version": _package_version("matplotlib", "matplotlib"),
        "seaborn_version": _package_version("seaborn", "seaborn"),
        "shap_version": _package_version("shap", "shap"),
        "repository_commit": _git_commit_hash(root),
        "timestamp_utc": timestamp or datetime.now(timezone.utc).isoformat(),
        "execution_mode": execution_mode,
        "configured_seeds": [int(seed) for seed in seeds],
    }


def save_json(value: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return destination


def build_run_manifest(
    *,
    dataset_ids: Iterable[str],
    model_names: Iterable[str],
    seeds: Iterable[int],
    execution_mode: str,
    output_paths: Iterable[str | Path],
    environment: dict[str, Any] | None = None,
    split_ratios: tuple[float, float, float] = (0.60, 0.20, 0.20),
    shift_configuration: dict[str, Any] | None = None,
    statistical_configuration: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_value = dict(CONFIG if config is None else config)
    manifest = {
        "execution_mode": execution_mode,
        "dataset_ids": list(dataset_ids),
        "model_names": list(model_names),
        "seed_list": [int(seed) for seed in seeds],
        "split_ratios": list(split_ratios),
        "target_coverage": config_value["target_coverage"],
        "coverage_tolerance": config_value["coverage_tolerance"],
        "shift_configuration": shift_configuration or {
            "measurement_noise": [0.1, 0.2],
            "missingness": [0.1, 0.2],
            "prevalence_shift": [0.70],
            "subgroup": ["age_if_available"],
        },
        "statistical_configuration": statistical_configuration or {
            "unit": "paired_seed",
            "primary_comparisons": ["EFTA_vs_B1", "EFTA_vs_B2"],
            "secondary_fdr": "benjamini_hochberg",
        },
        "configuration": config_value,
        "configuration_hash": sha256_json(config_value),
        "output_paths": [str(Path(path)) for path in output_paths],
        "environment": environment or environment_snapshot(execution_mode=execution_mode, seeds=seeds),
    }
    return manifest


def write_checksum_manifest(paths: Iterable[str | Path], destination: str | Path) -> Path:
    checksums = hash_files(paths)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(checksums.items()))
        ,
        encoding="utf-8",
    )
    return output


def save_split_manifest(
    *,
    dataset: str,
    seed: int,
    train_indices: Iterable[int],
    val_indices: Iterable[int],
    test_indices: Iterable[int],
    destination: str | Path,
    configuration_hash: str | None = None,
    execution_mode: str = "full",
    run_id: str | None = None,
    repair_version: str | None = None,
) -> Path:
    """Persist row membership for one deterministic split without raw data."""
    manifest = {
        "dataset": dataset,
        "seed": int(seed),
        "split_ratios": [0.60, 0.20, 0.20],
        "train_indices": [int(index) for index in train_indices],
        "validation_indices": [int(index) for index in val_indices],
        "test_indices": [int(index) for index in test_indices],
        "configuration_hash": configuration_hash,
        "execution_mode": execution_mode,
        "run_id": run_id or execution_mode,
        "repair_version": repair_version,
    }
    manifest["split_hash"] = sha256_json(
        {
            key: manifest[key]
            for key in ("dataset", "seed", "train_indices", "validation_indices", "test_indices")
        }
    )
    return save_json(manifest, destination)


def input_artifact_paths(root: Path = PROJECT_ROOT) -> list[Path]:
    """Return repository inputs that can be hashed without downloading data."""
    candidates = [root / "ml" / "config.py", root / "ml" / "data" / "dataset_manifest.yaml"]
    cache_dir = root / "ml" / "data" / "cache"
    if cache_dir.is_dir():
        candidates.extend(cache_dir.iterdir())
    return [path for path in candidates if path.is_file()]


class CheckpointStore:
    """Verified per-dataset/model/seed checkpoints with atomic writes."""

    def __init__(
        self,
        directory: str | Path,
        configuration_hash: str,
        execution_mode: str = "full",
        run_id: str | None = None,
        repair_version: str | None = None,
    ):
        self.directory = Path(directory)
        self.configuration_hash = configuration_hash
        self.execution_mode = execution_mode
        self.run_id = run_id
        self.repair_version = repair_version

    def _identity(self, dataset: str, model: str, seed: int) -> dict[str, Any]:
        return {"dataset": dataset, "model": model, "seed": int(seed)}

    def _paths(self, dataset: str, model: str, seed: int) -> tuple[Path, Path]:
        stem = f"{dataset}__{model}__seed_{int(seed)}"
        return self.directory / f"{stem}.pkl", self.directory / f"{stem}.json"

    def load_verified(self, dataset: str, model: str, seed: int):
        import pandas as pd

        data_path, metadata_path = self._paths(dataset, model, seed)
        if not data_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            identity = self._identity(dataset, model, seed)
            if metadata.get("identity") != identity or metadata.get("status") != "completed":
                return None
            if metadata.get("configuration_hash") != self.configuration_hash:
                return None
            if metadata.get("execution_mode") != self.execution_mode:
                return None
            expected_run_id = self.run_id or self.execution_mode
            if metadata.get("run_id") != expected_run_id:
                return None
            if metadata.get("repair_version") != self.repair_version:
                return None
            if metadata.get("data_sha256") != sha256_file(data_path):
                return None
            frame = pd.read_pickle(data_path)
        except (OSError, ValueError, json.JSONDecodeError, ImportError):
            return None
        return frame

    def save_completed(self, dataset: str, model: str, seed: int, frame) -> Path:
        data_path, metadata_path = self._paths(dataset, model, seed)
        existing = self.load_verified(dataset, model, seed)
        if existing is not None:
            raise FileExistsError(f"Verified checkpoint already exists: {data_path}")
        if data_path.exists() or metadata_path.exists():
            raise FileExistsError(f"Unverified checkpoint already exists: {data_path}")
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = data_path.with_suffix(".tmp.pkl")
        frame.to_pickle(temporary)
        temporary.replace(data_path)
        metadata = {
            "identity": self._identity(dataset, model, seed),
            "scenario": "all_conditions",
            "policy": "all_policies",
            "status": "completed",
            "output_path": str(data_path),
            "configuration_hash": self.configuration_hash,
            "execution_mode": self.execution_mode,
            "run_id": self.run_id or self.execution_mode,
            "repair_version": self.repair_version,
            "data_sha256": sha256_file(data_path),
        }
        save_json(metadata, metadata_path)
        return data_path
