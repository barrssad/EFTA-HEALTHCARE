"""Two-seed, artifact-producing smoke run for local and Colab validation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from ml.config import CONFIG
from ml.experiments import run_one_seed
from ml.reproducibility import (
	CheckpointStore,
	build_run_manifest,
	environment_snapshot,
	input_artifact_paths,
	save_json,
	write_checksum_manifest,
)
from ml.result_tables import export_table


def run_smoke() -> Path:
	"""Run only the first two prespecified seeds and export smoke artifacts."""
	seeds = tuple(int(seed) for seed in CONFIG["seeds"][:2])
	output_dir = ROOT / "ml" / "results" / "smoke"
	checkpoint_dir = output_dir / "checkpoints"
	raw_path = output_dir / "raw_results_smoke.csv"
	environment_path = output_dir / "environment_smoke.json"
	manifest_path = output_dir / "run_manifest_smoke.json"

	environment = environment_snapshot(execution_mode="smoke", seeds=seeds)
	manifest = build_run_manifest(
		dataset_ids=("wdbc",),
		model_names=("logistic_regression",),
		seeds=seeds,
		execution_mode="smoke",
		output_paths=(raw_path, checkpoint_dir),
		environment=environment,
	)
	save_json(environment, environment_path)
	save_json(manifest, manifest_path)
	store = CheckpointStore(
		checkpoint_dir,
		manifest["configuration_hash"],
		execution_mode="smoke",
	)

	frames = []
	for seed in seeds:
		frame = store.load_verified("wdbc", "logistic_regression", seed)
		if frame is None:
			frame = run_one_seed(
				seed,
				"logistic_regression",
				"wdbc",
				split_manifest_dir=output_dir / "splits",
				execution_mode="smoke",
			)
			if frame.empty:
				raise AssertionError(f"Smoke run produced no rows for seed {seed}.")
			store.save_completed("wdbc", "logistic_regression", seed, frame)
		assert {"g1", "g2", "g3", "decision", "policy"}.issubset(frame.columns)
		assert frame["dataset"].eq("wdbc").all()
		frames.append(frame)

	export_frames = []
	for frame in frames:
		export_frame = frame.copy()
		export_frame.attrs = {}
		export_frame["execution_mode"] = "smoke"
		export_frames.append(export_frame)
	raw = __import__("pandas").concat(export_frames, ignore_index=True)
	export_table(raw, raw_path)

	repeat = run_one_seed(seeds[0], "logistic_regression", "wdbc")
	repeat_export = repeat.copy()
	repeat_export["execution_mode"] = "smoke"
	if raw[raw["seed"] == seeds[0]].to_csv(index=False) != repeat_export.to_csv(index=False):
		raise AssertionError("Smoke rerun was not deterministic for the first seed.")

	checksum_path = output_dir / "checksums_smoke.sha256"
	write_checksum_manifest(
		[
			environment_path,
			manifest_path,
			raw_path,
			*checkpoint_dir.glob("*"),
			*((output_dir / "splits").glob("*")),
			*input_artifact_paths(),
		],
		checksum_path,
	)
	return raw_path


if __name__ == "__main__":
	result_path = run_smoke()
	print(f"SMOKE_TEST_OK execution_mode: smoke output: {result_path}")
