from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))
from ml.experiments import run_one_seed

cases = run_one_seed(40, "logistic_regression", "wdbc")
assert not cases.empty
assert {"g1", "g2", "g3", "decision", "policy"}.issubset(cases.columns)
assert "MODEL_BLOCKED" in Path(__file__).with_name("experiments.py").read_text(encoding="utf-8")
assert cases["dataset"].eq("wdbc").all()
print("SMOKE_TEST_OK", len(cases), cases["shift_condition"].nunique())
