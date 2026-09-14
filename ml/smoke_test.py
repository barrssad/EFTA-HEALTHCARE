from experiments import run_one_seed

rows = run_one_seed(40, "logistic_regression")
assert len(rows) == 5
assert all(0.0 <= row["coverage"] <= 1.0 for row in rows)
print("SMOKE_TEST_OK", len(rows), rows[0]["condition"])
