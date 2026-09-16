"""Shared configuration for the EFTA healthcare XAI experiments."""

CONFIG = {
    "target_coverage": 0.70,
    "coverage_tolerance": 0.05,
    "seeds": list(range(40, 50)),
    "top_k_features": 3,
    "perturbation_noise": 0.05,
    "num_perturbations": 20,
    "confidence_threshold": 0.80,
    "faithfulness_drop_threshold": 0.20,
    "stability_threshold": 0.70,
    "random_forest_trees": 300,
    "logistic_regression_C": 1.0,
    "logistic_regression_max_iter": 2000,
    "model_random_state": 0,
    "random_forest_imbalance_ratio_threshold": 1.5,
    # Placeholder: confirm with the supervisor before confirmatory runs.
    "g0_min_auroc": 0.60,
    # Placeholder: confirm with the supervisor before confirmatory runs.
    "g4_degradation_threshold": 0.05,
}

if __name__ == "__main__":
    print(CONFIG)
