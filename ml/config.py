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
}

if __name__ == "__main__":
    print(CONFIG)
