"""Centralized, hand-verified dataset metadata registry.

Every field is either verified from an authoritative source or explicitly
marked "unknown/TODO". Nothing here is computed at runtime — runtime facts
(actual sample counts, missingness, etc.) belong in ``audit.py``, which
checks these declared values against the real loaded data and raises if
they disagree, so this file cannot silently drift from reality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    name: str
    role: str
    source: str
    source_url: str
    license: str
    version_or_revision: str
    sample_count_before_exclusions: int | str
    feature_count: int
    feature_names: list[str]
    feature_definitions: dict[str, str]
    target_name: str
    target_encoding: dict[int, str]
    class_names: list[str]
    class_distribution: dict[str, int | str]
    missingness_summary: str
    exclusions_performed: str
    preprocessing_requirements: str
    dataset_limitation: str
    reproducibility_notes: str
    ground_truth_relevant_features: list[str] | None = field(default=None)


DATASET_REGISTRY: dict[str, DatasetMetadata] = {
    "synthetic": DatasetMetadata(
        dataset_id="synthetic",
        name="Synthetic Known-Mechanism Tabular Data",
        role="construct-validity test",
        source="Generated in-repo (ml/data/synthetic_dataset.py)",
        source_url="n/a (locally generated, no external source)",
        license="n/a (synthetic, no license required)",
        version_or_revision="generator v1, fixed seed",
        sample_count_before_exclusions=900,
        feature_count=10,
        feature_names=[f"feature_{i}" for i in range(10)],
        feature_definitions={
            "feature_0": "true causal feature (linear term)",
            "feature_1": "true causal feature (linear term)",
            "feature_2": "true causal feature (interaction term with feature_0)",
            "feature_3..9": "Gaussian noise, independent of label",
        },
        target_name="y",
        target_encoding={0: "negative_class", 1: "positive_class"},
        class_names=["negative_class", "positive_class"],
        class_distribution={"unknown_until_generated": "computed at runtime by audit.py"},
        missingness_summary="none by construction (fully synthetic, no missing values injected)",
        exclusions_performed="none",
        preprocessing_requirements="StandardScaler (all features numeric, no categoricals)",
        dataset_limitation="Not clinical data. Validates explanation-recovery methodology only; "
        "results here do not generalize to real healthcare data.",
        reproducibility_notes="Deterministic given (n_samples, seed); see generate_synthetic_dataset().",
        ground_truth_relevant_features=["feature_0", "feature_1", "feature_2"],
    ),
    "heart_disease": DatasetMetadata(
        dataset_id="heart_disease",
        name="UCI Heart Disease (Cleveland subset)",
        role="healthcare benchmark 1",
        source="UCI Machine Learning Repository, official source (via ucimlrepo package, "
        "id=45), Cleveland Clinic Foundation processed subset only. "
        "Hungary/Switzerland/VA Long Beach records are NOT included/mixed.",
        source_url="https://archive.ics.uci.edu/dataset/45/heart+disease",
        license="unknown/TODO — verify current license terms directly on the UCI dataset page "
        "before publication; do not assume CC BY without checking.",
        version_or_revision="unknown/TODO — ucimlrepo does not expose a dataset revision/version "
        "string at fetch time; record the fetch date manually when you run the loader.",
        sample_count_before_exclusions=303,
        feature_count=13,
        feature_names=[
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ],
        feature_definitions={
            "age": "age in years",
            "sex": "1 = male, 0 = female",
            "cp": "chest pain type (4 values)",
            "trestbps": "resting blood pressure (mm Hg)",
            "chol": "serum cholesterol (mg/dl)",
            "fbs": "fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
            "restecg": "resting electrocardiographic results (0-2)",
            "thalach": "maximum heart rate achieved",
            "exang": "exercise-induced angina (1 = yes, 0 = no)",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "slope": "slope of the peak exercise ST segment",
            "ca": "number of major vessels (0-3) colored by fluoroscopy",
            "thal": "3 = normal, 6 = fixed defect, 7 = reversible defect",
        },
        target_name="num (raw) -> binary_target (constructed)",
        target_encoding={0: "no_disease", 1: "disease_present"},
        class_names=["no_disease", "disease_present"],
        class_distribution={"unknown_until_generated": "computed at runtime by audit.py"},
        missingness_summary="Known: 'ca' and 'thal' columns contain missing values encoded as '?' "
        "in the raw Cleveland file (commonly 4 and 2 rows respectively across published "
        "analyses). This loader does NOT hardcode that count — audit.py verifies it "
        "against the actually-loaded data every run.",
        exclusions_performed="Cleveland records only; Hungary/Switzerland/VA Long Beach excluded "
        "by construction because a different, more-complete subset was never merged in.",
        preprocessing_requirements="Median/most-frequent imputation for 'ca'/'thal' FIT ON TRAINING "
        "DATA ONLY inside the pipeline (never globally before splitting); StandardScaler "
        "for Logistic Regression; categorical-like integer-coded columns (cp, restecg, "
        "slope, thal) are treated as numeric per the standard Cleveland benchmark setup, "
        "not one-hot encoded, to match the common published formulation.",
        dataset_limitation="Small (303 instances), collected in the late 1980s, single-site "
        "(Cleveland Clinic) after excluding other sites; known multi-site inconsistency "
        "issues across the combined UCI Heart Disease resource; not representative of "
        "modern clinical populations or measurement protocols.",
        reproducibility_notes="Raw target 'num' takes values 0-4 in the source file. Standard "
        "binary formulation used here: 0 stays 0 (no disease), any value 1-4 is mapped to "
        "1 (disease present). This conversion is explicit in heart_disease_loader.py, not "
        "silent.",
    ),
    "wdbc": DatasetMetadata(
        dataset_id="wdbc",
        name="Wisconsin Diagnostic Breast Cancer (WDBC)",
        role="healthcare benchmark 2",
        source="scikit-learn packaged dataset (sklearn.datasets.load_breast_cancer)",
        source_url="https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset "
        "(originally: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)",
        license="CC BY 4.0 (per scikit-learn dataset documentation)",
        version_or_revision="pinned to whatever scikit-learn version is installed; record "
        "sklearn.__version__ at run time (see environment logging in experiments.py)",
        sample_count_before_exclusions=569,
        feature_count=30,
        feature_names="see sklearn.datasets.load_breast_cancer().feature_names (10 measurements "
        "x mean/error/worst = 30 columns)",
        feature_definitions={
            "note": "10 base measurements (radius, texture, perimeter, area, smoothness, "
            "compactness, concavity, concave points, symmetry, fractal_dimension), each "
            "reported as mean, standard error, and 'worst' (largest) value -> 30 features.",
        },
        target_name="target",
        target_encoding={0: "malignant", 1: "benign"},
        class_names=["malignant", "benign"],
        class_distribution={"malignant": 212, "benign": 357},
        missingness_summary="none (packaged dataset has no missing values)",
        exclusions_performed="none; no ID field is present in the sklearn-packaged version "
        "(the raw UCI CSV's ID column is already dropped by sklearn's loader)",
        preprocessing_requirements="StandardScaler fit on training data only",
        dataset_limitation="Diagnostic feature benchmark derived from digitized fine needle "
        "aspirate images; not a deployment study; no demographic/protected attributes "
        "available, so no real subgroup audit (G5) is possible on this dataset alone.",
        reproducibility_notes="Fully deterministic; sklearn ships the data directly, no download "
        "step, no version drift beyond the installed scikit-learn release.",
    ),
}


def get_metadata(dataset_id: str) -> DatasetMetadata:
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset_id {dataset_id!r}. Valid ids: {list(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[dataset_id]
