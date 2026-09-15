## Datasets

Three datasets back the EFTA research pipeline, each behind one unified
loader interface (`ml/data/unified_interface.py: load_dataset(dataset_id)`),
so downstream splitting, model training, and gate evaluation code is
dataset-agnostic.

| Dataset | Role | Source | Samples | Features | Target | Limitation |
|---|---|---|---|---|---|---|
| Synthetic known-mechanism data | Construct-validity test | Generated in-repo (`ml/data/synthetic_dataset.py`), fixed seed | 900 | 10 (3 causal, 7 noise) | Binary, nonlinear mechanism | Not clinical data; validates explanation recovery only |
| UCI Heart Disease (Cleveland) | Healthcare benchmark 1 | [Official UCI source, id=45](https://archive.ics.uci.edu/dataset/45/heart+disease), Cleveland records only | 303 | 13 | Binary (raw `num` 0–4 mapped to 0/1) | Small, 1980s, single-site, known data-quality issues |
| WDBC | Healthcare benchmark 2 | `sklearn.datasets.load_breast_cancer` ([origin](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)) | 569 | 30 | Binary (0 = malignant, 1 = benign) | Diagnostic feature benchmark, not a deployment study; no demographic fields for a real subgroup audit |

None of these datasets represent real-world clinical deployment data or
support clinical-utility claims.

Full metadata for every dataset (license, exact source, exclusions,
missingness, preprocessing requirements) is recorded in
`ml/data/dataset_registry.py` and mirrored in
`ml/data/dataset_manifest.yaml`. Any value that could not be verified is
marked `unknown/TODO` rather than guessed.

### Reproducing the dataset audit

```bash
python -m ml.data.audit
```

Prints a JSON report per dataset: shape, feature names, target
distribution, missing values, duplicate rows, constant features, class
imbalance, and split sizes — computed from the actual loaded data, not
copied from the registry.

### Running the dataset tests

```bash
pytest ml/tests/test_datasets.py -v
```

Heart Disease tests are automatically skipped (not failed) if
`archive.ics.uci.edu` is unreachable and no local cache exists at
`ml/data/cache/heart_disease_cleveland_raw.csv`. See the fallback
instructions at the top of `ml/data/heart_disease_loader.py` to populate
the cache manually if needed.
