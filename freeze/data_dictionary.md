# Data Dictionary

| Dataset | Source/loader | Target | Features | Missingness/preprocessing |
|---|---|---|---:|---|
| synthetic | `ml/data/synthetic_dataset.py` | Binary `y` | 10 | No missing values by construction; training-only preprocessing |
| heart_disease | `ml/data/heart_disease_loader.py` via UCI Cleveland source | Raw `num` 0-4 mapped to binary | 13 | Source missing values in `ca`/`thal`; training-fitted imputation |
| wdbc | `ml/data/wdbc_loader.py` via scikit-learn packaged data | 0 malignant, 1 benign | 30 | No missing values; training-only preprocessing |

All datasets are loaded through `ml/data/unified_interface.py`. Dataset metadata,
limitations, source URLs, and label semantics are maintained in
`ml/data/dataset_registry.py` and `ml/data/dataset_manifest.yaml`.

The split protocol is implemented in `ml/data/split_protocol.py`. Row indices are
persisted in the freeze/smoke split manifests; train, validation, and test indices
must be disjoint and exhaustive for each dataset/seed.
