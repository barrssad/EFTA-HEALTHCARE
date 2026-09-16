# EFTA Scientific Freeze

## Status

This is the Part 10 protocol freeze for:

> Evaluation Before Trust: A Non-Compensatory Gate Architecture for Healthcare XAI Under Distribution Shift

The final ten-seed confirmatory experiment has **not** been run. The artifacts in `ml/results/smoke/` are pilot-only smoke artifacts and must not be reported as confirmatory research results.

## Frozen design

The study uses the existing three datasets (`synthetic`, `heart_disease`, and `wdbc`), Logistic Regression and Random Forest, the existing B0/B1/B2/EFTA policies, and the existing G0-G5 definitions. The prespecified seeds are 40 through 49. Every seed uses a stratified 60/20/20 train/validation/test split. Preprocessing is fitted on training data only.

Target coverage is 0.70 with tolerance 0.05. Threshold selection is validation-only and is applied unchanged to the held-out test partition. The top-k explanation rule, faithfulness definition, stability definition, shift types and levels, and Part 8 paired-seed statistical analysis are frozen as implemented in the repository.

## Confirmatory hypotheses

- H1: At matched coverage, EFTA has lower accepted-case selective error than B1 confidence-only release.
- H2: At matched coverage, EFTA has lower unsupported explanation release than B2 weighted-index release.
- H3: EFTA has smaller clean-to-shifted deterioration in accepted-case error than B1 confidence-only release.

No seed, model, shift, threshold, or policy may be selected based on held-out test outcomes. Datasets, models, seeds, thresholds, and hypotheses are not to be changed after this freeze without versioning the protocol and rerunning the affected grid.

## Analysis unit

The repeated evaluation unit is dataset x model x scenario x policy x prespecified seed. Competing policies share the same split within a seed. Primary comparisons use seed-level paired observations. Undefined metrics remain undefined and are not replaced by zero.

## Artifact policy

Raw case-level outputs, split manifests, environment records, source/configuration hashes, and generated tables must be retained. Smoke artifacts are labelled `execution_mode: smoke`; they are feasibility and determinism checks only.
