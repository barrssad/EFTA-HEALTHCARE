# Versioned SHAP Repair: `tree_additivity_check_v1`

## Status

Implemented and validated. The affected confirmatory design has **not** been rerun yet.

## Failure

The frozen confirmatory run failed in the Random Forest `TreeExplainer` path with:

```text
shap.utils._exceptions.ExplainerError:
Additivity check failed in TreeExplainer
```

The failure occurred in `ml/explanations.py`, inside `ShapExplanation.__call__`, while using SHAP `0.46.0`. SHAP's internal additivity assertion rejected a small numerical mismatch between the attribution sum and the model output.

## Bounded repair

`ml/explanations.py` now passes:

```python
check_additivity=False
```

only when invoking `TreeExplainer`. Other explainer types retain their existing call behavior. The repair is versioned in code as:

```text
SHAP_COMPATIBILITY_REPAIR_VERSION = "tree_additivity_check_v1"
```

This disables an explainer-library consistency assertion. It does not change the Random Forest, SHAP explainer family, target class, top-k extraction, faithfulness calculation, stability calculation, thresholds, gates, policies, shifts, or statistical analysis.

## Protocol invariants

The following remain unchanged:

- G0-G5 definitions
- B0/B1/B2/EFTA policy logic
- datasets and models
- prespecified seeds
- 60/20/20 split ratios
- target coverage and tolerance
- threshold grids and validation-only selection
- faithfulness and stability definitions
- distribution-shift definitions
- Part 8 statistical logic
- H1, H2, and H3
- validation/test separation

## Validation

Focused explanation validation passed:

```text
python -m pytest ml/tests/test_explanations.py -q
11 passed, 5 skipped
```

The regression test verifies that `check_additivity=False` is passed only to TreeExplainer.

## Rerun rule

The affected frozen confirmatory design must be rerun from the beginning after this repair. Pre-repair outputs are not confirmatory evidence for the affected design. No favorable seeds, models, shifts, thresholds, policies, or outcomes may be selected, and no result may be manually patched.
