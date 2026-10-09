# Machine Learning Pipeline Technical Audit

Audit date: 2026-10-10
Branch: audit-ml-quality-2026-10

## Scope

Source-level inspection of `README.md`, `requirements.txt`, `src/pipeline.py`, `src/regression.py`, `src/prepare.py`, and `src/features.py`. No training run, dependency installation, or model performance verification has been completed.

## Findings

| Severity | Location | Evidence | Recommendation |
| --- | --- | --- | --- |
| Medium | `src/regression.py`, `train_regression` | Candidate models are selected through cross-validation after one shared preparation step; verify all learned preprocessing is inside each CV fold rather than fitted on the full training set before CV. | Confirmed: `prepare_data()` fits transformations before `cross_val_score()` in `train_regression()`. Refactor cross-validation to fit transformations inside each fold and add leakage regression tests before claiming unbiased CV. Hold-out evaluation remains separate. |
| Medium | `src/prepare.py`, `prepare_data` | Classification imbalance logic computes `counts.min() / counts.max()` without an explicit guard for an empty label distribution. | Reject entirely missing labels and explicitly handle single-class labels with focused tests. Missing-label validation fixed on branch; test added but not run. |
| Medium | `requirements.txt` | Broad minimum-only package constraints across scikit-learn, pandas, Great Expectations, and MLflow do not guarantee that a fresh install reproduces the same environment. | Produce a tested lock or constraints file after a successful environment build. |
| Low | `src/pipeline.py`, `src/regression.py`, `src/prepare.py`, `src/features.py` | Long historical changelog docstrings, decorative log symbols and instructional comments obscure runtime behavior. | Remove nonessential narration in small behavior-preserving commits. |
| Informational | `README.md` | Report acknowledges that historical performance values and SHAP features are not current validated results. | Retain this honest distinction until a reproducible training and evaluation run is available. |

## Verification requirements

1. Establish a reproducible test environment and inventory existing tests.
2. Verify data splitting, feature engineering, training and inference with synthetic datasets.
3. Test preprocessing inside cross-validation folds.
4. Verify saved artifacts work with API inference.
5. Run CI before merge. No merge or production deployment should occur without verification.
