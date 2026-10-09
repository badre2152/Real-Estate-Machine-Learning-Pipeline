# Portfolio Evidence and Results Checklist

This page separates code capabilities from empirical evidence. The repository is a bootcamp project centered on Moroccan real estate price regression.

## What is verifiable from the source

- Data extraction from PostgreSQL or a compatible Parquet file
- Cleaning, train and test partitioning, feature engineering, and training-only preprocessing
- Baseline comparisons and regression candidates: Ridge, Random Forest, Gradient Boosting, and optional XGBoost
- Regression metrics including MAE, RMSE, R², and MAPE
- SHAP explainer and regression visualization code
- Optional classification and serving infrastructure

Implementation does not imply successful execution on the latest code revision.

## Evidence available today

| Evidence | Location | Limitation |
| --- | --- | --- |
| Historical model report | [HTML report](../reports/ml_report.html) | Older feature schema; not a result for the revised pipeline |
| Historical data validation | [JSON](../reports/validation_report.json) | 489 records, validation passed = false |
| Historical monitoring | [JSON](../reports/monitoring_report.json) | Shows a past run, not the latest revision |
| Historical SHAP ranking | [CSV](plots/reg_shap_importance.csv) | Contains price-derived features and listing URLs no longer used as predictors |

The HTML report records a historical R² of **0.7691**, MAE of **314,089.5592 MAD**, and MAPE of **22.4777%**. Do not put these values in a resume or portfolio as verified current performance.

## Evidence needed for the final portfolio showcase

| Deliverable | What to record | Status |
| --- | --- | --- |
| Data quality summary | Dataset snapshot/date, row counts, missing values, invalid prices, quality decisions | Pending a new run |
| Experiment setup | Split size, random seed, preprocessing inputs, training configuration | Pending a new run |
| Baseline comparison | Baseline MAE and RMSE alongside candidate models | Pending a new run |
| Final model metrics | MAE, RMSE, R², MAPE on untouched test data | Pending a new run |
| Error visualization | Predicted versus actual, residual distribution, error by price range | Pending a new run |
| Explainability | Fresh SHAP summary from current leakage-safe features | Pending a new run |
| Reproducibility | Exact commit, dependency versions, data provenance, generated artifacts | Pending a new run |

## Suggested portfolio visual story

1. **Data flow:** PostgreSQL to cleaning to features to regression and evaluation.
2. **Model comparison:** A table with real held-out regression metrics and an explicit baseline.
3. **Prediction quality:** Predicted versus actual prices and residual distribution.
4. **Interpretability:** SHAP global importance for the selected model using current features.
5. **Lessons learned:** Acknowledge the discovery and removal of price-derived leakage features; document limitations in data quality.

Do not use the historical SHAP importance CSV as a current-model interpretation. The archived validation failures also prevent claims that all records passed quality checks.

## Recommended portfolio wording

> Developed a modular real estate price prediction pipeline as a bootcamp project using Python, PostgreSQL, pandas, scikit-learn, and XGBoost. Implemented data preparation, feature engineering, leakage-aware feature selection, regression model comparisons, evaluation, and SHAP explainability.

Avoid quantitative claims until results from the revised pipeline have been generated and independently inspected.

**Audit scope:** Documentation and source review only. No tests, CI, training, or deployment were executed.
