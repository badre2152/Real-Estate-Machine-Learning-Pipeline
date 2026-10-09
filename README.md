# Real Estate Price Prediction | Morocco

An end to end machine learning project for estimating residential property prices in Morocco using Python, PostgreSQL, feature engineering, and regression models.

**Project type:** Bootcamp portfolio project  
**Author:** Brahim Badre | Data Analyst / Data Engineer  
**Related ETL project:** [Real Estate Data Pipeline](https://github.com/badre2152/real-estate-pipeline)

## Project objective

Turn structured property records into reproducible price estimates in Moroccan dirhams (MAD). The focus is the data to model workflow: extracting data, checking quality, preparing meaningful predictors, comparing regression algorithms, and explaining results.

This repository also contains optional classification, API, Docker, DVC, and MLflow components. They support the core work but are not prerequisites for understanding the regression project.

## Architecture

```text
Real estate records in PostgreSQL
               |
          Data extraction
               |
       Data quality checks
               |
         Train / test split
               |
   Calibration holdout (quantile mode)
               |
         Feature engineering
               |
   Train fitted preprocessing only
               |
      Regression model selection
               |
       MAE / RMSE / R² / MAPE
               |
   Feature importance / SHAP analysis
               |
       Saved model and reports
```

Core modules: `src/extract.py`, `src/prepare.py`, `src/features.py`, `src/regression.py`, `src/pipeline.py`, and `src/shap_explainer.py`.

## Data and feature engineering

The regression target is `prix`, representing property price in MAD. Current serving compatible predictors are selected from:

| Feature | Purpose |
| --- | --- |
| `surface_m2` | Property area |
| `ville`, `quartier` | Location categories |
| `nb_chambres`, `nb_salles_bain` | Room counts |
| `etage`, `age_bien` | Floor and property age |
| `surface_x_chambres` | Surface and bedroom interaction |
| `surface_par_chambre` | Area per bedroom adjustment |
| `ratio_chambres_bains` | Room ratio |

Feature selection depends on available columns in the prepared datasets. The preprocessing code excludes target-derived price features and geographic price aggregates from predictors. Encoders, imputers, and scalers are fitted on training data, not test or calibration data.

## Modeling approach

Regression candidates include **Ridge**, **Random Forest**, **Gradient Boosting**, and **XGBoost** when installed. The workflow also calculates dummy model baselines. Evaluation uses **MAE, RMSE, R², and MAPE**, with feature importance and SHAP analysis when available.

**Measured performance:** Not published here. The current repository does not include a verified, reproducible results summary from the latest training configuration. Numbers should not be inferred from historical sample reports.

Optional classification predicts property type internally and is secondary to price regression. Because property type is supplied to the API, the API returns the validated input type rather than a redundant classifier prediction.

## Reproducing the workflow

Requirements: Python environment, project dependencies, and access to the expected PostgreSQL data or an extracted compatible Parquet file.

```bash
git clone https://github.com/badre2152/Real-Estate-Machine-Learning-Pipeline.git
cd Real-Estate-Machine-Learning-Pipeline
pip install -r requirements.txt
cp .env.example .env
python src/pipeline.py --help
python src/pipeline.py
```

Configure the environment variables in `.env` before running. For an existing Parquet extract, use `--input-parquet`. The data is not included in this public repository.

The pipeline writes outputs to configured model and report directories, including `models/results.json`. To inspect empirical results, use artifacts generated from your own data and training run. DVC orchestration is described in [the DVC guide](docs/DVC_GUIDE.md) but has not been executed during this audit.

## Historical reports and evidence

The repository contains an older [HTML model report](reports/ml_report.html), [data validation report](reports/validation_report.json), [monitoring snapshot](reports/monitoring_report.json), and [SHAP importance CSV](docs/plots/reg_shap_importance.csv). These are historical artifacts, not validated results for the current training pipeline.

The historical HTML report records **R² = 0.7691**, **MAE = 314,089.5592 MAD**, and **MAPE = 22.4777%**. These values must not be presented as the current model's performance. The archived validation snapshot dated May 17, 2026 reports **489 rows** and **passed: false**, including failed price range and missing value checks. Its results cannot be used to claim a fully validated dataset.

Critically, the historical SHAP file ranks `ecart_prix_ville`, `ville_prix_median`, and `ville_rang_prix` ahead of `surface_m2`, and includes `prix_par_m2` and individual listing URLs. These inputs are not part of the current serving compatible training feature selection. The archived importance ranking therefore reflects an older feature schema with leakage risks, not trustworthy interpretability evidence for the corrected regression model.

A portfolio-ready results section needs freshly generated MAE, RMSE, R², baseline comparisons, and SHAP outputs from one consistent training run with documented data validation. Until then, the strongest verifiable claim is that this repository implements the end to end workflow, not that a particular predictive performance was achieved.

## Project deliverables

- A modular data preparation and regression workflow
- Comparison against baseline models
- Data validation and feature engineering
- Model evaluation metrics and explainability components
- Optional FastAPI inference service and infrastructure configuration

**Reproducibility limitation:** Saved model artifacts are not bundled in this repository. The API and deployment configuration need newly generated compatible models, preprocessors, and interval artifacts before providing predictions. No tests, CI, training, or deployment were run as part of the code review.

## Technical documentation

- [DVC workflow](docs/DVC_GUIDE.md)
- [Deployment notes](docs/DEPLOYMENT_GUIDE.md)
- [MLflow registry](docs/MLFLOW_REGISTRY_GUIDE.md)
- [Contributing](CONTRIBUTING.md)

## Tools and skills demonstrated

**Python, pandas, SQL, PostgreSQL, scikit-learn, XGBoost, data cleaning, feature engineering, machine learning evaluation, and model explainability.**

This is a learning and portfolio project, not a production valuation service.
