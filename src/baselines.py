"""
baselines.py
------------
Baseline Models pour établir une référence minimale avant les vrais modèles.

Un bon modèle ML doit TOUJOURS battre ses baselines.
Si ce n'est pas le cas, le problème vient du modèle ou des données.

Baselines disponibles :
  - Régression  : DummyRegressor (mean, median, quantile)
  - Classification : DummyClassifier (most_frequent, stratified, uniform)
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
)

from logger_setup import get_logger

logger = get_logger(__name__)

def run_regression_baselines(
    X_train, y_train, X_test, y_test
) -> dict[str, dict]:
    """
    Entraîne et évalue plusieurs DummyRegressor comme baseline.

    Stratégies testées :
      - mean     : prédit toujours la moyenne du train
      - median   : prédit toujours la médiane du train
      - quantile : prédit le 75e percentile du train

    Returns:
        dict { strategy: {MAE, RMSE, R2, MAPE} }
    """
    logger.info("\n" + "=" * 50)
    logger.info("BASELINES RÉGRESSION")
    logger.info("=" * 50)

    strategies = {
        "dummy_mean"      : DummyRegressor(strategy="mean"),
        "dummy_median"    : DummyRegressor(strategy="median"),
        "dummy_quantile75": DummyRegressor(strategy="quantile", quantile=0.75),
    }

    results = {}
    for name, model in strategies.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true = np.array(y_test)

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)

        nonzero_mask = np.abs(y_true) > np.finfo(float).eps
        if nonzero_mask.sum() > 0:
            mape = np.mean(
                np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask])
                       / np.abs(y_true[nonzero_mask]))
            ) * 100
        else:
            mape = float("nan")

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
        logger.info(
            f"   {name:<20s} → R²={r2:+.4f} | MAE={mae:>12,.0f} | MAPE={mape:.1f}%"
        )

    return results

def compare_vs_regression_baseline(
    model_metrics: dict, baseline_results: dict
) -> None:
    """
    Compare les métriques du vrai modèle contre les baselines
    et loggue si le modèle bat (ou non) chaque baseline.
    """
    logger.info("\nComparaison modèle vs baselines (régression) :")
    model_r2  = model_metrics.get("R2", 0)
    model_mae = model_metrics.get("MAE", float("inf"))

    for name, bm in baseline_results.items():
        r2_gain  = model_r2 - bm["R2"]
        mae_gain = bm["MAE"] - model_mae
        status   = "PASS" if r2_gain > 0 else "FAIL"
        logger.info(
            f"   {status} vs {name:<20s} → ΔR²={r2_gain:+.4f} | ΔMAE={mae_gain:>+12,.0f}"
        )

    best_baseline_r2 = max(b["R2"] for b in baseline_results.values())
    if model_r2 <= best_baseline_r2:
        logger.warning(
            "   Le modèle ne bat PAS la meilleure baseline ! "
            "Revoir les features ou le pipeline."
        )
    else:
        logger.info(
            f"   Modèle dépasse la meilleure baseline de ΔR²={model_r2 - best_baseline_r2:+.4f}"
        )

def run_classification_baselines(
    X_train, y_train, X_test, y_test
) -> dict[str, dict]:
    """
    Entraîne et évalue plusieurs DummyClassifier comme baseline.

    Stratégies testées :
      - most_frequent : prédit toujours la classe majoritaire
      - stratified    : prédit selon la distribution des classes du train
      - uniform       : prédit de façon aléatoire uniforme

    Returns:
        dict { strategy: {Accuracy, F1, Precision, Recall} }
    """
    logger.info("\n" + "=" * 50)
    logger.info("BASELINES CLASSIFICATION")
    logger.info("=" * 50)

    strategies = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified"   : DummyClassifier(strategy="stratified", random_state=42),
        "dummy_uniform"      : DummyClassifier(strategy="uniform", random_state=42),
    }

    results = {}
    for name, model in strategies.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {"Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec}
        logger.info(
            f"   {name:<25s} → F1={f1:.4f} | Acc={acc:.4f}"
        )

    return results

def compare_vs_classification_baseline(
    model_metrics: dict, baseline_results: dict
) -> None:
    """
    Compare le vrai modèle contre les baselines de classification.
    """
    logger.info("\nComparaison modèle vs baselines (classification) :")
    model_f1 = model_metrics.get("F1", 0)

    for name, bm in baseline_results.items():
        gain   = model_f1 - bm["F1"]
        status = "PASS" if gain > 0 else "FAIL"
        logger.info(f"   {status} vs {name:<25s} → ΔF1={gain:+.4f}")

    best_f1 = max(b["F1"] for b in baseline_results.values())
    if model_f1 <= best_f1:
        logger.warning(
            "   Le modèle ne bat PAS la meilleure baseline de classification !"
        )
    else:
        logger.info(
            f"   Modèle dépasse la meilleure baseline de ΔF1={model_f1 - best_f1:+.4f}"
        )

def build_baseline_report(
    model_reg_metrics: dict,
    baseline_reg: dict,
    model_clf_metrics: dict | None,
    baseline_clf: dict | None,
) -> pd.DataFrame:
    """
    Construit un DataFrame récapitulatif des performances modèle vs baselines.
    Utile pour l'export dans le rapport HTML.
    """
    rows = []

    rows.append({"type": "régression", "modèle": "Modèle réel", **model_reg_metrics})
    for name, m in baseline_reg.items():
        rows.append({"type": "régression", "modèle": name, **m})

    if model_clf_metrics and baseline_clf:
        rows.append({"type": "classification", "modèle": "Modèle réel", **model_clf_metrics})
        for name, m in baseline_clf.items():
            rows.append({"type": "classification", "modèle": name, **m})

    return pd.DataFrame(rows)
