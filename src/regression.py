"""Train and evaluate property price regression models."""

import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from logger_setup import get_logger

try:
    from config_loader import cfg as _cfg
    _RS  = int(_cfg.pipeline.random_state)
    _CV  = int(_cfg.regression.cv_folds)
    _NIT = int(_cfg.regression.n_iter_search)
except Exception:
    _RS, _CV, _NIT = 42, 5, 20

logger = get_logger(__name__)

def _try_xgboost():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except ImportError:
        logger.warning("     xgboost non installé: ignoré (pip install xgboost)")
        return None

def get_regression_models() -> dict:
    """Retourne le dictionnaire des modèles candidats."""
    models = {
        "Ridge"            : Ridge(alpha=1.0),
        "RandomForest"     : RandomForestRegressor(n_estimators=100, random_state=_RS, n_jobs=1),
        "GradientBoosting" : GradientBoostingRegressor(n_estimators=100, random_state=_RS),
    }
    XGB = _try_xgboost()
    if XGB:
        models["XGBoost"] = XGB(n_estimators=100, random_state=_RS, n_jobs=1, verbosity=0)
    return models

class PriceScaleRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator, log_target=False):
        self.estimator = estimator
        self.log_target = log_target

    def get_params(self, deep=True):
        params = {"estimator": self.estimator, "log_target": self.log_target}
        if deep and hasattr(self.estimator, "get_params"):
            params.update({
                f"estimator__{key}": value
                for key, value in self.estimator.get_params(deep=True).items()
            })
        return params

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        if "log_target" in params:
            self.log_target = params.pop("log_target")
        nested = {
            key[len("estimator__"):]: value
            for key, value in params.items()
            if key.startswith("estimator__")
        }
        if nested:
            self.estimator.set_params(**nested)
        return self

    def fit(self, X, y):
        from sklearn.base import clone
        self.estimator_ = clone(self.estimator)
        target = np.log1p(y) if self.log_target else y
        self.estimator_.fit(X, target)
        return self

    def predict(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "estimator_")
        values = self.estimator_.predict(X)
        return np.expm1(values) if self.log_target else values


def train_regression(
    X_train, y_train, use_log_target: bool = False, cv_frame=None
):
    """
    Entraîne plusieurs modèles de régression et retourne le meilleur.

    Args:
        X_train:        Features d'entraînement.
        y_train:        Target (prix brut).
        use_log_target: Si True, entraîne sur log1p(prix) pour distribution plus gaussienne.

    Returns:
        (best_model, best_name, use_log_target)
    """
    logger.info("\n" + "=" * 50)
    logger.info(" MODÈLE DE RÉGRESSION: Prédiction du Prix")
    logger.info("=" * 50)

    y = np.log1p(y_train) if use_log_target else y_train
    if use_log_target:
        logger.info("    Target : log1p(prix)")

    models  = get_regression_models()
    results = {}

    cv_features = X_train
    cv_preprocessor = None
    if cv_frame is not None:
        from sklearn.pipeline import Pipeline
        from sklearn.base import clone
        from prepare import build_preprocessor, detect_column_types

        numeric_cols, categorical_cols = detect_column_types(cv_frame)
        selected_cols = numeric_cols + categorical_cols
        if not selected_cols:
            raise ValueError("Cross-validation requires usable features")
        cv_features = cv_frame[selected_cols]
        cv_preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    for name, model in models.items():
        cv_model = model
        if cv_preprocessor is not None:
            cv_model = Pipeline([
                ("preprocessor", clone(cv_preprocessor)),
                ("estimator", clone(model)),
            ])
        scores = cross_val_score(cv_model, cv_features, y, cv=_CV, scoring="r2", n_jobs=-1)
        results[name] = scores.mean()
        logger.info(
            f"   {name:<25s} → R² CV : {scores.mean():.4f} (±{scores.std():.4f})"
        )

    best_name  = max(results, key=results.get)
    best_model = models[best_name]
    logger.info(f"\n Meilleur modèle : {best_name} (R² = {results[best_name]:.4f})")

    best_model.fit(X_train, y)
    return best_model, best_name, use_log_target

def optimize_model(model, X_train, y_train, n_iter: int = None, cv_frame=None):
    """
    Optimise les hyperparamètres via RandomizedSearchCV.
    Plus rapide que GridSearchCV, aussi efficace en pratique.
    """
    logger.info("\n Optimisation des hyperparamètres (RandomizedSearchCV) ...")
    model_name = type(model).__name__

    grids = {
        "RandomForestRegressor": {
            "n_estimators"     : [100, 200, 300],
            "max_depth"        : [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features"     : ["sqrt", "log2", 0.5],
        },
        "GradientBoostingRegressor": {
            "n_estimators" : [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth"    : [3, 5, 7],
            "subsample"    : [0.8, 1.0],
        },
        "XGBRegressor": {
            "n_estimators"    : [100, 200, 300],
            "learning_rate"   : [0.01, 0.05, 0.1],
            "max_depth"       : [3, 5, 7],
            "subsample"       : [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        },
    }
    param_dist = grids.get(model_name)
    if not param_dist:
        logger.warning(f"     Pas de grille pour {model_name}: optimisation ignorée")
        return model

    search_model = model
    search_features = X_train
    if cv_frame is not None:
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline
        from prepare import build_preprocessor, detect_column_types

        numeric_cols, categorical_cols = detect_column_types(cv_frame)
        selected_cols = numeric_cols + categorical_cols
        if not selected_cols:
            raise ValueError("Cross-validation requires usable features")
        search_features = cv_frame[selected_cols]
        search_model = Pipeline([
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("estimator", clone(model)),
        ])
        param_dist = {
            f"estimator__{key}": values for key, values in param_dist.items()
        }

    search = RandomizedSearchCV(
        search_model, param_dist, n_iter=(n_iter or _NIT), cv=_CV,
        scoring="r2", random_state=_RS, n_jobs=-1, verbose=0,
    )
    search.fit(search_features, y_train)
    logger.info("Optimized CV score: %.4f", search.best_score_)
    if cv_frame is None:
        return search.best_estimator_

    best_params = {
        key.removeprefix("estimator__"): value
        for key, value in search.best_params_.items()
    }
    tuned_model = clone(model).set_params(**best_params)
    tuned_model.fit(X_train, y_train)
    return tuned_model

def evaluate_regression(
    model,
    X_test,
    y_test,
    use_log_target: bool = False,
    baseline_results: Optional[dict] = None,
):
    """
    Évalue le modèle sur le test set.
    Si use_log_target=True, inverse-transforme les prédictions avant les métriques.

    Args:
        baseline_results : dict retourné par run_regression_baselines(),
                           si fourni, vérifie que le modèle bat les baselines.
    """
    y_pred_raw = model.predict(X_test)
    y_pred = np.expm1(y_pred_raw) if use_log_target else y_pred_raw
    y_true = np.array(y_test)

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    nonzero_mask = np.abs(y_true) > np.finfo(float).eps
    mape = (
        np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask])
                       / np.abs(y_true[nonzero_mask]))) * 100
        if nonzero_mask.sum() > 0 else float("nan")
    )

    logger.info("\n RÉSULTATS RÉGRESSION (Test Set) :")
    logger.info(f"   MAE  : {mae:>15,.2f} MAD")
    logger.info(f"   MSE  : {mse:>15,.2f}")
    logger.info(f"   RMSE : {rmse:>15,.2f} MAD")
    logger.info(f"   MAPE : {mape:>14.2f}%" if not np.isnan(mape) else "   MAPE : N/A")
    logger.info(f"   R²   : {r2:>14.4f}")

    if r2 >= 0.85:
        logger.info("    Excellent modèle !")
    elif r2 >= 0.70:
        logger.info("    Bon modèle: peut être amélioré")
    elif r2 >= 0.50:
        logger.info("    Modèle moyen: revoir les features")
    else:
        logger.info("    Modèle faible: approfondir l'analyse")

    if baseline_results:
        best_baseline_r2 = max(
            v.get("R2", -999) for v in baseline_results.values()
        )
        if r2 <= best_baseline_r2:
            logger.warning(
                f"\n     ALERTE BASELINE : R²={r2:.4f} ≤ meilleure baseline "
                f"R²={best_baseline_r2:.4f}: le modèle ML n'apporte pas de valeur ajoutée !"
                f"\n   → Vérifier : features, target leakage, données insuffisantes."
            )
        else:
            logger.info(
                f"    Baseline battue : R²={r2:.4f} > baseline={best_baseline_r2:.4f} "
                f"(+{r2 - best_baseline_r2:.4f})"
            )

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

def get_feature_importance(model, feature_names: list, top_n: int = 15):
    """
    Affiche les features les plus importantes.
    - tree-based : feature_importances_
    - Ridge      : |coef_| (importance absolue)
    """
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        importances = pd.Series(np.abs(model.coef_), index=feature_names)
    else:
        logger.warning("     Modèle sans feature importance")
        return None

    importances = importances.sort_values(ascending=False)
    logger.info(f"\n Top {top_n} features (régression) :")
    for feat, imp in importances.head(top_n).items():
        bar = "█" * int(imp * 40)
        logger.info(f"   {feat:<35s} {bar} {imp:.4f}")
    return importances

def save_model(model, path: str = "models/regression_model.pkl") -> None:
    """Sauvegarde le modèle entraîné sur disque."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f" Modèle régression sauvegardé → {path}")

def load_model(path: str = "models/regression_model.pkl"):
    """Charge un modèle depuis disque."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f" Modèle régression chargé depuis {path}")
    return model
