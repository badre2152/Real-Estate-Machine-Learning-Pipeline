"""
prediction_intervals.py
-----------------------
Intervalles de prédiction pour les modèles de régression.

Au lieu de renvoyer une seule valeur, le modèle retourne un intervalle
de confiance [lower, upper] autour de la prédiction centrale.

Deux méthodes disponibles :
  1. Quantile (GradientBoosting / XGBoost natif): précise et rapide
  2. Bootstrap: universelle, fonctionne avec tout modèle sklearn

Usage :
    from prediction_intervals import PredictionIntervalBuilder
    builder = PredictionIntervalBuilder(method="bootstrap", n_bootstrap=200)
    builder.fit(model, X_train, y_train)
    df_pred = builder.predict_with_interval(X_test)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from logger_setup import get_logger

logger = get_logger(__name__)

class PredictionIntervalBuilder:
    """
    Calcule des intervalles de prédiction pour les modèles de régression.

    Méthodes :
      - "quantile"  : utilise les quantiles des résidus du train (rapide)
      - "bootstrap" : réentraîne N modèles sur des sous-ensembles bootstrap
    """

    def __init__(
        self,
        method: str = "quantile",
        confidence_level: float = 0.95,
        n_bootstrap: int = 200,
        random_state: int = 42,
    ):
        """
        Args:
            method:           "quantile" ou "bootstrap".
            confidence_level: Niveau de confiance (ex: 0.95 pour 95% CI).
            n_bootstrap:      Nombre de modèles bootstrap (ignoré si method="quantile").
            random_state:     Graine pour la reproductibilité.
        """
        self.method = method
        self.confidence_level = float(confidence_level)
        self.n_bootstrap = int(n_bootstrap)
        if self.method not in ("quantile", "bootstrap"):
            raise ValueError("Unknown prediction interval method")
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between zero and one")
        if self.method == "bootstrap" and self.n_bootstrap < 1:
            raise ValueError("Bootstrap requires at least one model")

        self.random_state = random_state
        self.alpha        = 1 - self.confidence_level
        self._fitted      = False

        self._residual_lower: Optional[float] = None
        self._residual_upper: Optional[float] = None
        self._bootstrap_models: list = []
        self._base_model = None

        logger.info(
            f"   PredictionIntervals : method={self.method}, "
            f"CI={self.confidence_level:.0%}, alpha={self.alpha:.3f}"
        )

    def fit(self, model, X_train, y_train, X_cal=None, y_cal=None) -> "PredictionIntervalBuilder":
        """
        Calibre les intervalles de prédiction sur les données d'entraînement.

        Args:
            model:   Modèle sklearn déjà entraîné (pour method="quantile")
                     ou template à cloner (pour method="bootstrap").
            X_train: Features d'entraînement.
            y_train: Target d'entraînement.

        Returns:
            self (chainable)
        """
        self._base_model = model

        if self.method == "quantile":
            if X_cal is None or y_cal is None or len(y_cal) < 2:
                raise ValueError("Quantile calibration requires an independent calibration set")
            self._fit_quantile(model, X_cal, y_cal)
        elif self.method == "bootstrap":
            self._fit_bootstrap(model, X_train, y_train)
        else:
            raise ValueError(f"Méthode inconnue : {self.method}. Choisir 'quantile' ou 'bootstrap'.")

        self._fitted = True
        return self

    def _fit_quantile(self, model, X_cal, y_cal) -> None:
        """Calcule les quantiles des résidus sur le train."""
        logger.info("   PI : calibration quantile des résidus ...")
        y_pred_cal = model.predict(X_cal)
        residuals = np.asarray(y_cal) - y_pred_cal
        if not np.all(np.isfinite(residuals)):
            raise ValueError("Nonfinite calibration residuals")

        import math
        n = len(residuals)
        rank = min(n, math.ceil((n + 1) * (1 - self.alpha)))
        radius = float(np.partition(np.abs(residuals), rank - 1)[rank - 1])
        self._residual_lower = -radius
        self._residual_upper = radius

        logger.info(
            f"   PI quantile calibré : [{self._residual_lower:+,.0f}, {self._residual_upper:+,.0f}]"
        )

    def _fit_bootstrap(self, model, X_train, y_train) -> None:
        """Entraîne N modèles bootstrap pour estimer la variance de prédiction."""
        from sklearn.base import clone

        logger.info(f"   PI : bootstrap avec {self.n_bootstrap} modèles ...")
        rng = np.random.default_rng(self.random_state)
        n   = len(y_train)
        y_np = np.asarray(y_train)

        self._bootstrap_models = []
        for i in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            X_sample = X_train.iloc[idx] if hasattr(X_train, "iloc") else np.asarray(X_train)[idx]
            m = clone(model)
            m.fit(X_sample, y_np[idx])
            self._bootstrap_models.append(m)
            if (i + 1) % 50 == 0:
                logger.debug(f"      Bootstrap {i+1}/{self.n_bootstrap}")

        logger.info(f"   ✅ {self.n_bootstrap} modèles bootstrap entraînés")

    def predict_with_interval(self, X_test) -> pd.DataFrame:
        """
        Génère des prédictions avec intervalles de confiance.

        Returns:
            DataFrame avec colonnes :
              - prediction     : prédiction centrale
              - lower          : borne inférieure de l'intervalle
              - upper          : borne supérieure de l'intervalle
              - interval_width : largeur de l'intervalle (upper - lower)
        """
        if not self._fitted:
            raise RuntimeError("Appeler .fit() avant .predict_with_interval()")

        point_pred = self._base_model.predict(X_test)

        if self.method == "quantile":
            lower = point_pred + self._residual_lower
            upper = point_pred + self._residual_upper

        elif self.method == "bootstrap":
            if not self._bootstrap_models:
                raise ValueError("No bootstrap models available")
            all_preds = np.stack([m.predict(X_test) for m in self._bootstrap_models], axis=1)
            lower = np.quantile(all_preds, self.alpha / 2, axis=1)
            upper = np.quantile(all_preds, 1 - self.alpha / 2, axis=1)

        if not np.all(np.isfinite(point_pred)) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Nonfinite prediction interval values")
        if np.any(point_pred < 0):
            raise ValueError("Negative point prediction")

        lower = np.maximum(lower, 0.0)
        lower = np.minimum(lower, point_pred)
        upper = np.maximum(upper, point_pred)

        interval_width = upper - lower

        n_degenerate = int(np.sum(interval_width <= 0))
        if n_degenerate > 0:
            logger.warning(
                f"   ⚠️  {n_degenerate} intervalles dégénérés (lower >= upper) "
                f"Vérifier le calibrage du modèle."
            )

        df = pd.DataFrame({
            "prediction"     : point_pred,
            "lower"          : lower,
            "upper"          : upper,
            "interval_width" : interval_width,
        })

        mean_width = df["interval_width"].mean()
        logger.info(
            f"   PI {self.confidence_level:.0%} : largeur moyenne = {mean_width:,.0f} MAD | "
            f"min={df['lower'].min():,.0f} | max={df['upper'].max():,.0f}"
        )

        return df

    def evaluate_coverage(
        self, X_test, y_test
    ) -> dict:
        """
        Évalue la couverture réelle de l'intervalle sur des données avec vérité terrain.

        Returns:
            dict {coverage, mean_width, picp, mpiw}
        """
        df    = self.predict_with_interval(X_test)
        y_arr = np.array(y_test)

        in_interval = ((y_arr >= df["lower"]) & (y_arr <= df["upper"]))
        coverage    = in_interval.mean()

        metrics = {
            "picp"          : float(coverage),          # Prediction Interval Coverage Prob.
            "target_coverage": self.confidence_level,
            "mpiw"          : float(df["interval_width"].mean()),  # Mean Prediction Interval Width
            "coverage_gap"  : float(coverage - self.confidence_level),
        }

        logger.info(
            f"   📐 Couverture réelle : {coverage:.1%} "
            f"(cible : {self.confidence_level:.1%}, "
            f"écart : {metrics['coverage_gap']:+.1%})"
        )
        logger.info(f"   📐 Largeur moyenne   : {metrics['mpiw']:,.0f}")

        if abs(metrics["coverage_gap"]) > 0.05:
            logger.warning(
                f"   ⚠️  Écart couverture > 5%: envisager une recalibration"
            )

        return metrics

    def format_prediction(self, row: pd.Series) -> str:
        """Formate une ligne de résultat en texte lisible."""
        return (
            f"{row['prediction']:,.0f} MAD "
            f"[{row['lower']:,.0f} à {row['upper']:,.0f}] "
            f"(±{row['interval_width']/2:,.0f})"
        )

def predict_with_ci(
    model,
    X_train, y_train,
    X_test,
    method: str = "quantile",
    confidence: float = 0.95,
    X_cal=None,
    y_cal=None,
) -> pd.DataFrame:
    """
    Raccourci : calibre et prédit avec intervalles en une seule ligne.

    Returns:
        DataFrame prediction + lower + upper + interval_width
    """
    builder = PredictionIntervalBuilder(
        method=method, confidence_level=confidence
    )
    builder.fit(model, X_train, y_train, X_cal=X_cal, y_cal=y_cal)
    return builder.predict_with_interval(X_test)
