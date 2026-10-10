"""
shap_explainer.py
-----------------
Interprétabilité des modèles via SHAP (SHapley Additive exPlanations).

Génère automatiquement :
  - Summary plot   : vue globale des features les plus impactantes
  - Bar plot       : importance moyenne |SHAP| par feature
  - Waterfall plot : explication d'une prédiction individuelle
  - SHAP values    : exportées pour le rapport HTML

Usage :
    from shap_explainer import SHAPExplainer
    explainer = SHAPExplainer(model, X_train, feature_names)
    explainer.run(X_test, output_dir="docs/plots")
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from logger_setup import get_logger

logger = get_logger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("WARNING  shap non installé: interprétabilité désactivée (pip install shap)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

class SHAPExplainer:
    """
    Wrapper SHAP compatible Tree-based et modèles linéaires.
    Fallback silencieux si shap n'est pas installé.
    """

    def __init__(
        self,
        model,
        X_background,
        feature_names: list[str],
        model_type: str = "auto",
        n_samples: int = 500,
    ):
        """
        Args:
            model:         Modèle sklearn entraîné.
            X_background:  Données d'arrière-plan (subset du train).
            feature_names: Noms des features.
            model_type:    "tree" | "linear" | "kernel" | "auto".
            n_samples:     Nombre max d'échantillons pour le calcul SHAP.
        """
        self.model        = model
        self.feature_names = feature_names
        self.n_samples    = n_samples
        self.explainer    = None
        self.shap_values  = None

        try:
            from config_loader import cfg
            self.n_samples   = n_samples or cfg.shap.n_samples
            self.max_display = cfg.shap.max_display
        except Exception:
            self.max_display = 20

        if not SHAP_AVAILABLE:
            return

        bg = X_background
        if hasattr(bg, "shape") and bg.shape[0] > self.n_samples:
            idx = np.random.choice(bg.shape[0], self.n_samples, replace=False)
            bg = bg.iloc[idx] if hasattr(bg, "iloc") else bg[idx]

        self.explainer = self._build_explainer(model, bg, model_type)

    def _build_explainer(self, model, background, model_type: str):
        """Choisit le bon type d'explainer SHAP selon le modèle."""
        if not SHAP_AVAILABLE:
            return None

        model_class = type(model).__name__
        tree_models = {
            "RandomForestRegressor", "RandomForestClassifier",
            "GradientBoostingRegressor", "GradientBoostingClassifier",
            "XGBRegressor", "XGBClassifier",
            "DecisionTreeRegressor", "DecisionTreeClassifier",
        }
        linear_models = {"Ridge", "Lasso", "LogisticRegression", "LinearRegression"}

        try:
            if model_type == "tree" or (model_type == "auto" and model_class in tree_models):
                logger.info(f"   SHAP : TreeExplainer pour {model_class}")
                return shap.TreeExplainer(model)

            elif model_type == "linear" or (model_type == "auto" and model_class in linear_models):
                logger.info(f"   SHAP : LinearExplainer pour {model_class}")
                return shap.LinearExplainer(model, background)

            else:
                logger.info(f"   SHAP : KernelExplainer pour {model_class} (plus lent)")
                summary = shap.kmeans(background, 10)
                return shap.KernelExplainer(model.predict, summary)

        except Exception as exc:
            logger.warning(f"   WARNING  SHAP explainer échoué : {exc}")
            return None

    def compute(self, X_test) -> Optional[np.ndarray]:
        """
        Calcule les SHAP values sur X_test.
        Retourne un tableau numpy ou None si SHAP indisponible.
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return None

        X = X_test
        if hasattr(X, "shape") and X.shape[0] > self.n_samples:
            idx = np.random.choice(X.shape[0], self.n_samples, replace=False)
            X = X.iloc[idx] if hasattr(X, "iloc") else X[idx]

        try:
            logger.info(f"   Calcul SHAP values sur {X.shape[0]} échantillons ...")
            sv = self.explainer.shap_values(X)

            if isinstance(sv, list):
                sv = sv[-1]

            self.shap_values = sv
            self.X_explained = X
            logger.info("   PASS SHAP values calculées")
            return sv

        except Exception as exc:
            logger.warning(f"   WARNING  Calcul SHAP échoué : {exc}")
            return None

    def plot_summary(self, output_dir: str, prefix: str = "") -> Optional[str]:
        """Summary plot (beeswarm): vue globale des features."""
        if not self._ready() or not MPL_AVAILABLE:
            return None
        try:
            path = os.path.join(output_dir, f"{prefix}shap_summary.png")
            plt.figure(figsize=(10, 7))
            shap.summary_plot(
                self.shap_values, self.X_explained,
                feature_names=self.feature_names,
                max_display=self.max_display,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"    SHAP summary → {path}")
            return path
        except Exception as exc:
            logger.warning(f"   WARNING  SHAP summary plot échoué : {exc}")
            return None

    def plot_bar(self, output_dir: str, prefix: str = "") -> Optional[str]:
        """Bar plot: importance moyenne |SHAP| par feature."""
        if not self._ready() or not MPL_AVAILABLE:
            return None
        try:
            path = os.path.join(output_dir, f"{prefix}shap_bar.png")
            plt.figure(figsize=(10, 7))
            shap.summary_plot(
                self.shap_values, self.X_explained,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=self.max_display,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"    SHAP bar → {path}")
            return path
        except Exception as exc:
            logger.warning(f"   WARNING  SHAP bar plot échoué : {exc}")
            return None

    def plot_waterfall(
        self, output_dir: str, sample_idx: int = 0, prefix: str = ""
    ) -> Optional[str]:
        """Waterfall plot: explication d'une seule prédiction."""
        if not self._ready() or not MPL_AVAILABLE:
            return None
        try:
            path = os.path.join(output_dir, f"{prefix}shap_waterfall.png")
            X_row = self.X_explained.iloc[[sample_idx]] if hasattr(self.X_explained, "iloc") \
                    else self.X_explained[[sample_idx]]
            sv_row = self.shap_values[sample_idx]

            explanation = shap.Explanation(
                values=sv_row,
                base_values=self.explainer.expected_value
                    if not isinstance(self.explainer.expected_value, list)
                    else self.explainer.expected_value[-1],
                data=X_row.values[0] if hasattr(X_row, "values") else X_row[0],
                feature_names=self.feature_names,
            )
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"    SHAP waterfall → {path}")
            return path
        except Exception as exc:
            logger.warning(f"   WARNING  SHAP waterfall échoué : {exc}")
            return None

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Retourne un DataFrame trié par importance SHAP moyenne (|SHAP|).
        Utile pour le rapport et MLflow.
        """
        if not self._ready():
            return None
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature"   : self.feature_names,
            "shap_mean" : mean_abs,
        }).sort_values("shap_mean", ascending=False).reset_index(drop=True)
        return df

    def run(
        self, X_test, output_dir: str = "docs/plots", prefix: str = ""
    ) -> dict[str, Optional[str]]:
        """
        Calcule les SHAP values et génère tous les plots configurés.

        Returns:
            dict { plot_type: chemin_fichier }
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info("\n" + "=" * 50)
        logger.info(" SHAP: Interprétabilité du modèle")
        logger.info("=" * 50)

        if not SHAP_AVAILABLE:
            logger.warning("   SHAP non disponible: skip")
            return {}

        self.compute(X_test)
        if self.shap_values is None:
            return {}

        plots = {}
        plots["summary"]   = self.plot_summary(output_dir, prefix)
        plots["bar"]       = self.plot_bar(output_dir, prefix)
        plots["waterfall"] = self.plot_waterfall(output_dir, prefix=prefix)

        importance_df = self.get_feature_importance()
        if importance_df is not None:
            csv_path = os.path.join(output_dir, f"{prefix}shap_importance.csv")
            importance_df.to_csv(csv_path, index=False)
            plots["importance_csv"] = csv_path
            logger.info(f"    SHAP importance → {csv_path}")

        return plots

    def _ready(self) -> bool:
        return SHAP_AVAILABLE and self.explainer is not None and self.shap_values is not None
