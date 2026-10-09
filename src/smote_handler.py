"""
smote_handler.py
----------------
Gestion du déséquilibre des classes via SMOTE (Synthetic Minority Over-sampling).

Fonctionnalités :
  - Détection automatique du déséquilibre
  - Application SMOTE sur le train uniquement (jamais sur le test)
  - Rapport avant/après rééquilibrage
  - Fallback silencieux si imbalanced-learn non installé

Usage :
    from smote_handler import SmoteHandler
    handler = SmoteHandler()
    X_res, y_res = handler.fit_resample(X_train, y_train)
"""

from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from logger_setup import get_logger

logger = get_logger(__name__)

try:
    from imblearn.over_sampling import SMOTE, SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logger.warning("⚠️  imbalanced-learn non installé: SMOTE désactivé (pip install imbalanced-learn)")

def _class_distribution(y) -> dict:
    """Retourne la distribution des classes sous forme de dict trié."""
    counts = Counter(y)
    total  = sum(counts.values())
    return {
        cls: {"n": n, "pct": round(n / total * 100, 1)}
        for cls, n in sorted(counts.items(), key=lambda x: str(x[0]))
    }

def _imbalance_ratio(y) -> float:
    """
    Ratio déséquilibre = n_minorité / n_majorité.
    1.0 = parfaitement équilibré, 0.0 = complètement déséquilibré.
    """
    counts = Counter(y)
    if len(counts) < 2:
        return 1.0
    vals = list(counts.values())
    return min(vals) / max(vals)

def detect_imbalance(y, threshold: float = 0.5) -> Tuple[bool, float]:
    """
    Détecte si un déséquilibre significatif existe.

    Args:
        y:         Labels (train uniquement).
        threshold: Ratio en dessous duquel on considère qu'il y a déséquilibre.

    Returns:
        (is_imbalanced: bool, ratio: float)
    """
    ratio = _imbalance_ratio(y)
    is_imbalanced = ratio < threshold
    return is_imbalanced, ratio

class SmoteHandler:
    """
    Applique SMOTE sur les données d'entraînement si déséquilibre détecté.
    Compatible avec les features numériques et mixtes (SMOTENC).
    """

    def __init__(
        self,
        sampling_strategy: str = "auto",
        k_neighbors: int = 5,
        random_state: int = 42,
        auto_detect: bool = True,
        imbalance_threshold: float = 0.5,
        categorical_features: Optional[list] = None,
    ):
        """
        Args:
            sampling_strategy:    "auto" rééquilibre toutes les classes minoritaires.
            k_neighbors:          Nombre de voisins pour la génération de nouveaux points.
            random_state:         Graine aléatoire pour la reproductibilité.
            auto_detect:          Si True, applique SMOTE uniquement si déséquilibre détecté.
            imbalance_threshold:  Seuil de ratio pour déclencher SMOTE.
            categorical_features: Indices des features catégorielles (pour SMOTENC).
        """
        try:
            from config_loader import cfg
            self.sampling_strategy  = sampling_strategy or cfg.smote.sampling_strategy
            self.k_neighbors        = k_neighbors or int(cfg.smote.k_neighbors)
            self.imbalance_threshold = float(cfg.classification.imbalance_ratio_threshold)
        except Exception:
            self.sampling_strategy  = sampling_strategy
            self.k_neighbors        = k_neighbors
            self.imbalance_threshold = imbalance_threshold

        self.random_state        = random_state
        self.auto_detect         = auto_detect
        self.categorical_features = categorical_features
        self._applied            = False
        self._before_dist        = None
        self._after_dist         = None

    def fit_resample(
        self, X_train, y_train, force: bool = False
    ) -> Tuple:
        """
        Rééchantillonne les données d'entraînement si nécessaire.

        Args:
            X_train: Features d'entraînement (numpy array ou DataFrame).
            y_train: Labels d'entraînement.
            force:   Forcer SMOTE même si pas de déséquilibre détecté.

        Returns:
            (X_resampled, y_resampled): mêmes types que l'entrée.
        """
        logger.info("\n" + "=" * 50)
        logger.info("⚖️  SMOTE: Gestion du déséquilibre des classes")
        logger.info("=" * 50)

        self._before_dist = _class_distribution(y_train)
        is_imbalanced, ratio = detect_imbalance(y_train, self.imbalance_threshold)

        self._log_distribution("Avant SMOTE", self._before_dist)
        logger.info(f"   Ratio déséquilibre : {ratio:.3f} (seuil : {self.imbalance_threshold})")

        if not is_imbalanced and not force:
            logger.info("   ℹ️  Déséquilibre non significatif: SMOTE non appliqué")
            self._applied = False
            return X_train, y_train

        if not SMOTE_AVAILABLE:
            logger.warning("   ⚠️  SMOTE non disponible: données inchangées")
            self._applied = False
            return X_train, y_train

        try:
            smote = self._build_smote(X_train)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            self._after_dist = _class_distribution(y_res)
            self._applied = True

            self._log_distribution("Après SMOTE", self._after_dist)
            gain = len(X_res) - len(X_train)
            logger.info(
                f"   ✅ SMOTE appliqué : {len(X_train):,} → {len(X_res):,} "
                f"(+{gain:,} samples synthétiques)"
            )
            return X_res, y_res

        except Exception as exc:
            logger.warning(f"   ⚠️  SMOTE échoué : {exc}: données originales conservées")
            self._applied = False
            return X_train, y_train

    def _build_smote(self, X_train):
        """Construit le bon objet SMOTE selon le type de données."""
        k = min(self.k_neighbors, self._min_class_count(X_train) - 1)
        k = max(1, k)  # au moins 1 voisin

        if self.categorical_features:
            logger.info(
                f"   SMOTENC activé: {len(self.categorical_features)} features catégorielles"
            )
            return SMOTENC(
                categorical_features=self.categorical_features,
                sampling_strategy=self.sampling_strategy,
                k_neighbors=k,
                random_state=self.random_state,
            )
        return SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=k,
            random_state=self.random_state,
        )

    def _min_class_count(self, X_train) -> int:
        """Nombre d'échantillons dans la classe la plus petite (pour ajuster k)."""
        return min(info["n"] for info in self._before_dist.values()) if self._before_dist else 5

    def _log_distribution(self, label: str, dist: dict) -> None:
        logger.info(f"   {label} :")
        for cls, info in dist.items():
            bar = "█" * int(info["pct"] / 5)
            logger.info(f"     {str(cls):<15} {info['n']:>6,}  ({info['pct']:5.1f}%)  {bar}")

    def get_report(self) -> dict:
        """Retourne un rapport dictionnaire pour MLflow / rapport HTML."""
        return {
            "smote_applied"   : self._applied,
            "before_dist"     : self._before_dist,
            "after_dist"      : self._after_dist,
            "k_neighbors"     : self.k_neighbors,
            "sampling_strategy": self.sampling_strategy,
        }

    @property
    def was_applied(self) -> bool:
        return self._applied

def apply_smote_if_needed(
    X_train, y_train,
    threshold: float = 0.5,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple:
    """
    Fonction utilitaire rapide : applique SMOTE si déséquilibre détecté.

    Returns:
        (X_resampled, y_resampled)
    """
    handler = SmoteHandler(
        k_neighbors=k_neighbors,
        random_state=random_state,
        imbalance_threshold=threshold,
    )
    return handler.fit_resample(X_train, y_train)