"""
drift_detector.py
-----------------
Drift Detection للـ MLOps الحقيقي: Avito Real Estate Pipeline v3.

نوعان من الـ Drift:
  1. Data Drift    : توزيع الـ features تغيّر (السوق تغيّر، بيانات جديدة مختلفة)
  2. Concept Drift : العلاقة بين features والـ target تغيّرت (الأسعار تضخّمت مثلاً)

Tests المستعملة:
  - PSI  (Population Stability Index)  → أشهر test في الـ industry
  - KS   (Kolmogorov-Smirnov)          → مقارنة توزيعات numerical
  - Chi² (Chi-Squared)                 → categorical features
  - Prediction Drift                   → توزيع الـ predictions تغيّر

Thresholds المعيارية:
  PSI < 0.10  → PASS Stable
  PSI 0.10-0.20 → WARNING Warning
  PSI > 0.20  → DRIFT Drift détecté

Usage :
    from drift_detector import DriftDetector

    detector = DriftDetector(reference_data=X_train)

    report = detector.detect(current_data=X_new)

    if report.has_drift:
        print(report.summary())
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from logger_setup import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)

def _load_drift_cfg():
    try:
        from config_loader import cfg
        return {
            "psi_stable" : float(getattr(cfg.drift, "psi_threshold_warning", 0.10)),
            "psi_warning": float(getattr(cfg.drift, "psi_threshold_drift",   0.20)),
            "ks_pval"    : float(getattr(cfg.drift, "ks_pvalue_threshold",    0.05)),
            "chi2_pval"  : float(getattr(cfg.drift, "chi2_pvalue_threshold",  0.05)),
        }
    except Exception:
        return {"psi_stable": 0.10, "psi_warning": 0.20, "ks_pval": 0.05, "chi2_pval": 0.05}

_DRIFT_CFG = _load_drift_cfg()

PSI_STABLE            = _DRIFT_CFG["psi_stable"]
PSI_WARNING           = _DRIFT_CFG["psi_warning"]
KS_PVALUE_THRESHOLD   = _DRIFT_CFG["ks_pval"]
CHI2_PVALUE_THRESHOLD = _DRIFT_CFG["chi2_pval"]

def _psi_severity(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_WARNING:
        return "warning"
    return "drift"

@dataclass
class FeatureDriftResult:
    """Résultat de drift pour une seule feature."""
    feature     : str
    test_used   : str          # "psi", "ks", "chi2"
    statistic   : float        # PSI value ou KS statistic
    p_value     : Optional[float]
    severity    : str          # "stable", "warning", "drift"
    ref_mean    : Optional[float] = None
    cur_mean    : Optional[float] = None
    ref_std     : Optional[float] = None
    cur_std     : Optional[float] = None

    @property
    def has_drift(self) -> bool:
        return self.severity == "drift"

    @property
    def has_warning(self) -> bool:
        return self.severity in ("warning", "drift")

    def to_dict(self) -> dict:
        return {
            "feature"  : self.feature,
            "test"     : self.test_used,
            "statistic": round(self.statistic, 4),
            "p_value"  : round(self.p_value, 4) if self.p_value else None,
            "severity" : self.severity,
            "ref_mean" : round(self.ref_mean, 2) if self.ref_mean is not None else None,
            "cur_mean" : round(self.cur_mean, 2) if self.cur_mean is not None else None,
        }

@dataclass
class PredictionDriftResult:
    """Résultat de drift sur les prédictions du modèle."""
    psi            : float
    ks_statistic   : float
    ks_p_value     : float
    severity       : str
    ref_mean       : float
    cur_mean       : float
    mean_shift_pct : float    # % de changement de la moyenne

    @property
    def has_drift(self) -> bool:
        return self.severity == "drift"

    @property
    def has_warning(self) -> bool:
        return self.severity in ("warning", "drift")

    def to_dict(self) -> dict:
        return {
            "psi"           : round(self.psi, 4),
            "ks_statistic"  : round(self.ks_statistic, 4),
            "ks_p_value"    : round(self.ks_p_value, 4),
            "severity"      : self.severity,
            "ref_mean"      : round(self.ref_mean, 2),
            "cur_mean"      : round(self.cur_mean, 2),
            "mean_shift_pct": round(self.mean_shift_pct, 2),
        }

@dataclass
class DriftReport:
    """Rapport complet de drift: features + predictions."""
    timestamp          : str
    n_features_checked : int
    n_drifted          : int
    n_warnings         : int
    feature_results    : list[FeatureDriftResult]
    prediction_drift   : Optional[PredictionDriftResult]
    dataset_psi        : float           # PSI global (moyenne pondérée)
    recommendation     : str             # "ok", "monitor", "retrain"
    ref_size           : int
    cur_size           : int

    @property
    def has_drift(self) -> bool:
        return self.n_drifted > 0

    @property
    def needs_retraining(self) -> bool:
        return self.recommendation == "retrain"

    def drifted_features(self) -> list[str]:
        return [r.feature for r in self.feature_results if r.has_drift]

    def warning_features(self) -> list[str]:
        return [r.feature for r in self.feature_results if r.severity == "warning"]

    def summary(self) -> str:
        """Résumé lisible en une ligne."""
        icon = {"ok": "PASS", "monitor": "WARNING", "retrain": "DRIFT"}.get(self.recommendation, "")
        return (
            f"{icon} Drift [{self.recommendation.upper()}]: "
            f"{self.n_drifted}/{self.n_features_checked} features driftées | "
            f"PSI global={self.dataset_psi:.3f} | "
            f"{'Prédictions OK' if not self.prediction_drift or not self.prediction_drift.has_drift else 'Prédictions driftées WARNING'}"
        )

    def to_dict(self) -> dict:
        return {
            "timestamp"          : self.timestamp,
            "summary"            : self.summary(),
            "recommendation"     : self.recommendation,
            "dataset_psi"        : round(self.dataset_psi, 4),
            "n_features_checked" : self.n_features_checked,
            "n_drifted"          : self.n_drifted,
            "n_warnings"         : self.n_warnings,
            "drifted_features"   : self.drifted_features(),
            "warning_features"   : self.warning_features(),
            "ref_size"           : self.ref_size,
            "cur_size"           : self.cur_size,
            "features"           : [r.to_dict() for r in self.feature_results],
            "prediction_drift"   : self.prediction_drift.to_dict() if self.prediction_drift else None,
        }

    def save(self, output_dir: str = "reports/runtime") -> str:
        """Sauvegarde le rapport en JSON."""
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"drift_report_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"    Drift report → {path}")
        return path

def _compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI).

    PSI = Σ (current% - reference%) × ln(current% / reference%)

    Thresholds :
        PSI < 0.10  → Stable
        0.10-0.20   → Warning
        PSI > 0.20  → Drift
    """
    combined_min = min(np.nanmin(reference), np.nanmin(current))
    combined_max = max(np.nanmax(reference), np.nanmax(current))

    if combined_min == combined_max:
        return 0.0  # Pas assez de variabilité

    breakpoints = np.linspace(combined_min, combined_max, n_bins + 1)

    def _bin_counts(data: np.ndarray) -> np.ndarray:
        counts = np.histogram(data, bins=breakpoints)[0]
        counts = np.where(counts == 0, 0.0001, counts)
        return counts / counts.sum()

    ref_pct = _bin_counts(reference)
    cur_pct = _bin_counts(current)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(np.abs(psi))  # abs pour éviter les valeurs négatives marginales

class DriftDetector:
    """
    Détecteur de drift pour le pipeline Avito Real Estate.

    Le détecteur est initialisé avec les données de référence (X_train).
    Pour chaque nouvelle fenêtre de données, appeler detect() pour obtenir
    un rapport complet.

    Parameters
    ----------
    reference_data   : DataFrame de référence (X_train après preprocessing)
    numerical_cols   : liste des colonnes numériques (auto-détecté si None)
    categorical_cols : liste des colonnes catégorielles (auto-détecté si None)
    psi_threshold_warning : seuil PSI pour "warning" (défaut: 0.10)
    psi_threshold_drift   : seuil PSI pour "drift" (défaut: 0.20)
    output_dir       : dossier pour les rapports JSON
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        numerical_cols: Optional[list[str]] = None,
        categorical_cols: Optional[list[str]] = None,
        psi_threshold_warning: float = PSI_STABLE,
        psi_threshold_drift: float = PSI_WARNING,
        output_dir: str = "reports/runtime",
    ):
        self.reference     = reference_data.copy()
        self.output_dir    = output_dir
        self._psi_warn     = psi_threshold_warning
        self._psi_drift    = psi_threshold_drift

        if numerical_cols is not None:
            self.numerical_cols = numerical_cols
        else:
            self.numerical_cols = list(
                reference_data.select_dtypes(include=["number"]).columns
            )

        if categorical_cols is not None:
            self.categorical_cols = categorical_cols
        else:
            self.categorical_cols = list(
                reference_data.select_dtypes(include=["object", "category"]).columns
            )

        self._reports: list[DriftReport] = []

        logger.info(
            f" DriftDetector initialisé: "
            f"ref={len(reference_data)} lignes | "
            f"{len(self.numerical_cols)} num | "
            f"{len(self.categorical_cols)} cat"
        )

    def detect(
        self,
        current_data: pd.DataFrame,
        predictions_ref: Optional[np.ndarray] = None,
        predictions_cur: Optional[np.ndarray] = None,
        save_report: bool = True,
    ) -> DriftReport:
        """
        Analyse complète du drift entre les données de référence et actuelles.

        Parameters
        ----------
        current_data     : nouvelles données à comparer
        predictions_ref  : prédictions sur les données de référence (optionnel)
        predictions_cur  : prédictions sur les nouvelles données (optionnel)
        save_report      : sauvegarder le rapport JSON

        Returns
        -------
        DriftReport avec tous les résultats
        """
        logger.info(
            f"\n{'='*55}\n"
            f" DRIFT DETECTION\n"
            f"   Référence : {len(self.reference)} lignes\n"
            f"   Courant   : {len(current_data)} lignes\n"
            f"{'='*55}"
        )

        feature_results: list[FeatureDriftResult] = []

        for col in self.numerical_cols:
            if col not in current_data.columns:
                continue
            result = self._check_numerical(col, current_data[col])
            feature_results.append(result)

        for col in self.categorical_cols:
            if col not in current_data.columns:
                continue
            result = self._check_categorical(col, current_data[col])
            feature_results.append(result)

        pred_drift = None
        if predictions_ref is not None and predictions_cur is not None:
            pred_drift = self._check_predictions(
                np.array(predictions_ref),
                np.array(predictions_cur),
            )

        psi_values = [
            r.statistic for r in feature_results
            if r.test_used == "psi" and r.statistic is not None
        ]
        dataset_psi = float(np.mean(psi_values)) if psi_values else 0.0

        n_drifted  = sum(1 for r in feature_results if r.has_drift)
        n_warnings = sum(1 for r in feature_results if r.severity == "warning")

        recommendation = self._recommend(
            n_drifted=n_drifted,
            n_total=len(feature_results),
            dataset_psi=dataset_psi,
            pred_drift=pred_drift,
        )

        report = DriftReport(
            timestamp          = datetime.now().isoformat(),
            n_features_checked = len(feature_results),
            n_drifted          = n_drifted,
            n_warnings         = n_warnings,
            feature_results    = feature_results,
            prediction_drift   = pred_drift,
            dataset_psi        = dataset_psi,
            recommendation     = recommendation,
            ref_size           = len(self.reference),
            cur_size           = len(current_data),
        )

        self._reports.append(report)

        self._log_report(report)

        if save_report:
            report.save(self.output_dir)

        return report

    def _check_numerical(
        self,
        col: str,
        current: pd.Series,
    ) -> FeatureDriftResult:
        """PSI + KS test pour une feature numérique."""
        ref_vals = self.reference[col].dropna().values
        cur_vals = current.dropna().values

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            return FeatureDriftResult(
                feature=col, test_used="psi",
                statistic=0.0, p_value=None, severity="stable",
            )

        psi = _compute_psi(ref_vals, cur_vals)
        severity = self._psi_severity_custom(psi)

        ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, cur_vals)

        if severity == "warning" and ks_pvalue < KS_PVALUE_THRESHOLD:
            severity = "drift"

        return FeatureDriftResult(
            feature   = col,
            test_used = "psi",
            statistic = psi,
            p_value   = float(ks_pvalue),
            severity  = severity,
            ref_mean  = float(np.mean(ref_vals)),
            cur_mean  = float(np.mean(cur_vals)),
            ref_std   = float(np.std(ref_vals)),
            cur_std   = float(np.std(cur_vals)),
        )

    def _check_categorical(
        self,
        col: str,
        current: pd.Series,
    ) -> FeatureDriftResult:
        """Chi-squared test pour une feature catégorielle."""
        ref_vals = self.reference[col].dropna()
        cur_vals = current.dropna()

        if len(ref_vals) < 5 or len(cur_vals) < 5:
            return FeatureDriftResult(
                feature=col, test_used="chi2",
                statistic=0.0, p_value=1.0, severity="stable",
            )

        all_cats = set(ref_vals.unique()) | set(cur_vals.unique())
        ref_counts = ref_vals.value_counts().reindex(all_cats, fill_value=0.0001)
        cur_counts = cur_vals.value_counts().reindex(all_cats, fill_value=0.0001)

        chi2_stat, p_value = stats.chisquare(
            cur_counts.values,
            f_exp=ref_counts.values / ref_counts.sum() * cur_counts.sum(),
        )

        severity = "drift" if p_value < CHI2_PVALUE_THRESHOLD else "stable"

        return FeatureDriftResult(
            feature   = col,
            test_used = "chi2",
            statistic = float(chi2_stat),
            p_value   = float(p_value),
            severity  = severity,
        )

    def _check_predictions(
        self,
        ref_preds: np.ndarray,
        cur_preds: np.ndarray,
    ) -> PredictionDriftResult:
        """Drift sur la distribution des prédictions."""
        psi       = _compute_psi(ref_preds, cur_preds)
        ks_stat, ks_pvalue = stats.ks_2samp(ref_preds, cur_preds)
        severity  = self._psi_severity_custom(psi)

        ref_mean = float(np.mean(ref_preds))
        cur_mean = float(np.mean(cur_preds))
        mean_shift_pct = (
            ((cur_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0.0
        )

        return PredictionDriftResult(
            psi           = psi,
            ks_statistic  = float(ks_stat),
            ks_p_value    = float(ks_pvalue),
            severity      = severity,
            ref_mean      = ref_mean,
            cur_mean      = cur_mean,
            mean_shift_pct= mean_shift_pct,
        )

    def _recommend(
        self,
        n_drifted: int,
        n_total: int,
        dataset_psi: float,
        pred_drift: Optional[PredictionDriftResult],
    ) -> str:
        """
        Logique de recommandation :
          "ok"      → tout stable
          "monitor" → warning, surveiller
          "retrain" → drift confirmé, retrainer le modèle
        """
        if pred_drift and pred_drift.has_drift:
            return "retrain"

        if n_total > 0 and n_drifted / n_total > 0.30:
            return "retrain"

        if dataset_psi > PSI_WARNING:
            return "retrain"

        if dataset_psi > PSI_STABLE or n_drifted > 0 or (pred_drift and pred_drift.has_warning):
            return "monitor"

        return "ok"

    def _psi_severity_custom(self, psi: float) -> str:
        if psi < self._psi_warn:
            return "stable"
        if psi < self._psi_drift:
            return "warning"
        return "drift"

    def _log_report(self, report: DriftReport) -> None:
        """Log structuré du rapport."""
        icon = {"ok": "PASS", "monitor": "WARNING", "retrain": "DRIFT"}.get(report.recommendation, "")

        logger.info(f"\n{icon} {report.summary()}")
        logger.info(f"\n{'='*55}")
        logger.info(f"{'Feature':<25} {'Test':<6} {'Statistic':>10} {'p-value':>10} {'Status'}")
        logger.info(f"{'='*55}")

        for r in sorted(report.feature_results, key=lambda x: x.statistic, reverse=True):
            status_icon = {"stable": "PASS", "warning": "WARNING", "drift": "DRIFT"}.get(r.severity, "")
            pval_str = f"{r.p_value:.4f}" if r.p_value is not None else "  N/A  "
            logger.info(
                f"  {r.feature:<23} {r.test_used:<6} {r.statistic:>10.4f} "
                f"{pval_str:>10} {status_icon} {r.severity}"
            )

        if report.prediction_drift:
            pd_r = report.prediction_drift
            pd_icon = {"stable": "PASS", "warning": "WARNING", "drift": "DRIFT"}.get(pd_r.severity, "")
            logger.info(f"{'='*55}")
            logger.info(
                f"  {'[PREDICTIONS]':<23} {'psi':<6} {pd_r.psi:>10.4f} "
                f"{pd_r.ks_p_value:>10.4f} {pd_icon} {pd_r.severity} "
                f"(shift: {pd_r.mean_shift_pct:+.1f}%)"
            )

        logger.info(f"{'='*55}")

        if report.recommendation == "retrain":
            logger.warning(
                f"\n  DRIFT DRIFT CONFIRMÉ: Retraining recommandé !\n"
                f"     Features driftées : {report.drifted_features()}\n"
                f"     PSI global        : {report.dataset_psi:.4f}\n"
                f"     Lancer : make train\n"
            )
        elif report.recommendation == "monitor":
            logger.warning(
                f"\n  WARNING DRIFT EN COURS: Surveiller de près\n"
                f"     Features en warning : {report.warning_features()}\n"
            )

    def trend(self) -> Optional[dict]:
        """
        Tendance du drift au fil du temps.
        Retourne None si moins de 2 rapports.
        """
        if len(self._reports) < 2:
            return None

        psi_history = [r.dataset_psi for r in self._reports]
        return {
            "n_reports"         : len(self._reports),
            "psi_history"       : [round(p, 4) for p in psi_history],
            "psi_trend"         : "increasing" if psi_history[-1] > psi_history[0] else "decreasing",
            "retrain_triggered" : sum(1 for r in self._reports if r.recommendation == "retrain"),
        }

    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """
        Met à jour les données de référence (après retraining).
        Appelé après chaque retraining réussi.
        """
        self.reference = new_reference.copy()
        logger.info(
            f"    Référence mise à jour → {len(new_reference)} lignes"
        )
