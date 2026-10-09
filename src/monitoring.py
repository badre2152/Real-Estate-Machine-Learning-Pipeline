"""
monitoring.py
-------------
Monitoring du pipeline ML : durée d'exécution, métriques par étape,
alertes sur les seuils de performance.

Fonctionnalités :
  - Chronomètre par étape (context manager)
  - Résumé des temps d'exécution
  - Alertes si R² ou F1 < seuils configurés
  - Export JSON des métriques de monitoring

Usage :
    from monitoring import PipelineMonitor
    monitor = PipelineMonitor()

    with monitor.step("data_loading"):
        df = load_data()

    with monitor.step("training"):
        model.fit(X_train, y_train)

    monitor.check_regression_alert(r2=0.72)
    monitor.print_summary()
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from logger_setup import get_logger

logger = get_logger(__name__)

@dataclass
class StepTiming:
    """Durée d'une étape du pipeline."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    error_msg: Optional[str] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def duration_str(self) -> str:
        d = self.duration_s
        if d is None:
            return "en cours..."
        if d < 60:
            return f"{d:.2f}s"
        return f"{d/60:.1f}min"

@dataclass
class MetricAlert:
    """Alerte déclenchée quand une métrique passe sous un seuil."""
    metric_name: str
    value: float
    threshold: float
    severity: str = "warning"

    @property
    def message(self) -> str:
        return (
            f"{self.metric_name}={self.value:.4f} < seuil={self.threshold:.4f}"
        )

class PipelineMonitor:
    """
    Moniteur de pipeline : temps d'exécution + alertes métriques.
    Thread-safe pour une utilisation dans des pipelines séquentiels.
    """

    def __init__(self, output_dir: str | None = None):
        try:
            from config_loader import cfg
            self.alert_r2  = float(cfg.monitoring.alert_r2_threshold)
            self.alert_f1  = float(cfg.monitoring.alert_f1_threshold)
            self.log_steps = bool(cfg.monitoring.log_step_times)
            self.output_dir = output_dir or cfg.paths.reports_dir
        except Exception:
            self.alert_r2  = 0.50
            self.alert_f1  = 0.50
            self.log_steps = True
            self.output_dir = output_dir or "reports/runtime"

        self._steps: list[StepTiming] = []
        self._alerts: list[MetricAlert] = []
        self._pipeline_start = time.perf_counter()
        self._pipeline_name  = "ML Pipeline"

    @contextmanager
    def step(self, name: str) -> Generator:
        """
        Context manager qui chronomètre une étape du pipeline.

        Usage:
            with monitor.step("feature_engineering"):
                X = build_features(df)
        """
        timing = StepTiming(name=name, start_time=time.perf_counter())
        self._steps.append(timing)

        if self.log_steps:
            logger.info(f"   ⏱  [{name}] démarré ...")

        try:
            yield timing
            timing.end_time = time.perf_counter()
            timing.success  = True
            if self.log_steps:
                logger.info(f"   ✅ [{name}] terminé en {timing.duration_str}")

        except Exception as exc:
            timing.end_time  = time.perf_counter()
            timing.success   = False
            timing.error_msg = str(exc)
            logger.error(f"   ❌ [{name}] ERREUR après {timing.duration_str} : {exc}")
            raise

    def check_regression_alert(self, r2: float, mae: Optional[float] = None) -> None:
        """
        Vérifie les métriques de régression et lève des alertes si nécessaire.

        Args:
            r2:  Coefficient de détermination R².
            mae: Erreur absolue moyenne (optionnel, pas de seuil défini).
        """
        if r2 < self.alert_r2:
            alert = MetricAlert("R2", r2, self.alert_r2, severity="warning")
            self._alerts.append(alert)
            logger.warning(f"   🔔 ALERTE RÉGRESSION : {alert.message}")
        else:
            logger.info(f"   ✅ R²={r2:.4f} ≥ seuil {self.alert_r2}: OK")

    def check_classification_alert(self, f1: float, accuracy: Optional[float] = None) -> None:
        """
        Vérifie les métriques de classification et lève des alertes si nécessaire.
        """
        if f1 < self.alert_f1:
            alert = MetricAlert("F1", f1, self.alert_f1, severity="warning")
            self._alerts.append(alert)
            logger.warning(f"   🔔 ALERTE CLASSIFICATION : {alert.message}")
        else:
            logger.info(f"   ✅ F1={f1:.4f} ≥ seuil {self.alert_f1}: OK")

    def print_summary(self) -> None:
        """Affiche un récapitulatif des temps d'exécution de toutes les étapes."""
        total = time.perf_counter() - self._pipeline_start

        logger.info("\n" + "=" * 60)
        logger.info("📊 MONITORING: RÉSUMÉ D'EXÉCUTION")
        logger.info("=" * 60)

        if not self._steps:
            logger.info("   Aucune étape enregistrée.")
            return

        max_name = max(len(s.name) for s in self._steps)

        for s in self._steps:
            status = "✅" if s.success else "❌"
            dur    = s.duration_s or 0
            pct    = dur / total * 100 if total > 0 else 0
            bar    = "█" * int(pct / 5)
            logger.info(
                f"   {status} {s.name:<{max_name}}  {s.duration_str:>8}  "
                f"({pct:4.1f}%)  {bar}"
            )

        logger.info(f"\n   ⏱  Total pipeline : {_format_duration(total)}")

        if self._alerts:
            logger.info(f"\n   🔔 {len(self._alerts)} alerte(s) déclenchée(s) :")
            for a in self._alerts:
                logger.warning(f"      [{a.severity.upper()}] {a.message}")
        else:
            logger.info("   ✅ Aucune alerte: toutes les métriques dans les seuils")

        logger.info("=" * 60)

    def save_report(self) -> str:
        """Sauvegarde le rapport de monitoring en JSON."""
        os.makedirs(self.output_dir, exist_ok=True)
        total = time.perf_counter() - self._pipeline_start

        report = {
            "pipeline"      : self._pipeline_name,
            "timestamp"     : datetime.now().isoformat(),
            "total_duration_s": round(total, 3),
            "steps"         : [
                {
                    "name"      : s.name,
                    "duration_s": round(s.duration_s, 3) if s.duration_s else None,
                    "success"   : s.success,
                    "error"     : s.error_msg,
                }
                for s in self._steps
            ],
            "alerts": [
                {
                    "metric"   : a.metric_name,
                    "value"    : round(a.value, 4),
                    "threshold": a.threshold,
                    "severity" : a.severity,
                }
                for a in self._alerts
            ],
        }

        path = os.path.join(self.output_dir, "monitoring_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"   💾 Monitoring → {path}")
        return path

    @property
    def total_duration_s(self) -> float:
        return time.perf_counter() - self._pipeline_start

    @property
    def has_alerts(self) -> bool:
        return len(self._alerts) > 0

    @property
    def all_steps_succeeded(self) -> bool:
        return all(s.success for s in self._steps)

    def get_step_durations(self) -> dict[str, float]:
        """Retourne un dict {step_name: duration_s} pour MLflow."""
        return {
            f"timing/{s.name}_s": round(s.duration_s, 3)
            for s in self._steps
            if s.duration_s is not None
        }

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}min {s:.0f}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}min {s:.0f}s"
