"""
mlflow_tracking.py
------------------
Intégration MLflow pour le tracking des expériences, métriques et paramètres.

Chaque run du pipeline est enregistré avec :
  - Paramètres   : options du pipeline, hyperparamètres du modèle
  - Métriques    : R², MAE, RMSE, MAPE, F1, Accuracy, ROC-AUC
  - Artefacts    : modèles .pkl, plots, rapport HTML
  - Tags         : version, date, nom du modèle

Usage :
    from mlflow_tracking import MLflowTracker
    with MLflowTracker() as tracker:
        tracker.log_params({...})
        tracker.log_metrics({...})
        tracker.log_model(model, "regression")
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from logger_setup import get_logger

logger = get_logger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("⚠️  mlflow non installé: tracking désactivé (pip install mlflow)")

class MLflowTracker:
    """
    Wrapper MLflow avec fallback silencieux si mlflow n'est pas installé.
    Toutes les méthodes sont no-op si MLflow est indisponible.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[dict] = None,
    ):
        try:
            from config_loader import cfg
            self._experiment = experiment_name or cfg.mlflow.experiment_name
            self._uri = tracking_uri or cfg.paths.mlflow_uri
            self._tags = tags or dict(cfg.mlflow.tags)
        except Exception:
            self._experiment = experiment_name or "avito-real-estate"
            self._uri = tracking_uri or "mlruns"
            self._tags = tags or {}

        self._run_name = run_name
        self._run = None
        self._active = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end(success=(exc_type is None))
        return False  # ne supprime pas les exceptions

    def start(self) -> None:
        if not MLFLOW_AVAILABLE:
            return
        try:
            mlflow.set_tracking_uri(self._uri)
            mlflow.set_experiment(self._experiment)
            self._run = mlflow.start_run(
                run_name=self._run_name,
                tags=self._tags,
            )
            self._active = True
            logger.info(f"🔬 MLflow run démarré: ID : {self._run.info.run_id}")
        except Exception as exc:
            logger.warning(f"⚠️  MLflow start échoué : {type(exc).__name__}")

    def end(self, success: bool = True) -> None:
        if not MLFLOW_AVAILABLE or not self._active:
            return
        try:
            status = "FINISHED" if success else "FAILED"
            mlflow.end_run(status=status)
            self._active = False
            logger.info(f"🔬 MLflow run terminé [{status}]")
        except Exception as exc:
            logger.warning(f"⚠️  MLflow end échoué : {type(exc).__name__}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Enregistre les paramètres du pipeline (hyperparamètres, options)."""
        if not MLFLOW_AVAILABLE or not self._active:
            return
        try:
            safe = {k: str(v)[:500] for k, v in params.items()}
            mlflow.log_params(safe)
            logger.debug(f"   MLflow params : {list(safe.keys())}")
        except Exception as exc:
            logger.warning(f"⚠️  mlflow.log_params échoué : {type(exc).__name__}")

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Enregistre les métriques de performance."""
        if not MLFLOW_AVAILABLE or not self._active:
            return
        try:
            clean = {k: float(v) for k, v in metrics.items() if v is not None}
            mlflow.log_metrics(clean, step=step)
            logger.debug(f"   MLflow metrics : {list(clean.keys())}")
        except Exception as exc:
            logger.warning(f"⚠️  mlflow.log_metrics échoué : {type(exc).__name__}")

    def log_model(self, model: Any, artifact_name: str) -> None:
        """Enregistre un modèle sklearn comme artefact MLflow."""
        if not MLFLOW_AVAILABLE or not self._active:
            return
        try:
            mlflow.sklearn.log_model(model, artifact_name)
            logger.info(f"   📦 Modèle MLflow enregistré → {artifact_name}")
        except Exception as exc:
            logger.warning(f"⚠️  mlflow.log_model échoué : {type(exc).__name__}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Enregistre un fichier (plot, rapport, CSV) comme artefact."""
        if not MLFLOW_AVAILABLE or not self._active:
            return
        if not Path(local_path).exists():
            logger.warning("MLflow artifact file not found")
            return
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"   MLflow artefact : {local_path}")
        except Exception as exc:
            logger.warning(f"⚠️  mlflow.log_artifact échoué : {type(exc).__name__}")

    def log_artifacts_dir(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Enregistre tous les fichiers d'un dossier comme artefacts."""
        if not MLFLOW_AVAILABLE or not self._active:
            return
        if not Path(local_dir).exists():
            return
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"   📁 Dossier MLflow : {local_dir}")
        except Exception as exc:
            logger.warning(f"⚠️  mlflow.log_artifacts échoué : {type(exc).__name__}")

    def set_tag(self, key: str, value: str) -> None:
        """Ajoute un tag au run courant."""
        if not MLFLOW_AVAILABLE or not self._active:
            return
        try:
            mlflow.set_tag(key, str(value))
        except Exception as exc:
            logger.warning(f"⚠️  mlflow.set_tag échoué : {type(exc).__name__}")

    def log_pipeline_params(self, options: dict, data_info: dict) -> None:
        """Raccourci pour logger tous les paramètres pipeline en une fois."""
        params = {
            "pipeline.test_size"     : options.get("test_size"),
            "pipeline.random_state"  : options.get("random_state"),
            "pipeline.use_smote"     : options.get("use_smote"),
            "pipeline.use_log_target": options.get("use_log_target"),
            "pipeline.optimize"      : options.get("optimize"),
            "data.n_train"           : data_info.get("n_train"),
            "data.n_test"            : data_info.get("n_test"),
            "data.n_features"        : data_info.get("n_features"),
        }
        self.log_params(params)

    def log_regression_results(self, metrics: dict, model_name: str) -> None:
        """Log métriques régression avec préfixe 'reg/'."""
        self.set_tag("regression.model", model_name)
        prefixed = {f"reg/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed)

    def log_classification_results(self, metrics: dict, model_name: str) -> None:
        """Log métriques classification avec préfixe 'clf/'."""
        self.set_tag("classification.model", model_name)
        prefixed = {f"clf/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed)

    def log_baseline_results(self, reg_baselines: dict, clf_baselines: dict | None) -> None:
        """Log les métriques des baselines pour comparaison dans MLflow."""
        for name, m in reg_baselines.items():
            prefixed = {f"baseline_reg/{name}/{k}": v for k, v in m.items()}
            self.log_metrics(prefixed)
        if clf_baselines:
            for name, m in clf_baselines.items():
                prefixed = {f"baseline_clf/{name}/{k}": v for k, v in m.items()}
                self.log_metrics(prefixed)

    @property
    def run_id(self) -> Optional[str]:
        """Retourne l'ID du run MLflow courant."""
        if self._run:
            return self._run.info.run_id
        return None
