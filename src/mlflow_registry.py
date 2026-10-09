"""
mlflow_registry.py
------------------
MLflow Model Registry: إدارة lifecycle ديال الـ models.

الفرق بين Tracking والـ Registry:
  - Tracking : يسجّل كل run (metrics, params, artifacts)
  - Registry : يختار أحسن model ويحط فيه label (Staging / Production)

Lifecycle:
  [run enregistré] → register() → "None" → promote_to_staging()
       → "Staging" → promote_to_production() → "Production"
       → archive() → "Archived"

Usage:
    from mlflow_registry import MLflowRegistry

    registry = MLflowRegistry(model_name="avito-regression")

    version = registry.register(run_id, "reg/R2", higher_is_better=True)

    if registry.is_better_than_production(run_id, "reg/R2"):
        registry.promote_to_staging(version)
        registry.promote_to_production(version)

    model = registry.load_production_model()
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

from logger_setup import get_logger

logger = get_logger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("⚠️  mlflow non installé: Registry désactivé")

class Stage:
    NONE       = "None"
    STAGING    = "Staging"
    PRODUCTION = "Production"
    ARCHIVED   = "Archived"

class MLflowRegistry:
    """
    Gestion complète du Model Registry MLflow.

    Chaque instance gère un seul nom de modèle (ex: "avito-regression").
    Utiliser deux instances pour régression + classification.
    """

    def __init__(
        self,
        model_name: str,
        tracking_uri: Optional[str] = None,
    ):
        try:
            from config_loader import cfg
            self._uri = tracking_uri or cfg.paths.mlflow_uri
        except Exception:
            self._uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")

        self.model_name = model_name
        self._client: Optional[Any] = None

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self._uri)
            self._client = MlflowClient(tracking_uri=self._uri)
            logger.info(f"📋 MLflow Registry prêt: modèle : '{model_name}'")

    def _no_mlflow(self, method: str) -> bool:
        if not MLFLOW_AVAILABLE or self._client is None:
            logger.debug(f"   Registry.{method} ignoré (mlflow indisponible)")
            return True
        return False

    def register(
        self,
        run_id: str,
        artifact_path: str = "model",
        description: Optional[str] = None,
    ) -> Optional[str]:
        """
        Enregistre un run MLflow dans le Registry.

        Parameters
        ----------
        run_id        : ID du run MLflow (depuis MLflowTracker.run_id)
        artifact_path : chemin de l'artefact dans le run (ex: "regression")
        description   : description optionnelle de la version

        Returns
        -------
        version : str: numéro de version (ex: "3"), ou None si échec
        """
        if self._no_mlflow("register"):
            return None
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            result = mlflow.register_model(
                model_uri=model_uri,
                name=self.model_name,
            )
            version = result.version

            if description:
                self._client.update_model_version(
                    name=self.model_name,
                    version=version,
                    description=description,
                )

            self._client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key="registered_at",
                value=datetime.now().isoformat(),
            )

            logger.info(
                f"   📋 Modèle enregistré dans Registry → "
                f"'{self.model_name}' v{version}"
            )
            return version

        except MlflowException as exc:
            logger.warning(f"⚠️  Registry.register échoué : {type(exc).__name__}")
            return None

    def _transition(self, version: str, stage: str, archive_existing: bool = True) -> bool:
        """Transition générique vers un stage."""
        if self._no_mlflow("_transition"):
            return False
        try:
            self._client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing,
            )
            logger.info(
                f"   🔄 '{self.model_name}' v{version} → {stage}"
            )
            return True
        except MlflowException as exc:
            logger.warning(f"⚠️  Transition vers {stage} échouée : {type(exc).__name__}")
            return False

    def promote_to_staging(self, version: str) -> bool:
        """
        Passe la version en Staging.
        Staging = modèle validé, en cours de test avant production.
        """
        return self._transition(version, Stage.STAGING)

    def promote_to_production(self, version: str) -> bool:
        """
        Passe la version en Production.
        Archive automatiquement l'ancienne version production.
        """
        ok = self._transition(version, Stage.PRODUCTION, archive_existing=True)
        if ok:
            try:
                self._client.set_model_version_tag(
                    name=self.model_name,
                    version=version,
                    key="promoted_to_production_at",
                    value=datetime.now().isoformat(),
                )
            except Exception:
                pass
        return ok

    def archive(self, version: str) -> bool:
        """Archive une version (ne la supprime pas, juste la désactive)."""
        return self._transition(version, Stage.ARCHIVED, archive_existing=False)

    def demote_from_production(self, version: str) -> bool:
        """Retire un modèle de Production → Archived (rollback)."""
        logger.warning(f"   ⚠️  Rollback : '{self.model_name}' v{version} retiré de Production")
        return self.archive(version)

    def get_production_metric(self, metric_key: str) -> Optional[float]:
        """
        Retourne la valeur d'une métrique du modèle actuellement en Production.
        Retourne None si aucun modèle en Production.
        """
        if self._no_mlflow("get_production_metric"):
            return None
        try:
            versions = self._client.get_latest_versions(
                name=self.model_name,
                stages=[Stage.PRODUCTION],
            )
            if not versions:
                logger.info(f"   ℹ️  Aucun modèle en Production pour '{self.model_name}'")
                return None

            prod_version = versions[0]
            run = self._client.get_run(prod_version.run_id)
            value = run.data.metrics.get(metric_key)

            if value is None:
                logger.warning(f"   ⚠️  Métrique '{metric_key}' introuvable dans le run Production")
            return value

        except MlflowException as exc:
            logger.warning(f"⚠️  get_production_metric échoué : {type(exc).__name__}")
            return None

    def is_better_than_production(
        self,
        run_id: str,
        metric_key: str,
        higher_is_better: bool = True,
    ) -> bool:
        """
        Compare les métriques d'un nouveau run vs le modèle en Production.

        Returns True si :
          - Aucun modèle en Production (premier déploiement)
          - Le nouveau run est meilleur selon la métrique donnée

        Parameters
        ----------
        run_id           : ID du nouveau run à comparer
        metric_key       : clé de la métrique (ex: "reg/R2", "clf/F1")
        higher_is_better : True pour R², F1 / False pour MAE, RMSE
        """
        if self._no_mlflow("is_better_than_production"):
            return True  # fallback : toujours déployer si pas de MLflow

        try:
            new_run = self._client.get_run(run_id)
            new_value = new_run.data.metrics.get(metric_key)

            if new_value is None:
                logger.warning(f"   ⚠️  Métrique '{metric_key}' absente du run {run_id}")
                return False

            prod_value = self.get_production_metric(metric_key)

            if prod_value is None:
                logger.info(f"   ✅ Premier déploiement: pas de production à battre")
                return True

            is_better = new_value > prod_value if higher_is_better else new_value < prod_value
            symbol = ">" if higher_is_better else "<"

            logger.info(
                f"   📊 Comparaison '{metric_key}' : "
                f"nouveau={new_value:.4f} {symbol} production={prod_value:.4f} "
                f"→ {'✅ MEILLEUR' if is_better else '❌ MOINS BON'}"
            )
            return is_better

        except MlflowException as exc:
            logger.warning(f"⚠️  is_better_than_production échoué : {type(exc).__name__}")
            return False

    def load_production_model(self) -> Optional[Any]:
        """
        Charge le modèle sklearn actuellement en Production depuis le Registry.

        Usage dans l'API :
            model = registry.load_production_model()
            if model:
                pred = model.predict(X)
        """
        if self._no_mlflow("load_production_model"):
            return None
        try:
            model_uri = f"models:/{self.model_name}/{Stage.PRODUCTION}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"   ✅ Modèle Production chargé : '{self.model_name}'")
            return model
        except MlflowException as exc:
            logger.warning(f"⚠️  load_production_model échoué : {type(exc).__name__}")
            return None

    def load_staging_model(self) -> Optional[Any]:
        """Charge le modèle en Staging (pour validation)."""
        if self._no_mlflow("load_staging_model"):
            return None
        try:
            model_uri = f"models:/{self.model_name}/{Stage.STAGING}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"   ✅ Modèle Staging chargé : '{self.model_name}'")
            return model
        except MlflowException as exc:
            logger.warning(f"⚠️  load_staging_model échoué : {type(exc).__name__}")
            return None

    def load_version(self, version: str) -> Optional[Any]:
        """Charge une version spécifique (pour A/B testing ou rollback)."""
        if self._no_mlflow("load_version"):
            return None
        try:
            model_uri = f"models:/{self.model_name}/{version}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"   ✅ Modèle v{version} chargé : '{self.model_name}'")
            return model
        except MlflowException as exc:
            logger.warning(f"⚠️  load_version({version}) échoué : {type(exc).__name__}")
            return None

    def get_latest_versions(self, stages: Optional[list[str]] = None) -> list[dict]:
        """
        Retourne les dernières versions par stage.

        Returns liste de dicts avec : version, stage, run_id, description, created_at
        """
        if self._no_mlflow("get_latest_versions"):
            return []
        try:
            stages = stages or [Stage.NONE, Stage.STAGING, Stage.PRODUCTION, Stage.ARCHIVED]
            versions = self._client.get_latest_versions(
                name=self.model_name,
                stages=stages,
            )
            return [
                {
                    "version"    : v.version,
                    "stage"      : v.current_stage,
                    "run_id"     : v.run_id,
                    "description": v.description,
                    "created_at" : datetime.fromtimestamp(v.creation_timestamp / 1000).isoformat(),
                    "tags"       : dict(v.tags),
                }
                for v in versions
            ]
        except MlflowException as exc:
            logger.warning(f"⚠️  get_latest_versions échoué : {type(exc).__name__}")
            return []

    def get_production_info(self) -> Optional[dict]:
        """Retourne les infos du modèle actuellement en Production."""
        versions = self.get_latest_versions(stages=[Stage.PRODUCTION])
        return versions[0] if versions else None

    def print_summary(self) -> None:
        """Affiche un résumé du Registry dans les logs."""
        if self._no_mlflow("print_summary"):
            return
        versions = self.get_latest_versions()
        logger.info(f"\n{'='*55}")
        logger.info(f"   📋 Registry : '{self.model_name}'")
        logger.info(f"{'='*55}")
        if not versions:
            logger.info("   Aucune version enregistrée.")
        for v in versions:
            stage_icon = {
                Stage.PRODUCTION: "🟢",
                Stage.STAGING   : "🟡",
                Stage.ARCHIVED  : "⚫",
                Stage.NONE      : "⚪",
            }.get(v["stage"], "❓")
            logger.info(
                f"   {stage_icon} v{v['version']:>3} | {v['stage']:<12} | "
                f"run={v['run_id'][:8]}... | {v['created_at'][:10]}"
            )
        logger.info(f"{'='*55}\n")

def auto_register_and_promote(
    run_id: str,
    model_name: str,
    artifact_path: str,
    primary_metric: str,
    higher_is_better: bool = True,
    tracking_uri: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """
    Workflow complet : register → compare → promote si meilleur.

    Appelée automatiquement depuis pipeline.py après chaque training.

    Returns dict avec : version, promoted, reason
    """
    if not MLFLOW_AVAILABLE:
        return {"version": None, "promoted": False, "reason": "mlflow unavailable"}

    registry = MLflowRegistry(model_name=model_name, tracking_uri=tracking_uri)

    version = registry.register(
        run_id=run_id,
        artifact_path=artifact_path,
        description=description or f"Auto-registered | run={run_id[:8]}",
    )
    if version is None:
        return {"version": None, "promoted": False, "reason": "registration failed"}

    is_better = registry.is_better_than_production(
        run_id=run_id,
        metric_key=primary_metric,
        higher_is_better=higher_is_better,
    )

    if is_better:
        registry.promote_to_staging(version)
        registry.promote_to_production(version)
        logger.info(f"   🚀 '{model_name}' v{version} promu en Production !")
        registry.print_summary()
        return {
            "version"  : version,
            "promoted" : True,
            "reason"   : f"New {primary_metric} is better than production",
        }
    else:
        logger.info(
            f"   ℹ️  '{model_name}' v{version} enregistré en Staging "
            f"(pas meilleur que Production)"
        )
        registry.promote_to_staging(version)
        registry.print_summary()
        return {
            "version"  : version,
            "promoted" : False,
            "reason"   : f"New {primary_metric} did not beat production",
        }
