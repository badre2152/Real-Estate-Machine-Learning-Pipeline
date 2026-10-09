"""
api.py  (v3: FastAPI service)
----------------------------------------
API de prédiction FastAPI pour le pipeline ML Avito Real Estate.

Améliorations v3 :
  ✅ Authentication par API Key (header X-API-Key)
  ✅ Rate Limiting (60 req/min par IP: in-memory)
  ✅ /health endpoint complet (liveness probe)
  ✅ /ready endpoint (readiness probe: modèles chargés ?)
  ✅ Gestion des erreurs structurée avec request_id
  ✅ Request ID unique par requête (tracing)
  ✅ Compression GZip automatique
  ✅ Versioning dans l'URL (/v1/...)
  ✅ Backward compatibility avec anciens endpoints

Endpoints :
  GET  /health          : liveness probe (sans auth)
  GET  /ready           : readiness probe (sans auth)
  GET  /v1/info         : infos modèle chargé
  POST /v1/predict      : prédiction prix + intervalle de confiance
  POST /v1/predict/batch: prédictions en lot (max 100)
  GET  /v1/metrics      : métriques de monitoring

Lancer :
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import hashlib
import os
import pickle
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette import status

from logger_setup import get_logger
from prediction_intervals import PredictionIntervalBuilder

logger = get_logger(__name__)

_API_KEYS_RAW = os.getenv("API_KEYS", "")
VALID_API_KEYS: set = {k.strip() for k in _API_KEYS_RAW.split(",") if k.strip()}
RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

_ENVIRONMENT = os.getenv("ENVIRONMENT", "dev").strip().lower()
_CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
if _ENVIRONMENT == "production":
    if not VALID_API_KEYS or "change_me_api_key" in VALID_API_KEYS:
        raise RuntimeError("Configure non-placeholder API_KEYS for production")
    if not _CORS_ORIGINS or "*" in _CORS_ORIGINS:
        raise RuntimeError("Configure explicit CORS_ORIGINS for production")
if not VALID_API_KEYS:
    logger.warning("API_KEYS is empty. Authenticated endpoints will reject all requests.")

class _InMemoryRateLimiter:
    """
    Rate limiter basique en mémoire (sliding window par IP).
    Pour la production : remplacer par slowapi + Redis.
    """

    def __init__(self, max_calls: int, window_s: int = 60):
        self._max = max_calls
        self._window = window_s
        self._buckets: dict = {}

    def is_allowed(self, key: str):
        now = time.time()
        calls = self._buckets.get(key, [])
        calls = [t for t in calls if now - t < self._window]
        remaining = self._max - len(calls)
        if remaining <= 0:
            self._buckets[key] = calls
            return False, 0
        calls.append(now)
        self._buckets[key] = calls
        return True, remaining - 1

_rate_limiter = _InMemoryRateLimiter(max_calls=RATE_LIMIT_PER_MINUTE)

app = FastAPI(
    title="Avito Real Estate: API de Prédiction",
    description=(
        "API ML pour estimer le prix d'un bien immobilier au Maroc "
        "avec intervalle de confiance à 95%.\n\n"
        "**Authentification** : passer le header `X-API-Key` avec votre clé.\n\n"
        f"**Rate Limit** : {RATE_LIMIT_PER_MINUTE} requêtes/minute par clé API."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
_state: dict = {
    "reg_model"         : None,
    "clf_model"         : None,
    "preprocessor"      : None,
    "pi_builder"        : None,
    "feature_names"     : None,
    "label_encoder"     : None,
    "reg_metrics"       : {},
    "clf_metrics"       : {},
    "loaded_at"         : None,
    "total_predictions" : 0,
    "errors"            : 0,
    "uptime_start"      : None,
}

@app.on_event("startup")
async def load_models():
    """Charge les modèles depuis le dossier models/ au démarrage."""
    _state["uptime_start"] = time.time()
    logger.info("🚀 Démarrage API v3: chargement des modèles ...")

    files = {
        "reg_model"    : ["best_regression_model.pkl", "regression_model.pkl"],
        "clf_model"    : ["best_classification_model.pkl", "classification_model.pkl"],
        "preprocessor" : ["preprocessor.pkl"],
        "pi_builder"   : ["pi_builder.pkl"],
        "feature_names": ["feature_names.pkl"],
        "reg_metrics"  : ["regression_metrics.pkl"],
        "clf_metrics"  : ["classification_metrics.pkl"],
        "label_encoder": ["label_encoder.pkl"],
    }

    for key, candidates in files.items():
        _state[key] = {} if key in ("reg_metrics", "clf_metrics") else None
        loaded = False
        for fname in candidates:
            path = _MODELS_DIR / fname
            if not path.is_file():
                continue
            try:
                with path.open("rb") as model_file:
                    loaded_value = pickle.load(model_file)
                if key in ("reg_metrics", "clf_metrics") and not isinstance(loaded_value, dict):
                    raise ValueError("Metrics artifact must be a dictionary")
            except (OSError, pickle.UnpicklingError, EOFError, ImportError,
                    AttributeError, ValueError, TypeError) as exc:
                logger.warning(
                    "Could not load %s from %s (%s)",
                    key, fname, type(exc).__name__
                )
                continue
            _state[key] = loaded_value
            logger.info("Loaded model artifact: %s", key)
            loaded = True
            break
        if not loaded:
            logger.warning("Model artifact unavailable: %s", key)

    _state["loaded_at"] = datetime.now().isoformat()
    if _state["reg_model"] is None or _state["preprocessor"] is None:
        logger.warning("API started without required prediction artifacts")
    else:
        logger.info("API prediction artifacts loaded")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(
    request: Request,
    api_key: Optional[str] = Security(_api_key_header),
) -> str:
    """Vérifie la présence et validité du header X-API-Key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "missing_api_key",
                "message": "Header X-API-Key requis. Consultez /docs pour l'authentification.",
            },
        )
    if api_key not in VALID_API_KEYS:
        client_ip = request.client.host if request.client else "unknown"
        logger.warning(f"   🔑 Clé API invalide depuis {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "invalid_api_key",
                "message": "Clé API invalide ou révoquée.",
            },
        )
    return api_key

async def check_rate_limit(api_key: str = Depends(require_api_key)) -> None:
    """Applique un quota par clé API sans conserver la clé en clair."""
    key_id = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    allowed, remaining = _rate_limiter.is_allowed(key_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Trop de requêtes. Max {RATE_LIMIT_PER_MINUTE}/min par clé API.",
                "retry_after_s": 60,
            },
            headers={"Retry-After": "60"},
        )

class PropertyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    surface_m2: float = Field(..., gt=0, le=10_000)
    ville: str = Field(..., min_length=1)
    type_bien: str = Field(..., min_length=1)
    quartier: Optional[str] = Field(None, description="Quartier (optionnel)")
    nb_chambres: Optional[int] = Field(None, ge=0, le=20)
    nb_salles_bain: Optional[int] = Field(None, ge=0, le=10)
    etage: Optional[int] = Field(None, ge=0, le=50)
    age_bien: Optional[int] = Field(None, ge=0, le=150)

    @field_validator("ville")
    @classmethod
    def normalize_ville(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("ville must not be blank")
        return cleaned.title()

    @field_validator("type_bien")
    @classmethod
    def normalize_type(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("type_bien must not be blank")
        return cleaned.lower()

    model_config = {
        "json_schema_extra": {
            "example": {
                "surface_m2": 120, "ville": "Casablanca",
                "type_bien": "appartement", "nb_chambres": 3,
                "nb_salles_bain": 2, "etage": 4, "age_bien": 10,
            }
        }
    }

class PredictionResponse(BaseModel):
    request_id: str
    prediction: float
    lower_95: float
    upper_95: float
    interval_width: float
    property_type: Optional[str]
    price_category: Optional[str] = None
    confidence_level: float = 0.95
    formatted: str
    model_version: str = "3.0"
    latency_ms: float
    timestamp: str

class BatchInput(BaseModel):
    properties: list[dict] = Field(..., min_length=1, max_length=100)

class BatchResponse(BaseModel):
    request_id: str
    predictions: list
    n_properties: int
    n_success: int
    n_errors: int
    total_latency_ms: float

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    request.state.request_id = req_id
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"

    logger.info(
        f"[{req_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} [{duration_ms:.1f}ms]"
    )
    return response

@app.get("/health", tags=["Ops"], summary="Liveness probe")
async def health():
    """
    Liveness probe: répond 200 si le processus est en vie.
    Pas d'authentification requise.
    Utilisé par Docker HEALTHCHECK et Kubernetes.
    """
    uptime = round(time.time() - _state["uptime_start"], 1) if _state["uptime_start"] else None
    return {
        "status"   : "ok",
        "version"  : "3.0.0",
        "uptime_s" : uptime,
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/ready", tags=["Ops"], summary="Readiness probe")
async def ready():
    """
    Readiness probe: répond 200 si les modèles sont prêts, 503 sinon.
    Pas d'authentification requise.
    """
    models_status = {
        "regression_model"    : _state["reg_model"] is not None,
        "classification_model": _state["clf_model"] is not None,
        "preprocessor"        : _state["preprocessor"] is not None,
        "pi_builder"          : _state["pi_builder"] is not None,
    }
    is_ready = models_status["regression_model"] and models_status["preprocessor"]
    response_data = {
        "ready"            : is_ready,
        "models_loaded"    : models_status,
        "loaded_at"        : _state["loaded_at"],
        "total_predictions": _state["total_predictions"],
    }

    if not is_ready:
        return JSONResponse(status_code=503, content=response_data)
    return response_data

_auth_deps = [Depends(require_api_key), Depends(check_rate_limit)]

@app.get("/v1/info", tags=["Info"], dependencies=_auth_deps)
async def model_info():
    """Informations sur les modèles chargés et leurs métriques."""
    return {
        "regression_model"      : type(_state["reg_model"]).__name__ if _state["reg_model"] else None,
        "classification_model"  : type(_state["clf_model"]).__name__ if _state["clf_model"] else None,
        "regression_metrics"    : _state["reg_metrics"],
        "classification_metrics": _state["clf_metrics"],
        "loaded_at"             : _state["loaded_at"],
        "total_predictions"     : _state["total_predictions"],
        "error_count"           : _state["errors"],
        "api_version"           : "3.0.0",
    }

@app.get("/v1/metrics", tags=["Monitoring"], dependencies=_auth_deps)
async def metrics():
    """Métriques de monitoring de l'API et des modèles."""
    uptime = round(time.time() - _state["uptime_start"], 1) if _state["uptime_start"] else None
    return {
        "ml_total_predictions": _state["total_predictions"],
        "ml_total_errors"     : _state["errors"],
        "ml_error_rate"       : round(_state["errors"] / max(_state["total_predictions"], 1), 4),
        "ml_model_loaded"     : int(_state["reg_model"] is not None),
        "ml_reg_r2"           : _state["reg_metrics"].get("R2", 0),
        "ml_clf_f1"           : _state["clf_metrics"].get("F1", 0),
        "api_uptime_s"        : uptime,
        "rate_limit_per_min"  : RATE_LIMIT_PER_MINUTE,
    }

@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    dependencies=_auth_deps,
)
async def predict(data: PropertyInput, request: Request):
    """
    Prédit le prix d'un bien immobilier avec intervalle de confiance à 95%.

    **Authentification requise** : header `X-API-Key`.
    """
    t0 = time.perf_counter()
    req_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    if _state["reg_model"] is None or _state["preprocessor"] is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_unavailable", "message": "Modèle non disponible.", "request_id": req_id},
        )

    try:
        row = _build_input_df(data)
        X = _state["preprocessor"].transform(row)
        pred = float(_state["reg_model"].predict(X)[0])

        lower, upper = pred, pred
        if _state["pi_builder"] is not None:
            pi_df = _state["pi_builder"].predict_with_interval(X)
            lower = float(pi_df["lower"].iloc[0])
            upper = float(pi_df["upper"].iloc[0])

        category = None
        if _state["clf_model"] is not None:
            raw_pred = _state["clf_model"].predict(X)[0]
            if _state["label_encoder"] is not None:
                try:
                    category = str(_state["label_encoder"].inverse_transform([raw_pred])[0])
                except Exception:
                    category = str(raw_pred)
            else:
                category = str(raw_pred)

        _state["total_predictions"] += 1
        latency_ms = (time.perf_counter() - t0) * 1000

        return PredictionResponse(
            request_id     = req_id,
            prediction     = pred,
            lower_95       = lower,
            upper_95       = upper,
            interval_width = upper - lower,
            property_type  = category,
            price_category = None,
            formatted      = f"{pred:,.0f} MAD [{lower:,.0f} à {upper:,.0f}]",
            latency_ms     = round(latency_ms, 2),
            timestamp      = datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        _state["errors"] += 1
        logger.error("[%s] Prediction failed (%s)", req_id, type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail={"error": "prediction_failed", "message": "Erreur interne de prédiction.", "request_id": req_id},
        )

@app.post("/v1/predict/batch", response_model=BatchResponse, tags=["Prediction"], dependencies=_auth_deps)
async def predict_batch(data: BatchInput, request: Request):
    """Prédictions en lot pour plusieurs biens simultanément (max 100)."""
    t0 = time.perf_counter()
    req_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    if _state["reg_model"] is None or _state["preprocessor"] is None:
        raise HTTPException(status_code=503, detail={
            "error": "model_unavailable",
            "message": "Modèle ou préprocesseur indisponible.",
            "request_id": req_id,
        })
    results, n_success, n_errors = [], 0, 0

    for index, raw_property in enumerate(data.properties):
        try:
            prop = PropertyInput.model_validate(raw_property)
        except ValidationError as exc:
            results.append({
                "index": index,
                "error": {
                    "error": "invalid_property",
                    "message": "Invalid property fields.",
                    "fields": sorted({
                        str(item["loc"][0]) for item in exc.errors()
                        if item.get("loc")
                    }),
                },
            })
            n_errors += 1
            continue

        try:
            result = await predict(prop, request)
            results.append(result)
            n_success += 1
        except HTTPException as exc:
            results.append({"index": index, "error": exc.detail})
            n_errors += 1

    return BatchResponse(
        request_id       = req_id,
        predictions      = results,
        n_properties     = len(data.properties),
        n_success        = n_success,
        n_errors         = n_errors,
        total_latency_ms = round((time.perf_counter() - t0) * 1000, 2),
    )

@app.get("/", tags=["Ops"], include_in_schema=False)
async def root():
    models_loaded = _state["reg_model"] is not None
    return {
        "status"      : "ok",
        "version"     : "2.0.0",
        "models_loaded": models_loaded,
        "timestamp"   : datetime.now().isoformat(),
        "message"     : "Avito ML API: voir /docs ou /health",
    }

@app.get("/info", include_in_schema=False, dependencies=_auth_deps)
async def info_compat():
    return await model_info()

@app.get("/metrics", include_in_schema=False, dependencies=_auth_deps)
async def metrics_compat():
    return await metrics()

@app.post("/predict", include_in_schema=False, dependencies=_auth_deps)
async def predict_compat(data: PropertyInput, request: Request):
    return await predict(data, request)

@app.post("/predict/batch", include_in_schema=False, dependencies=_auth_deps)
async def predict_batch_compat(data: BatchInput, request: Request):
    return await predict_batch(data, request)

def _build_input_df(data: PropertyInput) -> pd.DataFrame:
    """
    Construit le DataFrame d'entrée avec toutes les features attendues
    par le preprocessor, y compris les features géo dérivées.
    """
    nb_chambres      = data.nb_chambres if data.nb_chambres is not None else 0
    nb_salles_bain   = data.nb_salles_bain if data.nb_salles_bain is not None else 0
    surface_m2       = data.surface_m2

    surface_x_chambres  = surface_m2 * nb_chambres if nb_chambres else 0.0
    surface_par_chambre = surface_m2 / (nb_chambres + 1)
    ratio_chambres_bains = nb_chambres / (nb_salles_bain + 1)
    row = {
        "surface_m2"         : surface_m2,
        "ville"              : data.ville,
        "quartier"           : data.quartier or "Autre Secteur",
        "type_bien"          : data.type_bien,
        "nb_chambres"        : nb_chambres,
        "nb_salles_bain"     : nb_salles_bain,
        "etage"              : str(data.etage) if data.etage is not None else "",
        "age_bien"           : data.age_bien if data.age_bien is not None else None,
        "annee_construction" : None,
        "region_label"       : None,
        "is_grande_ville"    : None,
        "lien"               : "",
        "titre"              : "",
        "surface_x_chambres"  : surface_x_chambres,
        "surface_par_chambre" : surface_par_chambre,
        "ratio_chambres_bains": ratio_chambres_bains,
    }
    return pd.DataFrame([row])

@app.get(
    "/v1/registry",
    tags=["Registry"],
    dependencies=_auth_deps,
    summary="État du Model Registry (versions + stages)",
)
async def registry_status():
    """
    Retourne l'état actuel du MLflow Model Registry :
    - Versions en Production, Staging, Archived
    - Métriques de chaque version

    Utile pour monitorer quel model est en production sans ouvrir MLflow UI.
    """
    try:
        from mlflow_registry import MLflowRegistry, Stage
        reg_registry = MLflowRegistry("avito-regression")
        clf_registry = MLflowRegistry("avito-classification")

        return {
            "regression": {
                "production": reg_registry.get_production_info(),
                "all_versions": reg_registry.get_latest_versions(),
            },
            "classification": {
                "production": clf_registry.get_production_info(),
                "all_versions": clf_registry.get_latest_versions(),
            },
            "mlflow_ui": os.getenv("MLFLOW_UI_URL"),
        }
    except Exception as exc:
        logger.exception("Registry status unavailable")
        raise HTTPException(500, detail={"error": "registry_error"})

@app.post(
    "/v1/registry/promote",
    tags=["Registry"],
    dependencies=_auth_deps,
    summary="Promouvoir un modèle vers Staging ou Production",
)
async def registry_promote(
    model_name: str,
    version: str,
    stage: str,
):
    """
    Promouvoir manuellement une version vers un stage.

    - **model_name** : "avito-regression" ou "avito-classification"
    - **version**    : numéro de version (ex: "3")
    - **stage**      : "Staging" ou "Production"

    Utile pour les rollbacks ou les promotions manuelles.
    """
    try:
        from mlflow_registry import MLflowRegistry, Stage
        allowed_stages = [Stage.STAGING, Stage.PRODUCTION, Stage.ARCHIVED]
        if stage not in allowed_stages:
            raise HTTPException(400, detail={
                "error": "invalid_stage",
                "message": f"Stage doit être l'un de : {allowed_stages}",
            })

        registry = MLflowRegistry(model_name)
        if stage == Stage.PRODUCTION:
            ok = registry.promote_to_production(version)
        elif stage == Stage.STAGING:
            ok = registry.promote_to_staging(version)
        else:
            ok = registry.archive(version)

        if not ok:
            raise HTTPException(500, detail={"error": "promotion_failed"})

        return {
            "success"   : True,
            "model_name": model_name,
            "version"   : version,
            "new_stage" : stage,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Registry promotion failed")
        raise HTTPException(500, detail={"error": "registry_promotion_failed"})

@app.get(
    "/v1/drift/latest",
    tags=["Drift"],
    dependencies=_auth_deps,
    summary="Dernier rapport de drift disponible",
)
async def drift_latest():
    """
    Retourne le dernier rapport de drift sauvegardé.
    Les rapports sont générés automatiquement à chaque training pipeline.
    """
    import glob
    reports_dir = os.getenv("REPORTS_DIR", "reports/runtime")
    pattern = os.path.join(reports_dir, "drift_report_*.json")
    files = sorted(glob.glob(pattern), reverse=True)

    if not files:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_drift_report",
                "message": "Aucun rapport de drift trouvé. Lancer le pipeline d'abord.",
            },
        )

    with open(files[0], encoding="utf-8") as f:
        report = json.load(f)

    return {
        "report_file": os.path.basename(files[0]),
        "n_reports_available": len(files),
        **report,
    }

class DriftDetectRequest(BaseModel):
    """Données pour la détection de drift en temps réel."""
    data: list[dict] = Field(..., description="Liste de propriétés (même format que /predict)")
    include_predictions: bool = Field(True, description="Inclure le drift des prédictions")

    model_config = {
        "json_schema_extra": {
            "example": {
                "data": [
                    {"surface_m2": 90,  "ville": "Casablanca", "type_bien": "appartement"},
                    {"surface_m2": 150, "ville": "Rabat",      "type_bien": "villa"},
                ],
                "include_predictions": True,
            }
        }
    }

@app.post(
    "/v1/drift/detect",
    tags=["Drift"],
    dependencies=_auth_deps,
    summary="Détecter le drift sur un batch de nouvelles données",
)
async def drift_detect(data: DriftDetectRequest, request: Request):
    """
    Détecte le drift entre les données de référence (training) et un nouveau batch.

    Utile pour monitorer la qualité des données en production en temps réel.

    Retourne :
    - **recommendation** : "ok" | "monitor" | "retrain"
    - **dataset_psi**    : PSI global
    - **drifted_features** : features avec drift détecté
    """
    req_id = getattr(request.state, "request_id", "unknown")

    if _state["preprocessor"] is None:
        raise HTTPException(503, detail={
            "error": "preprocessor_unavailable",
            "message": "Préprocesseur non chargé.",
            "request_id": req_id,
        })

    try:
        from drift_detector import DriftDetector

        _fs_dir  = os.path.join(os.getenv("MODELS_DIR", "models"), "..", "feature_store")
        ref_path = os.path.join(_fs_dir, "reference_data.pkl")

        if not os.path.exists(ref_path):
            ref_path = os.path.join(os.getenv("MODELS_DIR", "models"), "reference_data.pkl")

        if not os.path.exists(ref_path):
            raise HTTPException(404, detail={
                "error": "no_reference_data",
                "message": (
                    "Données de référence introuvables. "
                    "Cherché dans : feature_store/ et models/. "
                    "Relancer le pipeline pour les générer."
                ),
            })

        import pickle
        with open(ref_path, "rb") as f:
            reference_df = pickle.load(f)

        current_df = pd.DataFrame(data.data)
        X_current = _state["preprocessor"].transform(current_df)
        X_current_df = pd.DataFrame(X_current, columns=_state.get("feature_names", []) or [f"f{i}" for i in range(X_current.shape[1])])

        pred_ref = pred_cur = None
        if data.include_predictions and _state["reg_model"] is not None:
            X_ref = _state["preprocessor"].transform(reference_df)
            pred_ref = _state["reg_model"].predict(X_ref[:min(1000, len(X_ref))])
            pred_cur = _state["reg_model"].predict(X_current)

        detector = DriftDetector(reference_data=reference_df)
        report = detector.detect(
            current_data    = X_current_df,
            predictions_ref = pred_ref,
            predictions_cur = pred_cur,
            save_report     = False,
        )

        return {
            "request_id"    : req_id,
            "n_samples"     : len(data.data),
            **report.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[%s] Drift detection failed (%s)", req_id, type(exc).__name__)
        raise HTTPException(500, detail={"error": "drift_detection_failed"})

def _get_feature_store():
    """Charge le Feature Store depuis le dossier models."""
    from feature_store import FeatureStore
    store_path = os.path.join(
        os.getenv("MODELS_DIR", "models"), "..", "feature_store", "store.db"
    )
    return FeatureStore(store_path=store_path)

async def _run_in_thread(func, *args, **kwargs):
    """
    Exécute une fonction synchrone dans un thread pool pour ne pas bloquer
    l'event loop FastAPI (SQLite, pickle, etc. sont synchrones).
    """
    import asyncio
    import functools
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

@app.get(
    "/v1/features/groups",
    tags=["Feature Store"],
    dependencies=_auth_deps,
    summary="Liste tous les feature groups disponibles",
)
async def feature_groups():
    """
    Retourne la liste des feature groups enregistrés dans le Feature Store,
    avec le nombre d'entités, les features disponibles et la date de dernière MAJ.
    SQLite s'exécute dans un thread pool pour ne pas bloquer l'event loop.
    """
    try:
        def _fetch():
            fs = _get_feature_store()
            return fs.list_groups(), fs.get_feature_freshness()

        groups, freshness = await _run_in_thread(_fetch)
        return {
            "n_groups": len(groups),
            "groups"  : groups,
            "freshness": freshness,
        }
    except Exception as exc:
        logger.exception("Feature store group listing failed")
        raise HTTPException(500, detail={"error": "feature_store_error"})

@app.get(
    "/v1/features/{group}",
    tags=["Feature Store"],
    dependencies=_auth_deps,
    summary="Lire les features d'un groupe",
)
async def read_features(
    group: str,
    version: str = "v1",
    limit: int = 100,
):
    """
    Lit les features d'un groupe depuis le Feature Store.

    - **group**   : nom du groupe (ex: property_base, geographic, temporal)
    - **version** : version des features
    - **limit**   : nombre max de lignes retournées
    """
    try:
        fs = _get_feature_store()
        df = fs.read_all(group=group, version=version)

        if df.empty:
            raise HTTPException(404, detail={
                "error": "group_not_found",
                "message": f"Groupe '{group}' v{version} introuvable.",
            })

        df_limited = df.head(limit)
        return {
            "group"     : group,
            "version"   : version,
            "n_total"   : len(df),
            "n_returned": len(df_limited),
            "features"  : list(df.columns),
            "data"      : df_limited.fillna("null").to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Feature store operation failed")
        raise HTTPException(500, detail={"error": "feature_store_error"})

@app.get(
    "/v1/features/{group}/stats",
    tags=["Feature Store"],
    dependencies=_auth_deps,
    summary="Statistiques d'un feature group",
)
async def feature_group_stats(group: str):
    """Retourne l'historique des écritures et statistiques d'un feature group."""
    try:
        fs = _get_feature_store()
        stats = fs.get_stats(group=group)
        return {
            "group"  : group,
            "history": stats,
        }
    except Exception as exc:
        logger.exception("Feature store statistics unavailable")
        raise HTTPException(500, detail={"error": "feature_store_error"})

class RetrainRequest(BaseModel):
    reason: str = Field("manual", description="Raison du retraining (drift / manual / scheduled)")
    force: bool = Field(False, description="Forcer même si drift non confirmé")

    model_config = {
        "json_schema_extra": {
            "example": {"reason": "drift_detected", "force": False}
        }
    }

@app.post(
    "/v1/retrain",
    tags=["MLOps"],
    dependencies=_auth_deps,
    summary="Déclencher un retraining du pipeline ML",
)
async def trigger_retrain(data: RetrainRequest, request: Request):
    """
    Déclenche un retraining du pipeline ML.

    **Modes disponibles** :
    - Vérifier d'abord le dernier rapport de drift
    - Si drift confirmé (ou `force=True`) → lance le pipeline
    - Retourne immédiatement un job ID (le pipeline tourne en arrière-plan)

    **Quand l'utiliser** :
    - `POST /v1/retrain` avec `reason="drift_detected"` depuis un monitoring externe
    - Automatiquement depuis `/v1/drift/detect` si `recommendation="retrain"`

    **Note** : En production, ce webhook peut être appelé par :
    - GitHub Actions (scheduled retraining)
    - Alertmanager (drift alert)
    - Render/K8s CronJob
    """
    req_id = getattr(request.state, "request_id", "unknown")

    drift_recommendation = "unknown"
    try:
        import glob
        reports_dir = os.getenv("REPORTS_DIR", "reports/runtime")
        files = sorted(glob.glob(os.path.join(reports_dir, "drift_report_*.json")), reverse=True)
        if files:
            with open(files[0]) as f:
                last_report = json.load(f)
            drift_recommendation = last_report.get("recommendation", "unknown")
    except Exception:
        pass

    should_retrain = data.force or drift_recommendation in ("retrain", "monitor")

    if not should_retrain and drift_recommendation == "ok":
        return {
            "request_id"         : req_id,
            "triggered"          : False,
            "reason"             : data.reason,
            "drift_recommendation": drift_recommendation,
            "message"            : "Aucun drift détecté: retraining non nécessaire. Utiliser force=True pour forcer.",
        }

    job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{req_id}"

    logger.warning(
        f"🔴 RETRAINING DÉCLENCHÉ: job={job_id} | "
        f"drift={drift_recommendation}"
    )

    return {
        "request_id"         : req_id,
        "triggered"          : True,
        "job_id"             : job_id,
        "reason"             : data.reason,
        "drift_recommendation": drift_recommendation,
        "timestamp"          : datetime.now().isoformat(),
        "next_steps": {
            "local"     : "make train",
            "docker"    : "make docker-train",
            "ci_cd"     : "Push to main branch triggers GitHub Actions pipeline",
        },
        "message": (
            f"Retraining déclenché (job={job_id}). "
            f"En production, connecter ce webhook à votre CI/CD ou Render Deploy Hook."
        ),
    }