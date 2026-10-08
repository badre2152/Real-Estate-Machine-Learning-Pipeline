"""
test_api_v3.py
--------------
Tests pour les nouvelles fonctionnalités de l'API v3 :
  - Authentication (API Key)
  - Rate Limiting
  - /health endpoint
  - /ready endpoint
  - Request ID dans les réponses
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Mock les imports qui nécessitent des fichiers externes
import sys
import logging

# Use patch.dict so the mocks are isolated and don't leak into other test modules
_mocks = {
    'logger_setup': MagicMock(get_logger=lambda name: logging.getLogger(name)),
    'prediction_intervals': MagicMock(),
}
with patch.dict('sys.modules', _mocks):
    import importlib
    import src.api as api_module
    from src.api import app, _state, VALID_API_KEYS

client = TestClient(app, raise_server_exceptions=False)

VALID_KEY = list(VALID_API_KEYS)[0]
HEADERS_AUTH = {"X-API-Key": VALID_KEY}


# 
# Tests /health (pas d'auth)
# 

class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_no_auth_required(self):
        """Health doit fonctionner sans API Key."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_structure(self):
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert "version" in data
        assert data["status"] == "ok"
        assert data["version"] == "3.0.0"

    def test_health_has_timestamp(self):
        r = client.get("/health")
        assert "timestamp" in r.json()


# 
# Tests /ready (pas d'auth)
# 

class TestReady:
    def test_ready_no_auth_required(self):
        r = client.get("/ready")
        assert r.status_code in [200, 503]  # dépend si les modèles sont chargés

    def test_ready_503_when_no_model(self):
        with patch.dict(_state, {"reg_model": None}):
            r = client.get("/ready")
            assert r.status_code == 503
            assert r.json()["ready"] is False

    def test_ready_structure(self):
        r = client.get("/ready")
        data = r.json()
        assert "ready" in data
        assert "models_loaded" in data
        assert "regression_model" in data["models_loaded"]


# 
# Tests Authentication
# 

class TestAuthentication:
    def test_no_api_key_returns_401(self):
        r = client.get("/v1/info")
        assert r.status_code == 401
        assert r.json()["detail"]["error"] == "missing_api_key"

    def test_invalid_api_key_returns_403(self):
        r = client.get("/v1/info", headers={"X-API-Key": "wrong-key-xyz"})
        assert r.status_code == 403
        assert r.json()["detail"]["error"] == "invalid_api_key"

    def test_valid_api_key_passes(self):
        r = client.get("/v1/info", headers=HEADERS_AUTH)
        assert r.status_code == 200

    def test_predict_requires_auth(self):
        r = client.post("/v1/predict", json={
            "surface_m2": 100, "ville": "Casablanca", "type_bien": "appartement"
        })
        assert r.status_code == 401

    def test_batch_requires_auth(self):
        r = client.post("/v1/predict/batch", json={"properties": []})
        assert r.status_code == 401


# 
# Tests Rate Limiting
# 

class TestRateLimit:
    def test_rate_limiter_allows_normal_traffic(self):
        from src.api import _rate_limiter, RATE_LIMIT_PER_MINUTE
        # Une seule requête doit passer
        allowed, remaining = _rate_limiter.is_allowed("test-ip-normal")
        assert allowed is True
        assert remaining == RATE_LIMIT_PER_MINUTE - 1

    def test_rate_limiter_blocks_after_limit(self):
        from src.api import _InMemoryRateLimiter
        # Créer un limiter avec max 3 appels
        limiter = _InMemoryRateLimiter(max_calls=3)
        for i in range(3):
            allowed, _ = limiter.is_allowed("test-ip-block")
            assert allowed is True
        # Le 4e doit être bloqué
        allowed, remaining = limiter.is_allowed("test-ip-block")
        assert allowed is False
        assert remaining == 0

    def test_rate_limiter_different_ips_independent(self):
        from src.api import _InMemoryRateLimiter
        limiter = _InMemoryRateLimiter(max_calls=1)
        allowed1, _ = limiter.is_allowed("ip-1")
        allowed2, _ = limiter.is_allowed("ip-2")
        assert allowed1 is True
        assert allowed2 is True


# 
# Tests Request ID
# 

class TestRequestId:
    def test_health_response_has_request_id_header(self):
        r = client.get("/health")
        assert "x-request-id" in r.headers

    def test_request_ids_are_unique(self):
        r1 = client.get("/health")
        r2 = client.get("/health")
        id1 = r1.headers.get("x-request-id")
        id2 = r2.headers.get("x-request-id")
        assert id1 != id2

    def test_response_time_header_present(self):
        r = client.get("/health")
        assert "x-response-time-ms" in r.headers


# 
# Tests /v1/metrics
# 

class TestMetrics:
    def test_metrics_requires_auth(self):
        r = client.get("/v1/metrics")
        assert r.status_code == 401

    def test_metrics_structure(self):
        r = client.get("/v1/metrics", headers=HEADERS_AUTH)
        assert r.status_code == 200
        data = r.json()
        assert "ml_total_predictions" in data
        assert "ml_total_errors" in data
        assert "ml_error_rate" in data
        assert "rate_limit_per_min" in data


# 
# Tests Backward Compatibility
# 

class TestBackwardCompatibility:
    def test_root_endpoint_still_works(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_old_predict_endpoint_still_requires_auth(self):
        r = client.post("/predict", json={
            "surface_m2": 100, "ville": "Casablanca", "type_bien": "appartement"
        })
        assert r.status_code == 401
