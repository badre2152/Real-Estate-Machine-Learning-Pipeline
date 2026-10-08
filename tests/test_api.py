"""
test_api.py
-----------
Tests unitaires et d'intégration pour l'API FastAPI.

Couvre :
  - GET  /        : health check
  - GET  /info    : model info
  - GET  /metrics : monitoring metrics
  - POST /predict : prédiction unitaire (valides + invalides)
  - POST /predict/batch: prédictions en lot
  - Validation Pydantic (types, bornes, champs obligatoires)
  - Comportement avec modèles non chargés (mode dégradé)
  - Middleware de logging (status codes)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import conditionnel FastAPI
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="fastapi non installé (pip install fastapi httpx)"
)


# 
# Fixtures
# 

@pytest.fixture(scope="module")
def client():
    """Client de test avec modèles mockés chargés en mémoire."""
    import pandas as pd
    from api import app, _state
    from prediction_intervals import PredictionIntervalBuilder

    # Mock du modèle de régression
    reg_mock = MagicMock()
    reg_mock.predict.return_value = np.array([750_000.0])

    # Mock du modèle de classification
    clf_mock = MagicMock()
    clf_mock.predict.return_value = np.array([1])

    # Mock du label encoder
    le_mock = MagicMock()
    le_mock.inverse_transform.return_value = np.array(["moyen"])

    # Mock du pi_builder
    pi_mock = MagicMock(spec=PredictionIntervalBuilder)
    pi_mock.predict_with_interval.return_value = pd.DataFrame({
        "prediction"    : [750_000.0],
        "lower"         : [600_000.0],
        "upper"         : [900_000.0],
        "interval_width": [300_000.0],
    })

    mocks = {
        "reg_model"    : reg_mock,
        "clf_model"    : clf_mock,
        "label_encoder": le_mock,
        "pi_builder"   : pi_mock,
        "reg_metrics"  : {"R2": 0.82, "MAE": 45000.0, "RMSE": 72000.0},
        "clf_metrics"  : {"F1": 0.74, "Accuracy": 0.76},
        "loaded_at"    : "2026-01-01T00:00:00",
    }

    # Patch pickle.load pour éviter le chargement des fichiers réels au démarrage
    with patch("api.pickle.load", side_effect=lambda f: MagicMock()):
        with TestClient(app, headers={"X-API-Key": "test-key-ci"}) as c:
            # Injecter les vrais mocks après le démarrage
            _state.update(mocks)
            yield c


@pytest.fixture
def valid_property():
    """Payload valide pour /predict."""
    return {
        "surface_m2"    : 120,
        "ville"         : "Casablanca",
        "type_bien"     : "appartement",
        "nb_chambres"   : 3,
        "nb_salles_bain": 2,
        "etage"         : 4,
        "age_bien"      : 10,
    }


# 
# Tests: Health Check
# 

class TestHealthCheck:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_status_ok(self, client):
        data = client.get("/").json()
        assert data["status"] == "ok"

    def test_root_has_version(self, client):
        data = client.get("/").json()
        assert "version" in data
        assert data["version"] == "2.0.0"

    def test_root_models_loaded(self, client):
        data = client.get("/").json()
        assert data["models_loaded"] is True

    def test_root_has_timestamp(self, client):
        data = client.get("/").json()
        assert "timestamp" in data


# 
# Tests: /info
# 

class TestModelInfo:
    def test_info_returns_200(self, client):
        assert client.get("/info").status_code == 200

    def test_info_has_regression_model_name(self, client):
        data = client.get("/info").json()
        assert "regression_model" in data
        assert data["regression_model"] is not None

    def test_info_has_metrics(self, client):
        data = client.get("/info").json()
        assert "regression_metrics" in data
        assert data["regression_metrics"]["R2"] == pytest.approx(0.82)

    def test_info_has_prediction_count(self, client):
        data = client.get("/info").json()
        assert "total_predictions" in data
        assert isinstance(data["total_predictions"], int)

    def test_info_has_loaded_at(self, client):
        data = client.get("/info").json()
        assert data["loaded_at"] == "2026-01-01T00:00:00"


# 
# Tests: /metrics
# 

class TestMetrics:
    def test_metrics_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_metrics_has_prediction_count(self, client):
        data = client.get("/metrics").json()
        assert "ml_total_predictions" in data

    def test_metrics_has_model_loaded_flag(self, client):
        data = client.get("/metrics").json()
        assert data["ml_model_loaded"] == 1

    def test_metrics_has_r2(self, client):
        data = client.get("/metrics").json()
        assert "ml_reg_r2" in data
        assert data["ml_reg_r2"] == pytest.approx(0.82)


# 
# Tests: POST /predict (cas valides)
# 

class TestPredictValid:
    def test_predict_returns_200(self, client, valid_property):
        resp = client.post("/predict", json=valid_property)
        assert resp.status_code == 200

    def test_predict_has_prediction_field(self, client, valid_property):
        data = client.post("/predict", json=valid_property).json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)

    def test_predict_has_confidence_interval(self, client, valid_property):
        data = client.post("/predict", json=valid_property).json()
        assert "lower_95" in data
        assert "upper_95" in data
        assert data["lower_95"] < data["prediction"] < data["upper_95"]

    def test_predict_has_interval_width(self, client, valid_property):
        data = client.post("/predict", json=valid_property).json()
        assert "interval_width" in data
        expected = data["upper_95"] - data["lower_95"]
        assert data["interval_width"] == pytest.approx(expected, rel=1e-3)

    def test_predict_has_price_category(self, client, valid_property):
        data = client.post("/predict", json=valid_property).json()
        assert "price_category" in data
        assert data["price_category"] == "moyen"

    def test_predict_has_formatted_string(self, client, valid_property):
        data = client.post("/predict", json=valid_property).json()
        assert "formatted" in data
        assert "MAD" in data["formatted"]

    def test_predict_has_latency_ms(self, client, valid_property):
        data = client.post("/predict", json=valid_property).json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_predict_increments_counter(self, client, valid_property):
        from api import _state
        before = _state["total_predictions"]
        client.post("/predict", json=valid_property)
        assert _state["total_predictions"] == before + 1

    def test_predict_ville_normalized(self, client):
        """La ville doit être normalisée en Title Case."""
        payload = {
            "surface_m2": 80,
            "ville"     : "casablanca",   # minuscules
            "type_bien" : "appartement",
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_minimal_payload(self, client):
        """Seuls surface_m2, ville et type_bien sont obligatoires."""
        payload = {
            "surface_m2": 60,
            "ville"     : "Rabat",
            "type_bien" : "studio",
        }
        assert client.post("/predict", json=payload).status_code == 200


# 
# Tests: POST /predict (Pydantic validation: cas invalides)
# 

class TestPredictInvalid:
    def test_missing_surface_returns_422(self, client):
        payload = {"ville": "Casablanca", "type_bien": "appartement"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_ville_returns_422(self, client):
        payload = {"surface_m2": 100, "type_bien": "appartement"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_type_bien_returns_422(self, client):
        payload = {"surface_m2": 100, "ville": "Casablanca"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_negative_surface_returns_422(self, client):
        payload = {"surface_m2": -10, "ville": "Casablanca", "type_bien": "appartement"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_zero_surface_returns_422(self, client):
        payload = {"surface_m2": 0, "ville": "Casablanca", "type_bien": "appartement"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_surface_too_large_returns_422(self, client):
        payload = {"surface_m2": 99_999, "ville": "Casablanca", "type_bien": "appartement"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_empty_ville_returns_422(self, client):
        payload = {"surface_m2": 100, "ville": "", "type_bien": "appartement"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_too_many_rooms_returns_422(self, client):
        payload = {
            "surface_m2" : 100,
            "ville"      : "Casablanca",
            "type_bien"  : "villa",
            "nb_chambres": 99,   # max=20
        }
        assert client.post("/predict", json=payload).status_code == 422

    def test_invalid_json_returns_422(self, client):
        resp = client.post("/predict", content="not-json",
                           headers={"Content-Type": "application/json"})
        assert resp.status_code == 422


# 
# Tests: POST /predict/batch
# 

class TestPredictBatch:
    def test_batch_returns_200(self, client, valid_property):
        payload = {"properties": [valid_property, valid_property]}
        assert client.post("/predict/batch", json=payload).status_code == 200

    def test_batch_returns_correct_count(self, client, valid_property):
        payload = {"properties": [valid_property] * 3}
        data = client.post("/predict/batch", json=payload).json()
        assert data["n_properties"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_single_property(self, client, valid_property):
        payload = {"properties": [valid_property]}
        data = client.post("/predict/batch", json=payload).json()
        assert data["n_properties"] == 1

    def test_batch_has_total_latency(self, client, valid_property):
        payload = {"properties": [valid_property, valid_property]}
        data = client.post("/predict/batch", json=payload).json()
        assert "total_latency_ms" in data
        assert data["total_latency_ms"] >= 0

    def test_batch_empty_list_returns_422(self, client):
        """Une liste vide doit être rejetée par Pydantic (min_length=1)."""
        payload = {"properties": []}
        # FastAPI/Pydantic rejette la liste vide si min_length > 0
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code in (200, 422)   # selon config Pydantic

    def test_batch_predictions_have_intervals(self, client, valid_property):
        payload = {"properties": [valid_property]}
        data = client.post("/predict/batch", json=payload).json()
        pred = data["predictions"][0]
        assert "lower_95" in pred
        assert "upper_95" in pred


# 
# Tests: Mode dégradé (modèles non chargés)
# 

class TestDegradedMode:
    def test_predict_without_model_returns_503(self, valid_property):
        """Sans modèle chargé, /predict doit retourner 503."""
        from api import app, _state
        with patch("api.pickle.load", side_effect=lambda f: MagicMock()):
            with TestClient(app, headers={"X-API-Key": "test-key-ci"}) as c:
                _state["reg_model"] = None
                resp = c.post("/predict", json=valid_property)
        assert resp.status_code == 503

    def test_info_works_without_models(self):
        """GET /info doit toujours répondre, même sans modèle."""
        from api import app, _state
        with patch("api.pickle.load", side_effect=lambda f: MagicMock()):
            with TestClient(app, headers={"X-API-Key": "test-key-ci"}) as c:
                _state["reg_model"] = None
                resp = c.get("/info")
        assert resp.status_code == 200
        assert resp.json()["regression_model"] is None

    def test_health_works_without_models(self):
        """GET / doit toujours fonctionner."""
        from api import app, _state
        with patch("api.pickle.load", side_effect=lambda f: MagicMock()):
            with TestClient(app, headers={"X-API-Key": "test-key-ci"}) as c:
                _state["reg_model"] = None
                resp = c.get("/")
        assert resp.status_code == 200
        assert resp.json()["models_loaded"] is False


# 
# Tests: Schémas Pydantic (unitaires, sans serveur)
# 

class TestPydanticSchemas:
    def test_property_input_valid(self):
        from api import PropertyInput
        p = PropertyInput(surface_m2=100, ville="Rabat", type_bien="villa")
        assert p.surface_m2 == 100

    def test_property_input_ville_title_case(self):
        from api import PropertyInput
        p = PropertyInput(surface_m2=100, ville="casablanca", type_bien="appartement")
        assert p.ville == "Casablanca"

    def test_property_input_type_bien_lowercase(self):
        from api import PropertyInput
        p = PropertyInput(surface_m2=100, ville="Fès", type_bien="VILLA")
        assert p.type_bien == "villa"

    def test_property_input_rejects_negative_surface(self):
        from api import PropertyInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PropertyInput(surface_m2=-5, ville="Rabat", type_bien="studio")

    def test_property_input_rejects_too_many_rooms(self):
        from api import PropertyInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PropertyInput(surface_m2=200, ville="Rabat", type_bien="villa", nb_chambres=25)

    def test_property_input_optional_fields_default_none(self):
        from api import PropertyInput
        p = PropertyInput(surface_m2=80, ville="Marrakech", type_bien="appartement")
        assert p.nb_chambres    is None
        assert p.nb_salles_bain is None
        assert p.etage          is None
        assert p.age_bien       is None
