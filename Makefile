# Makefile: Avito Real Estate ML Pipeline
# Usage : make <target>

.PHONY: help install install-dev test test-cov lint lint-full run run-full clean

PYTHON  = python
PYTEST  = pytest
SRC_DIR = src
TST_DIR = tests
API_KEY ?= change_me_api_key
MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db

# Aide
help:
	@echo ""
	@echo "  Avito Real Estate: ML Pipeline"
	@echo ""
	@echo "  make install       Installer les dépendances de production"
	@echo "  make install-dev   Installer toutes les dépendances (prod + dev)"
	@echo "  make test          Lancer les tests unitaires"
	@echo "  make test-cov      Tests + rapport de couverture"
	@echo "  make lint          Vérifications Ruff critiques"
	@echo "  make lint-full     Audit Ruff complet"
	@echo "  make run           Lancer le pipeline (mode standard)"
	@echo "  make run-full      Lancer le pipeline avec toutes les options"
	@echo "  make clean         Supprimer les fichiers générés"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

# Tests
test:
	$(PYTEST) $(TST_DIR)/ -v --tb=short

test-cov:
	$(PYTEST) $(TST_DIR)/ -v \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html:docs/coverage \
		--cov-fail-under=75

test-fast:
	$(PYTEST) $(TST_DIR)/ -v --tb=short -x -q

# Linting
lint:
	ruff check $(SRC_DIR)/ $(TST_DIR)/ --select E9,F63,F7,F82

lint-full:
	ruff check $(SRC_DIR)/ $(TST_DIR)/ --ignore E501,E402

# Pipeline
run:
	$(PYTHON) $(SRC_DIR)/pipeline.py

run-log-target:
	$(PYTHON) $(SRC_DIR)/pipeline.py --log-target

run-full:
	$(PYTHON) $(SRC_DIR)/pipeline.py --log-target --smote --optimize --calibrate

run-ci:
	$(PYTHON) $(SRC_DIR)/pipeline.py --no-plots

# Nettoyage
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -name "coverage.xml" -delete 2>/dev/null || true
	rm -f models/regression_model.pkl models/classification_model.pkl
	rm -f models/preprocessor.pkl models/results.json
	rm -f models/pipeline_*.log
	rm -f docs/plots/*.png
	@echo "✅ Nettoyage terminé"

# Docker (v3)
docker-build:
	docker build --target api -t avito-ml-api:3.0 .

docker-run:
	docker compose up -d

docker-train:
	docker compose --profile train up pipeline

docker-stop:
	docker compose down

docker-logs:
	docker compose logs -f api

docker-health:
	curl -s http://localhost:8000/health | python -m json.tool

docker-ready:
	curl -s http://localhost:8000/ready | python -m json.tool

# API testing (v3)
api-test-auth:
	@echo "Test sans clé → doit retourner 401:"
	curl -s -o /dev/null -w "Status: %{http_code}\n" http://localhost:8000/v1/info
	@echo "Test avec clé → doit retourner 200:"
	curl -s -o /dev/null -w "Status: %{http_code}\n" \
	  -H "X-API-Key: $(API_KEY)" http://localhost:8000/v1/info

api-predict:
	curl -s -X POST http://localhost:8000/v1/predict \
	  -H "Content-Type: application/json" \
	  -H "X-API-Key: $(API_KEY)" \
	  -d '{"surface_m2":120,"ville":"Casablanca","type_bien":"appartement","nb_chambres":3}' \
	  | python -m json.tool

# MLflow Registry (v3)
registry-status:
	curl -s -H "X-API-Key: $(API_KEY)" \
	  http://localhost:8000/v1/registry | python -m json.tool

registry-promote:
	@echo "Usage: make registry-promote MODEL=avito-regression VERSION=2 STAGE=Production"
	curl -s -X POST \
	  "http://localhost:8000/v1/registry/promote?model_name=$(MODEL)&version=$(VERSION)&stage=$(STAGE)" \
	  -H "X-API-Key: $(API_KEY)" | python -m json.tool

mlflow-ui:
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000

# Drift Detection (v3)
drift-latest:
	curl -s -H "X-API-Key: $(API_KEY)" \
	  http://localhost:8000/v1/drift/latest | python -m json.tool

drift-detect:
	curl -s -X POST http://localhost:8000/v1/drift/detect \
	  -H "Content-Type: application/json" \
	  -H "X-API-Key: $(API_KEY)" \
	  -d '{"data":[{"surface_m2":120,"ville":"Casablanca","type_bien":"appartement"},{"surface_m2":200,"ville":"Rabat","type_bien":"villa"}],"include_predictions":true}' \
	  | python -m json.tool

# Feature Store (v3)
fs-groups:
	curl -s -H "X-API-Key: $(API_KEY)" \
	  http://localhost:8000/v1/features/groups | python -m json.tool

fs-read:
	@echo "Usage: make fs-read GROUP=property_base"
	curl -s -H "X-API-Key: $(API_KEY)" \
	  "http://localhost:8000/v1/features/$(GROUP)?limit=5" | python -m json.tool

fs-stats:
	@echo "Usage: make fs-stats GROUP=property_base"
	curl -s -H "X-API-Key: $(API_KEY)" \
	  "http://localhost:8000/v1/features/$(GROUP)/stats" | python -m json.tool
