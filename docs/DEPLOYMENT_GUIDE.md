# Deployment Guide

This repository is a portfolio project with a FastAPI service and a separate machine learning training pipeline.

## Local setup

Clone the repository, create a local environment file and configure real credentials.

```bash
git clone https://github.com/badre2152/Real-Estate-Machine-Learning-Pipeline.git
cd Real-Estate-Machine-Learning-Pipeline
cp .env.example .env
docker compose up -d
```

Set `DB_PASSWORD` and `API_KEYS` in `.env` before starting Docker Compose. The development example values must be replaced for a real deployment.

Local services:

| Service | Local address |
| --- | --- |
| FastAPI | http://localhost:8000 |
| FastAPI health | http://localhost:8000/health |
| MLflow | http://localhost:5000 |
| PostgreSQL | localhost:5433 |

Database, API and MLflow ports are bound to localhost on the host machine. Docker containers use the internal service names and ports.

Run training separately:

```bash
docker compose --profile train up pipeline
```

## Render

The `render.yaml` Blueprint defines only the FastAPI web service. It does not provision PostgreSQL. An existing PostgreSQL instance and its connection details are required.

Configure `API_KEYS`, `CORS_ORIGINS`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` and `DB_PASSWORD` in the Render dashboard. Production configuration rejects missing or placeholder API keys and wildcard CORS origins.

Review Render's currently supported plan, resource limits and deployment configuration before attempting deployment. The Blueprint has not been deployed or validated against Render in this cleanup.

Model files are generated separately and are not copied into the Docker image. A deployed prediction service must be provided with compatible trained model files in `MODELS_DIR`; otherwise prediction endpoints may not be ready.

## Optional Nginx HTTPS proxy

```bash
docker compose --profile nginx up -d
```

Before enabling Nginx, place valid TLS certificates at `nginx/certs/fullchain.pem` and `nginx/certs/privkey.pem`. Without them, the supplied Nginx configuration cannot start.

## Verification and deployment status

No GitHub Actions CI or CD workflow is configured. Build, deployment and runtime verification are manual. This documentation describes the intended configuration and does not claim that an external deployment is live.
