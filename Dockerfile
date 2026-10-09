

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

ARG REQUIREMENTS_FILE=requirements-api.txt
COPY requirements.txt requirements-api.txt ./
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r "$REQUIREMENTS_FILE"

FROM python:3.11-slim AS runtime

LABEL maintainer="ML Pipeline" \
      version="3.0" \
      description="Avito Real Estate: ML Pipeline v3"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY src/        ./src/
COPY config/     ./config/

RUN mkdir -p models logs reports/runtime docs/plots mlruns \
    && useradd --no-create-home --shell /bin/false mluser \
    && chown -R mluser:mluser /app

USER mluser

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    MODELS_DIR=/app/models

FROM runtime AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["uvicorn", "src.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--access-log", \
     "--log-level", "info"]

FROM runtime AS pipeline

CMD ["python", "src/pipeline.py"]
