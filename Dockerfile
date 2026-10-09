
# Dockerfile v3: Avito Real Estate ML Pipeline
# Multi-stage build : builder + runtime


# Stage 1 : Builder
FROM python:3.11-slim AS builder

WORKDIR /build

# دپندنسيات النظام للـ compile
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

ARG REQUIREMENTS_FILE=requirements-api.txt
COPY requirements.txt requirements-api.txt ./
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r "$REQUIREMENTS_FILE"

# Stage 2 : Runtime
FROM python:3.11-slim AS runtime

LABEL maintainer="ML Pipeline" \
      version="3.0" \
      description="Avito Real Estate: ML Pipeline v3"

WORKDIR /app

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Packages من الـ builder
COPY --from=builder /install /usr/local

# Source code
COPY src/        ./src/
COPY config/     ./config/

# Directories + non-root user
RUN mkdir -p models logs reports/runtime docs/plots mlruns \
    && useradd --no-create-home --shell /bin/false mluser \
    && chown -R mluser:mluser /app

USER mluser

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    MODELS_DIR=/app/models

# Stage API
FROM runtime AS api

EXPOSE 8000

# ✅ FIXED: يستعمل /health بدل /: مناسب للـ API v3
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

# workers=1 في dev، زيد حسب الـ CPU في production
CMD ["uvicorn", "src.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--access-log", \
     "--log-level", "info"]

# Stage Pipeline (training)
FROM runtime AS pipeline

CMD ["python", "src/pipeline.py"]
