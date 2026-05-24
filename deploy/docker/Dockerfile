# syntax=docker/dockerfile:1.7

# --- Stage 1: build a venv with uv -------------------------------------------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app ./app
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache .

# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 mts

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app ./app
COPY data ./data
COPY scripts ./scripts

USER mts
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
