"""FastAPI route smoke tests. The graph is mocked so the LLM is never called."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient


def _make_client():
    from app.main import create_app

    app = create_app()
    return TestClient(app)


def test_healthz():
    with patch("app.observability.tracing.configure_tracing", lambda *_: None):
        client = _make_client()
        r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_endpoint_returns_prometheus():
    with patch("app.observability.tracing.configure_tracing", lambda *_: None):
        client = _make_client()
        r = client.get("/metrics")
    assert r.status_code == 200
    assert b"mts_compile_requests_total" in r.content
