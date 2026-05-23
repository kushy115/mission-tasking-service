"""Prometheus metrics. Mounted at /metrics by app.main.create_app."""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

COMPILE_REQUESTS = Counter(
    "mts_compile_requests_total",
    "Compile requests served, labeled by final mission status.",
    labelnames=("status",),
)
COMPILE_DURATION = Histogram(
    "mts_compile_duration_seconds",
    "End-to-end latency of compile requests.",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
REPAIR_LOOPS = Histogram(
    "mts_repair_loops",
    "Number of repair iterations per compile call.",
    buckets=(0, 1, 2, 3, 4, 5),
)
REJECTIONS_TOTAL = Counter("mts_rejections_total", "Plans returned as REJECTED.")
CLARIFICATIONS_TOTAL = Counter(
    "mts_clarifications_total", "Plans returned as NEEDS_CLARIFICATION."
)
TOKENS_TOTAL = Counter(
    "mts_tokens_total",
    "LLM tokens consumed, labeled by graph node.",
    labelnames=("node",),
)
TOOL_CALLS_TOTAL = Counter(
    "mts_tool_calls_total",
    "Tool invocations, labeled by tool name and outcome.",
    labelnames=("tool", "outcome"),
)
TOOL_ERRORS_TOTAL = Counter(
    "mts_tool_errors_total",
    "Tool errors, labeled by tool name.",
    labelnames=("tool",),
)

HTTP_REQUESTS = Counter(
    "mts_http_requests_total",
    "HTTP requests, labeled by path and status.",
    labelnames=("method", "path", "status"),
)
HTTP_DURATION = Histogram(
    "mts_http_duration_seconds",
    "HTTP latency by path.",
    labelnames=("method", "path"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)


class _PromMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        start = time.perf_counter()
        response: Response = await call_next(request)
        dur = time.perf_counter() - start
        path = request.scope.get("route").path if request.scope.get("route") else request.url.path
        HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        HTTP_DURATION.labels(request.method, path).observe(dur)
        return response


def register_metrics_middleware(app: FastAPI) -> None:
    app.add_middleware(_PromMiddleware)
