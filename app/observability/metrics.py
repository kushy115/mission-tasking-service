"""Prometheus metrics. Mounted at /metrics by app.main.create_app.

See `docs/observability.md` for the interpretation guide — what each metric
means, healthy ranges, and what to investigate when one moves.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

# ---- compile-level (the user-visible workflow) ------------------------------

COMPILE_REQUESTS = Counter(
    "mts_compile_requests_total",
    "Compile requests served, labeled by final mission status.",
    labelnames=("status",),
)
COMPILE_DURATION = Histogram(
    "mts_compile_duration_seconds",
    "End-to-end latency of compile requests (HTTP in -> HTTP out).",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
REPAIR_LOOPS = Histogram(
    "mts_repair_loops",
    "Number of repair iterations per compile call. 0 = first try was safe.",
    buckets=(0, 1, 2, 3, 4, 5),
)
REJECTIONS_TOTAL = Counter("mts_rejections_total", "Plans returned as REJECTED.")
CLARIFICATIONS_TOTAL = Counter("mts_clarifications_total", "Plans returned as NEEDS_CLARIFICATION.")

# ---- LLM-level (cost + reliability of the language model itself) ------------

LLM_CALLS_TOTAL = Counter(
    "mts_llm_calls_total",
    "LLM API calls, labeled by provider, model, and outcome (ok|error|rate_limited).",
    labelnames=("provider", "model", "outcome"),
)
LLM_DURATION = Histogram(
    "mts_llm_duration_seconds",
    "Latency of a single LLM API call (wall-clock).",
    labelnames=("provider", "model"),
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)

# ---- safety-kernel (which checks fail most often) ---------------------------

VALIDATION_VIOLATIONS = Counter(
    "mts_validation_violations_total",
    "Times the deterministic safety kernel flagged a violation, by check name.",
    labelnames=("check",),
)

# ---- plan shape (what kinds of plans is the planner producing?) -------------

PLAN_LEGS = Histogram(
    "mts_plan_legs",
    "Number of legs per finalized plan.",
    buckets=(1, 2, 3, 4, 5, 7, 10, 15),
)
PLAN_BATTERY_PCT = Histogram(
    "mts_plan_battery_pct",
    "Estimated battery percentage consumed by a finalized plan.",
    buckets=(5, 10, 20, 30, 50, 70, 90),
)
PLAN_DURATION = Histogram(
    "mts_plan_duration_seconds",
    "Estimated flight duration (seconds) of a finalized plan.",
    buckets=(60, 120, 300, 600, 1200, 1800, 2700),
)

# ---- HTTP layer -------------------------------------------------------------

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
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        dur = time.perf_counter() - start
        route = request.scope.get("route")
        path: str = route.path if route is not None else request.url.path
        HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        HTTP_DURATION.labels(request.method, path).observe(dur)
        return response


def register_metrics_middleware(app: FastAPI) -> None:
    app.add_middleware(_PromMiddleware)
