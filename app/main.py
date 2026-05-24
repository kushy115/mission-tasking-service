"""FastAPI application factory + lifespan.

Wires up:
- structured logging, OTel tracing, Prometheus metrics
- the compiled LangGraph (compile flow)
- the API routes (/v1/missions:compile, :approve, :verify, /healthz, /readyz)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.api.routes import router as api_router
from app.config import get_settings
from app.graph.builder import build_graph
from app.observability.logging import configure_logging
from app.observability.metrics import register_metrics_middleware
from app.observability.tracing import configure_tracing

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(app)
    log.info("compiling mission graph", extra={"env": settings.env})

    # Prefer Postgres checkpointing in production; fall back to in-memory if the
    # checkpoint connection cannot be established (e.g. local dev without DB).
    checkpointer = None
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        cp_ctx = PostgresSaver.from_conn_string(settings.checkpoint_database_url)
        checkpointer = cp_ctx.__enter__()
        try:
            checkpointer.setup()
        except Exception as e:  # noqa: BLE001
            log.warning("PostgresSaver.setup failed; tables may already exist: %s", e)
        app.state._checkpoint_ctx = cp_ctx
    except Exception as e:  # noqa: BLE001
        log.warning("Postgres checkpointing unavailable, using MemorySaver: %s", e)

    app.state.compile_graph = build_graph(checkpointer=checkpointer)
    log.info("startup complete")
    try:
        yield
    finally:
        cp_ctx = getattr(app.state, "_checkpoint_ctx", None)
        if cp_ctx is not None:
            cp_ctx.__exit__(None, None, None)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mission Tasking Service",
        version="0.1.0",
        description="Compiles natural-language operator commands into validated drone mission plans.",
        lifespan=lifespan,
    )
    register_metrics_middleware(app)
    app.include_router(api_router)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Single-page UI for compiling missions in a browser.
    # Lives in app/static/index.html; the file gets copied into the Docker
    # image because the Dockerfile copies the whole `app/` tree.
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def ui_index() -> FileResponse:
            return FileResponse(str(static_dir / "index.html"))

    return app


app = create_app()
