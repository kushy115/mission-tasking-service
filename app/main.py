"""FastAPI application factory + lifespan.

Wires up:
- structured logging, OTel tracing, Prometheus metrics
- the compiled LangGraph (compile flow)
- the API routes (/v1/missions:compile, :approve, :verify, /healthz, /readyz)
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.api.routes import router as api_router
from app.config import get_settings
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
    if os.getenv("MTS_SKIP_GRAPH_STARTUP") == "1":
        app.state.compile_graph = None
        log.info("mission graph startup skipped")
        yield
        return

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

    from app.graph.builder import build_graph

    app.state.compile_graph = build_graph(checkpointer=checkpointer)
    log.info("startup complete")
    try:
        yield
    finally:
        cp_ctx_close: Any = getattr(app.state, "_checkpoint_ctx", None)
        if cp_ctx_close is not None:
            cp_ctx_close.__exit__(None, None, None)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Perception",
        version="0.1.0",
        description="Compiles natural-language operator commands into validated drone mission plans.",
        lifespan=lifespan,
    )
    register_metrics_middleware(app)
    app.include_router(api_router)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.websocket("/ws/missions/{mission_id}/sim")
    async def mission_sim_socket(
        websocket: WebSocket,
        mission_id: str,
        speed: float = 20.0,
    ) -> None:
        """Live telemetry stream for a saved mission (DD-§10).

        The server walks the plan tick-by-tick against the same physics model the
        validator uses and emits flat-JSON telemetry frames. `speed` is the time-
        compression factor. For multi-drone group missions the UI opens one socket
        per drone so they animate in parallel.
        """
        from app.geo.store import (
            get_engine,
            load_drone_profile,
            load_geo_context,
            load_mission_detail,
        )
        from app.schemas.plan import MissionPlan
        from app.sim.stream import telemetry_stream

        await websocket.accept()
        try:
            detail = load_mission_detail(get_engine(), mission_id)
            if detail is None:
                await websocket.send_json({"error": f"mission {mission_id} not found"})
                await websocket.close()
                return
            plan = MissionPlan.model_validate(detail["plan"])
            # Re-fly against the SAME airframe the plan was compiled for — using
            # the wrong profile makes cruise speed / battery diverge and the
            # animation no longer matches the actual plan.
            profile_id = detail.get("drone_profile_id") or "long-endurance-quad"
            try:
                profile = load_drone_profile(get_engine(), profile_id)
            except Exception:  # noqa: BLE001 — profile renamed/removed; fall back
                profile = load_drone_profile(get_engine(), "long-endurance-quad")
            try:
                geo = load_geo_context(get_engine(), plan.area_id)
            except Exception:  # noqa: BLE001
                geo = None
            speed = max(0.5, min(float(speed or 20.0), 200.0))

            async for tframe in telemetry_stream(plan, profile, geo, speed=speed):
                try:
                    await websocket.send_json({"kind": "telemetry", **tframe.to_dict()})
                except Exception:  # noqa: BLE001
                    log.info("ws sim stream send failed; closing")
                    return
        except WebSocketDisconnect:
            log.info("ws sim stream client disconnected")
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()

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
