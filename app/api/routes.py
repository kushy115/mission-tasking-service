"""HTTP routes. No business logic — handlers delegate to graph/services."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import text

from app.api.models import (
    ApprovalRequest,
    ApprovalResponse,
    CompileRequest,
    CompileResponse,
    VerifyResponse,
)
from app.export import EXPORT_FORMATS, render_export
from app.geo.store import (
    get_engine,
    list_areas,
    list_drones,
    list_missions,
    load_drone_profile,
    load_mission,
    set_mission_approval,
)
from app.observability.metrics import (
    CLARIFICATIONS_TOTAL,
    COMPILE_DURATION,
    COMPILE_REQUESTS,
    REJECTIONS_TOTAL,
    REPAIR_LOOPS,
)
from app.schemas.enums import MissionStatus
from app.schemas.plan import MissionPlan
from app.sim.executor import simulate_execution

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/areas")
def list_areas_endpoint() -> list[dict]:
    """Return all operating areas with boundary + NFZs as GeoJSON. Used by the UI."""
    return list_areas(get_engine())


@router.get("/v1/drones")
def list_drones_endpoint() -> list[dict]:
    """Return all drone profiles. Used by the UI."""
    return list_drones(get_engine())


@router.get("/readyz")
def readyz() -> dict[str, Any]:
    """Liveness + dependency check. Returns 503 if Postgres is unreachable."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"postgres unavailable: {e}") from e
    return {"status": "ready"}


@router.post("/v1/missions:compile", response_model=CompileResponse)
def compile_mission(req: CompileRequest, request: Request) -> CompileResponse:
    """Compile a natural-language command into a validated MissionPlan."""
    graph = request.app.state.compile_graph
    thread_id = req.request_id or str(uuid.uuid4())
    # Cap graph steps so a runaway agent loop fails fast with a clear error
    # instead of hanging until the container is killed.
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}

    initial_state = {
        "raw_command": req.command,
        "area_id": req.area_id,
        "operator_clearance": req.operator_clearance,
        "drone_state": req.drone_state.model_dump(),
        "request_id": req.request_id,
        "messages": [],
        "repair_attempts": 0,
    }

    log.info("compile request thread_id=%s area=%s command=%r", thread_id, req.area_id, req.command[:120])
    try:
        with COMPILE_DURATION.time():
            final_state = graph.invoke(initial_state, config=config)
    except Exception as e:  # noqa: BLE001
        COMPILE_REQUESTS.labels(status="error").inc()
        log.exception("compile graph raised: %s", e)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    log.info("compile graph done thread_id=%s status=%s", thread_id, final_state.get("status"))
    raw_plan = final_state.get("draft_plan")
    if raw_plan is None:
        REJECTIONS_TOTAL.inc()
        COMPILE_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="planner produced no plan")

    plan = MissionPlan.model_validate(raw_plan)
    repair_loops = int(final_state.get("repair_attempts", 0))
    REPAIR_LOOPS.observe(repair_loops)

    if plan.status == MissionStatus.REJECTED:
        REJECTIONS_TOTAL.inc()
    elif plan.status == MissionStatus.NEEDS_CLARIFICATION:
        CLARIFICATIONS_TOTAL.inc()
    COMPILE_REQUESTS.labels(status=plan.status.value).inc()

    return CompileResponse(
        plan=plan,
        repair_loops=repair_loops,
        awaiting_approval=(plan.status == MissionStatus.READY_FOR_APPROVAL),
    )


@router.post("/v1/missions:approve", response_model=ApprovalResponse)
def approve_mission(req: ApprovalRequest) -> ApprovalResponse:
    """Operator approval gate. Resolves the interrupt by recording the decision."""
    engine = get_engine()
    raw = load_mission(engine, req.mission_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"mission {req.mission_id} not found")
    plan = MissionPlan.model_validate(raw)
    if plan.status != MissionStatus.READY_FOR_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"mission status is {plan.status.value}, cannot approve",
        )
    set_mission_approval(engine, req.mission_id, req.approve, req.operator_note)
    final = "APPROVED" if req.approve else "OPERATOR_REJECTED"
    return ApprovalResponse(
        mission_id=req.mission_id,
        final_status=final,
        operator_note=req.operator_note,
    )


@router.get("/v1/missions")
def list_missions_endpoint(limit: int = 50) -> list[dict]:
    """List recent missions (newest first). See docs/DESIGN_DECISIONS.md §1."""
    limit = max(1, min(int(limit), 200))
    return list_missions(get_engine(), limit=limit)


@router.get("/v1/missions/{mission_id}")
def get_mission_endpoint(mission_id: str) -> dict:
    """Return the full stored MissionPlan JSON for one mission."""
    raw = load_mission(get_engine(), mission_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"mission {mission_id} not found")
    return raw


@router.get("/v1/missions/{mission_id}/export")
def export_mission_endpoint(mission_id: str, format: str = "kml") -> Response:
    """Export a mission plan as KML/GPX/DJI. See docs/DESIGN_DECISIONS.md §2."""
    fmt = format.lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown format '{format}'; supported: {sorted(EXPORT_FORMATS)}",
        )
    raw = load_mission(get_engine(), mission_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"mission {mission_id} not found")
    plan = MissionPlan.model_validate(raw)
    body, spec = render_export(plan, fmt)
    filename = f"mission-{mission_id}.{spec.extension}"
    return Response(
        content=body,
        media_type=spec.media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/v1/missions/{mission_id}:verify", response_model=VerifyResponse)
def verify_mission(mission_id: str) -> VerifyResponse:
    """Run the approved plan through the deterministic execution verifier."""
    engine = get_engine()
    raw = load_mission(engine, mission_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"mission {mission_id} not found")
    plan = MissionPlan.model_validate(raw)

    # Use the drone profile that the operator would actually fly. For the demo
    # we accept the area-default — production would persist this with the plan.
    profile = load_drone_profile(engine, "long-endurance-quad")
    result = simulate_execution(plan, profile)
    return VerifyResponse(
        mission_id=mission_id,
        flown_ok=result.flown_ok,
        actual_duration_s=result.actual_duration_s,
        actual_battery_pct=result.actual_battery_pct,
        deviations=result.deviations,
    )
