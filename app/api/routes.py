"""HTTP routes. No business logic — handlers delegate to graph/services."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import text

from app.api.models import (
    AreaUpsertRequest,
    AreaUpsertResponse,
    ApprovalRequest,
    ApprovalResponse,
    CompileRequest,
    CompileResponse,
    VerifyResponse,
)
from app.export import EXPORT_FORMATS, render_export
from app.geo.store import (
    delete_area,
    get_engine,
    list_areas,
    list_drones,
    list_missions,
    load_drone_profile,
    load_mission,
    set_mission_approval,
    snap_home_to_boundary,
    upsert_area,
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


@router.post("/v1/areas", response_model=AreaUpsertResponse)
def upsert_area_endpoint(req: AreaUpsertRequest) -> AreaUpsertResponse:
    """Create or replace an operating area drawn in the UI.

    See docs/DESIGN_DECISIONS.md §5: server-side polygon repair + snap-home-to-
    boundary. Upsert semantics: posting with an existing area_id overwrites.
    """
    from shapely.geometry import Point, shape

    try:
        boundary = shape(req.boundary)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid boundary geometry: {e}") from e
    if not boundary.is_valid:
        boundary = boundary.buffer(0)
    if boundary.is_empty or boundary.geom_type != "Polygon":
        raise HTTPException(status_code=400, detail="boundary must be a non-empty Polygon")

    nfz_polys = []
    for i, nfz in enumerate(req.nfzs):
        try:
            poly = shape(nfz)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"nfz[{i}] invalid: {e}") from e
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            raise HTTPException(status_code=400, detail=f"nfz[{i}] must be a non-empty Polygon")
        if not boundary.contains(poly) and not boundary.intersects(poly):
            raise HTTPException(
                status_code=400,
                detail=f"nfz[{i}] is entirely outside the boundary (unreachable)",
            )
        nfz_polys.append(poly)

    home_pt = Point(req.home_lon, req.home_lat)
    home_was_snapped = not boundary.exterior.distance(home_pt) < 1e-9
    snapped_lon, snapped_lat = snap_home_to_boundary(boundary, req.home_lon, req.home_lat)

    # Assemble a GeoJSON FeatureCollection that mirrors the seed format so
    # upsert_area can reuse the same path.
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": boundary.__geo_interface__,
                "properties": {"role": "boundary", "home": [snapped_lon, snapped_lat]},
            },
            *[
                {
                    "type": "Feature",
                    "geometry": poly.__geo_interface__,
                    "properties": {"role": "nfz"},
                }
                for poly in nfz_polys
            ],
        ],
    }
    upsert_area(get_engine(), req.area_id, fc, req.ceiling_m)
    return AreaUpsertResponse(
        area_id=req.area_id,
        home_lon=snapped_lon,
        home_lat=snapped_lat,
        nfz_count=len(nfz_polys),
        home_was_snapped=home_was_snapped,
    )


@router.delete("/v1/areas/{area_id}")
def delete_area_endpoint(area_id: str) -> dict[str, Any]:
    """Hard-delete an operating area. Cascades to its NFZs."""
    if not delete_area(get_engine(), area_id):
        raise HTTPException(status_code=404, detail=f"area {area_id} not found")
    return {"area_id": area_id, "deleted": True}


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


def _invoke_compile_graph(
    graph: Any,
    *,
    command: str,
    area_id: str,
    clearance: str,
    drone_state: dict,
    request_id: str | None,
    alternatives_requested: bool,
    multi_drone_slot: dict | None = None,
) -> dict:
    """One graph invocation for one drone. Returns the final state dict."""
    thread_id = (request_id or str(uuid.uuid4())) + (
        f"-d{multi_drone_slot['index']}" if multi_drone_slot else ""
    )
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
    initial_state = {
        "raw_command": command,
        "area_id": area_id,
        "operator_clearance": clearance,
        "drone_state": drone_state,
        "request_id": request_id,
        "messages": [],
        "repair_attempts": 0,
        "alternatives_requested": alternatives_requested,
        "multi_drone_slot": multi_drone_slot,
    }
    with COMPILE_DURATION.time():
        return graph.invoke(initial_state, config=config)


@router.post("/v1/missions:compile", response_model=CompileResponse)
def compile_mission(req: CompileRequest, request: Request) -> CompileResponse:
    """Compile a natural-language command into a validated MissionPlan.

    With `drone_ids` set, runs one planner invocation per drone with
    server-assigned altitude layers + takeoff staggering (see
    DESIGN_DECISIONS §9). Sibling plans are reported in `group_plans`.
    """
    graph = request.app.state.compile_graph
    log.info(
        "compile request area=%s drones=%s command=%r",
        req.area_id, req.drone_ids or [req.drone_state.drone_profile_id], req.command[:120],
    )

    # --- multi-drone path ---
    if req.drone_ids and len(req.drone_ids) >= 1:
        from app.graph.multi_drone import assign_slots

        slots = assign_slots(req.drone_ids)
        group_plans: list[MissionPlan] = []
        total_repairs = 0
        for slot in slots:
            drone_state = {
                "drone_profile_id": slot.drone_profile_id,
                "battery_pct": req.drone_state.battery_pct,
            }
            try:
                final = _invoke_compile_graph(
                    graph,
                    command=req.command,
                    area_id=req.area_id,
                    clearance=req.operator_clearance,
                    drone_state=drone_state,
                    request_id=req.request_id,
                    alternatives_requested=False,  # group mode disables alts
                    multi_drone_slot=slot.to_dict(),
                )
            except Exception as e:  # noqa: BLE001
                COMPILE_REQUESTS.labels(status="error").inc()
                log.exception("multi-drone compile failed for slot %d: %s", slot.index, e)
                raise HTTPException(status_code=500, detail=f"slot {slot.index}: {e}") from e
            raw = final.get("draft_plan")
            if raw is None:
                continue
            plan_i = MissionPlan.model_validate(raw)
            total_repairs += int(final.get("repair_attempts", 0))
            group_plans.append(plan_i)
            COMPILE_REQUESTS.labels(status=plan_i.status.value).inc()

        if not group_plans:
            raise HTTPException(status_code=500, detail="multi-drone: no plans produced")

        primary = group_plans[0]
        REPAIR_LOOPS.observe(total_repairs)
        return CompileResponse(
            plan=primary,
            repair_loops=total_repairs,
            awaiting_approval=all(p.status == MissionStatus.READY_FOR_APPROVAL for p in group_plans),
            group_plans=group_plans,
        )

    # --- single-drone path (legacy) ---
    try:
        final_state = _invoke_compile_graph(
            graph,
            command=req.command,
            area_id=req.area_id,
            clearance=req.operator_clearance,
            drone_state=req.drone_state.model_dump(),
            request_id=req.request_id,
            alternatives_requested=bool(req.alternatives),
        )
    except Exception as e:  # noqa: BLE001
        COMPILE_REQUESTS.labels(status="error").inc()
        log.exception("compile graph raised: %s", e)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    log.info("compile graph done status=%s", final_state.get("status"))
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

    raw_alts = final_state.get("alternatives") or []
    alternatives = [MissionPlan.model_validate(a) for a in raw_alts]

    return CompileResponse(
        plan=plan,
        repair_loops=repair_loops,
        awaiting_approval=(plan.status == MissionStatus.READY_FOR_APPROVAL),
        alternatives=alternatives,
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
