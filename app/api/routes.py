"""HTTP routes. No business logic — handlers delegate to graph/services."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import text

from app.api.models import (
    AreaResearchRequest,
    AreaResearchResponse,
    AreaUpsertRequest,
    AreaUpsertResponse,
    ApprovalRequest,
    ApprovalResponse,
    CompileRequest,
    CompileResponse,
    DroneUpsertRequest,
    MissionChatRequest,
    MissionChatResponse,
    SupervisorInjectRequest,
    SupervisorInjectResponse,
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
    load_mission_detail,
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
    # LFU cap: evict the least-used non-protected area BEFORE inserting the new
    # one so we never exceed AREA_LFU_CAP. Protected (seeded) areas are immune.
    # See DESIGN_DECISIONS §8.
    from app.geo.store import evict_lfu_area_if_needed
    evicted = evict_lfu_area_if_needed(get_engine(), exclude_area_id=req.area_id)
    if evicted:
        log.info("LFU evicted area %r to make room for %r", evicted, req.area_id)

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


@router.post("/v1/drones")
def upsert_drone_endpoint(req: DroneUpsertRequest) -> dict[str, Any]:
    """Operator-authored drone profile (DD-006). Upserts into the drones table."""
    from app.geo.store import upsert_drone
    profile = req.model_dump(exclude={"profile_id"})
    upsert_drone(get_engine(), req.profile_id, profile)
    return {"profile_id": req.profile_id, "saved": True}


@router.post("/v1/areas:research", response_model=AreaResearchResponse)
def research_area_endpoint(req: AreaResearchRequest) -> AreaResearchResponse:
    """LLM-driven area research (DD-007). Returns suggested NFZs + ceiling + advisory note.

    The polygon is sent to the planner LLM with a structured-output schema so
    we get JSON back, not prose. Output is ADVISORY — the operator can edit
    every field before saving the area. The deterministic kernel then becomes
    the source of truth at compile time, as usual.
    """
    from app.graph.nodes import _llm_lazy
    import json as _json
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        coords = req.boundary.get("coordinates", [[]])[0]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad boundary: {e}") from e

    system = (
        "You are an airspace-advisory assistant. Given a polygon (lon/lat) and a "
        "home point, return ADVISORY guidance about likely flight restrictions in "
        "that area as a single JSON object — no prose, no markdown. Schema:\n"
        '{ "flight_permitted": <bool>, '
        '  "ceiling_m": <float between 50 and 400>, '
        '  "suggested_nfzs": [<GeoJSON Polygon geometry>, ...], '
        '  "notes": "<1-3 sentences explaining the choices>" }\n'
        "Suggested NFZs should be plausible obstacles inside the boundary (e.g. "
        "buildings, runways, towers) drawn as small polygons in lon/lat. If you "
        "cannot identify anything specific, return an empty list. Be CONSERVATIVE: "
        "if the area looks like it might be near an airport or restricted airspace, "
        "set flight_permitted=false."
    )
    user = (
        f"Boundary vertices (lon,lat): {coords}\n"
        f"Home point (lon,lat): {req.home_lon},{req.home_lat}\n"
        "Return ONE JSON object matching the schema."
    )
    try:
        ai = _llm_lazy().invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}") from e
    raw = ai.content if hasattr(ai, "content") else str(ai)
    # Strip ```json fences and pull the first {...} blob.
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1].lstrip()
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    blob = s[start : end + 1] if 0 <= start < end else raw
    try:
        data = _json.loads(blob)
    except Exception as e:  # noqa: BLE001
        log.warning("research JSON parse failed: %s; raw=%r", e, raw[:300])
        # Degrade gracefully — return a safe-ish default so the UI doesn't dead-end.
        data = {
            "flight_permitted": True, "ceiling_m": 120.0, "suggested_nfzs": [],
            "notes": "LLM response could not be parsed; using defaults.",
        }
    return AreaResearchResponse(
        flight_permitted=bool(data.get("flight_permitted", True)),
        ceiling_m=float(data.get("ceiling_m", 120.0)),
        suggested_nfzs=list(data.get("suggested_nfzs", [])),
        notes=str(data.get("notes", "")),
    )


@router.get("/v1/missions")
def list_missions_endpoint(limit: int = 100) -> list[dict]:
    """Return recent missions (summary fields only). Used by the UI history page."""
    limit = max(1, min(int(limit), 200))
    return list_missions(get_engine(), limit=limit)


@router.get("/v1/missions/{mission_id}")
def get_mission_endpoint(mission_id: str) -> dict:
    """Return full mission detail including the repair-draft timeline."""
    detail = load_mission_detail(get_engine(), mission_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"mission {mission_id} not found")
    return detail


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
    conversation_history: list[dict] | None = None,
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
        "conversation_history": conversation_history or [],
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
            conversation_history=req.conversation_history or None,
        )
        # LFU bookkeeping: count this access on the area (DD-008).
        try:
            from app.geo.store import touch_area_access
            touch_area_access(get_engine(), req.area_id)
        except Exception:  # noqa: BLE001
            pass
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


@router.post("/v1/missions/{mission_id}:chat", response_model=MissionChatResponse)
def chat_about_mission(mission_id: str, req: MissionChatRequest) -> MissionChatResponse:
    """Follow-up conversation with an LLM scoped to one mission.

    The plan + its critique + advisor suggestions are loaded once and injected
    into the system prompt so the LLM can reason about THIS plan specifically.
    History is stateless: the client sends every prior turn each time. This
    keeps the endpoint horizontally scalable and trace-friendly.

    Uses LangChain directly (init_chat_model + Human/AI/SystemMessage). The
    advisor and critique are NOT re-run here — this is a discussion layer on
    top of an already-produced plan.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from app.graph.nodes import _llm_lazy

    raw = load_mission(get_engine(), mission_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"mission {mission_id} not found")
    plan = MissionPlan.model_validate(raw)

    if not req.history:
        raise HTTPException(status_code=400, detail="history must contain at least one user turn")
    if req.history[-1].role != "user":
        raise HTTPException(status_code=400, detail="last history turn must be a user message")

    # Compress plan into a chat-friendly summary. We don't dump the whole
    # waypoint list — that's noise; the LLM reasons about leg-level decisions.
    leg_lines = []
    for i, leg in enumerate(plan.legs, 1):
        alt = leg.geometry[0].alt_m if leg.geometry else 0
        sensor = leg.sensor_mode.value if leg.sensor_mode else "OFF"
        leg_lines.append(
            f"  Leg {i}: {leg.leg_type.value} · {len(leg.geometry)} waypoints · "
            f"alt~{alt:.0f}m · sensor={sensor} · {leg.est_duration_s:.0f}s · "
            f"{leg.est_battery_pct:.1f}% battery"
        )

    advisor_lines = []
    if plan.advisory:
        advisor_lines.append(f"Advisor summary: {plan.advisory.summary}")
        for s in plan.advisory.suggestions:
            advisor_lines.append(
                f"  - [{s.impact} · {s.category}] {s.title} — {s.rationale}"
            )
        if plan.advisory.resource_constrained_fallback:
            advisor_lines.append(
                f"  Fallback if resource-constrained: {plan.advisory.resource_constrained_fallback}"
            )

    system_prompt = f"""You are a senior drone-operations advisor in a chat with
the operator about a SPECIFIC mission plan they just compiled. The plan has
already passed the safety kernel. Your job is to:
  - answer questions about the plan
  - propose tweaks the operator could adopt
  - explain tradeoffs (battery vs coverage, sensor choice vs visibility, etc.)
  - if the operator wants to change the plan, describe what they should re-prompt
    the main compile endpoint with — you do NOT modify the plan yourself

Stay concise (1–4 short paragraphs). Be specific about altitudes, leg numbers,
sensor modes — operators want concrete recommendations, not platitudes. Do NOT
restate the whole plan; assume the operator can see it on screen.

PLAN UNDER DISCUSSION:
  mission_id: {plan.mission_id}
  area: {plan.area_id}
  status: {plan.status.value}
  total duration: {plan.total_duration_s:.0f}s ({plan.total_duration_s/60:.1f} min)
  battery use: {plan.total_battery_pct:.1f}%  reserve at landing: {plan.battery_reserve_pct:.1f}%
  reasoning: {plan.reasoning_trace}
  legs:
{chr(10).join(leg_lines) if leg_lines else '  (no legs)'}

{chr(10).join(advisor_lines) if advisor_lines else '(no advisor output)'}
"""

    messages: list = [SystemMessage(content=system_prompt)]
    for turn in req.history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        elif turn.role == "assistant":
            messages.append(AIMessage(content=turn.content))

    try:
        ai = _llm_lazy().invoke(messages)
    except Exception as e:  # noqa: BLE001
        log.exception("mission chat LLM call failed: %s", e)
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}") from e
    reply = ai.content if hasattr(ai, "content") else str(ai)
    if isinstance(reply, list):
        # Anthropic SDK sometimes returns content as a list of blocks; flatten.
        reply = "".join(getattr(b, "text", str(b)) for b in reply)
    return MissionChatResponse(reply=str(reply).strip(), mission_id=mission_id)


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


@router.post(
    "/v1/missions/{mission_id}/sim:inject",
    response_model=SupervisorInjectResponse,
)
def inject_supervisor_event(
    mission_id: str, req: SupervisorInjectRequest
) -> SupervisorInjectResponse:
    """Inject an in-flight event for the live supervisor (DD-014).

    The active live session is identified by `(mission_id, session_id)`. If
    no such session is currently streaming, returns 404. The event is dropped
    on the queue and consumed by the supervisor on its next decision tick.
    """
    from app.supervisor.events import Event, get_queue

    q = get_queue(mission_id, req.session_id)
    if q is None:
        raise HTTPException(
            status_code=404,
            detail=f"no active live session for mission_id={mission_id!r}, "
                   f"session_id={req.session_id!r}",
        )
    try:
        event = Event.from_dict({"type": req.type, "payload": req.payload, "note": req.note})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        q.put_nowait(event)
    except Exception as e:  # noqa: BLE001 — queue full
        raise HTTPException(
            status_code=503, detail=f"event queue full: {e}",
        ) from e
    return SupervisorInjectResponse(
        accepted=True, queue_size=q.qsize(), detail=f"queued {event.type.value}",
    )
