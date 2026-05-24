"""Graph nodes for the compile flow.

intake → plan → validate → (repair → plan)* → finalize | clarify | REJECTED

The `plan` node is the only one that calls the LLM. Every other node is
deterministic. The repair loop is bounded by settings.repair_loop_cap (default
3) — a deliberate safety/cost ceiling; document changes here.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import json
import time

from langchain.chat_models import init_chat_model

from app.config import get_settings
from app.observability.metrics import (
    LLM_CALLS_TOTAL,
    LLM_DURATION,
    PLAN_BATTERY_PCT,
    PLAN_DURATION,
    PLAN_LEGS,
    VALIDATION_VIOLATIONS,
)
from app.geo.store import get_engine, load_drone_profile, load_geo_context, save_mission
from app.graph.state import CompileState
from app.schemas.enums import MissionStatus
from app.schemas.plan import MissionPlan
from app.tools.planning_tools import PLANNING_TOOLS
from app.validation.kernel import validate_plan

log = logging.getLogger(__name__)


# ---- intake -----------------------------------------------------------------

# Heuristic markers for "ambiguous enough to skip planning entirely".
_AMBIGUOUS_MARKERS = ("?", "something", "stuff", "whatever", "somewhere", "around")


def intake_node(state: CompileState) -> dict[str, Any]:  # noqa: D401
    log.info("→ intake (area=%s)", state.get("area_id"))
    """Normalize the request and load the geo context for the requested area.

    If the command is hopelessly underspecified, short-circuit to clarify.
    Operator clearance below STANDARD short-circuits to REJECTED.
    """
    engine = get_engine()
    area_id = state["area_id"]
    try:
        geo = load_geo_context(engine, area_id)
    except KeyError:
        return {
            "status": MissionStatus.REJECTED.value,
            "rejection_reasons": [f"unknown area_id '{area_id}'"],
        }

    geo_serialized = {
        "area_id": geo.area_id,
        "boundary_lonlat": list(geo.boundary.exterior.coords),
        "nfzs": [list(p.exterior.coords) for p in geo.nfz_polygons],
        "altitude_ceiling_m": geo.altitude_ceiling_m,
        "home_lon": geo.home_point[0],
        "home_lat": geo.home_point[1],
    }

    cmd = (state.get("raw_command") or "").strip()
    if len(cmd) < 5 or any(m in cmd.lower() for m in _AMBIGUOUS_MARKERS):
        return {
            "geo_context": geo_serialized,
            "status": MissionStatus.NEEDS_CLARIFICATION.value,
            "clarification_questions": [
                "What is the operational objective (patrol, search, inspection)?",
                "Which specific area or asset within the operating area?",
                "What sensor should be used (EO, IR, none)?",
            ],
        }

    if state.get("operator_clearance", "STANDARD") not in ("STANDARD", "ELEVATED", "ADMIN"):
        return {
            "geo_context": geo_serialized,
            "status": MissionStatus.REJECTED.value,
            "rejection_reasons": ["insufficient operator clearance"],
        }

    return {
        "geo_context": geo_serialized,
        "repair_attempts": 0,
    }


# ---- plan -------------------------------------------------------------------
#
# IMPORTANT ARCHITECTURE NOTE — read before changing this file.
#
# The `plan` node uses a DIRECT LLM call (one llm.invoke per attempt), NOT a
# LangChain `create_agent` tool-calling agent. This is a deliberate change from
# the original CLAUDE.md design. Why we moved away from the agent:
#
#   - The agent loop (create_agent + tools + response_format=MissionPlan)
#     made 5+ LLM calls per compile request: one per tool decision, plus a
#     final structured-output call. That is slow, expensive, and on free-tier
#     Gemini blew through the 20 RPD quota in 2–3 compiles.
#   - The `response_format=Pydantic` + Gemini tool-calling combo was fragile:
#     when the model finally produced a final response after transient 503s,
#     the structured-output parser would silently kill the uvicorn worker
#     (no traceback, exit code 0). Hard to debug, harder to fix upstream.
#   - We don't actually NEED dynamic tool calling here. The geo context, the
#     drone profile, and the area boundary are all known at plan-time — we
#     can inline them into the prompt. The deterministic kernel (validate_node)
#     is the source of truth on safety anyway; the LLM does NOT need to "check"
#     anything mid-loop.
#
# So: one prompt in, one JSON object out, manual json.loads + Pydantic
# validation that we control. If the plan fails the safety kernel, the repair
# loop re-prompts with the violation list. ~1 LLM call per repair pass instead
# of 5+. Faster, cheaper, more reliable across providers, easier to reason
# about. The cost is that the LLM cannot dynamically call a tool like
# lookup_search_pattern — but we can deterministically generate that geometry
# in the node itself if needed in the future.

_SYSTEM_PROMPT = """You are the planner for the Mission Tasking Service.

Given a natural-language operator command + the operating area's geofence + the
drone profile, produce ONE mission plan as a single JSON object matching the
schema below. No prose, no markdown — JSON only, starting with `{` and ending
with `}`.

REQUIRED schema:
{
  "mission_id": "<short slug>",
  "area_id": "<echo back the input area_id>",
  "status": "READY_FOR_APPROVAL" | "NEEDS_CLARIFICATION" | "REJECTED",
  "legs": [
    {
      "leg_type": "TRANSIT" | "SEARCH_PATTERN" | "LOITER" | "SENSOR_TASK" | "RETURN_TO_BASE",
      "geometry": [{"lat": <float>, "lon": <float>, "alt_m": <float>}, ...],
      "pattern_name": "lawnmower" | "expanding_square" | "sector" | null,
      "sensor_mode": "EO" | "IR" | "OFF" | null,
      "est_duration_s": <float>,
      "est_battery_pct": <float>
    }
  ],
  "total_duration_s": <float>,
  "total_battery_pct": <float>,
  "battery_reserve_pct": <float>,
  "reasoning_trace": "<one paragraph explaining the choices>",
  "clarification_questions": [<strings>] (only if status=NEEDS_CLARIFICATION),
  "rejection_reasons": [<strings>]      (only if status=REJECTED)
}

HARD CONSTRAINTS (the deterministic safety kernel will check these and reject
your plan if they fail):
1. Every waypoint MUST be inside the geofence boundary polygon.
2. NO leg may cross a no-fly-zone polygon.
3. Every altitude MUST be at or below the ceiling AND at or above 20m AGL.
4. Battery reserve at landing MUST be ≥ 20%.
5. Total duration MUST be ≤ the drone's rated endurance.
6. The LAST leg MUST be a RETURN_TO_BASE back to the home point.

STATUS DECISION RULES — read carefully, pick the strictest applicable:
- status=REJECTED is REQUIRED (not optional) when:
    * the command names an NFZ asset or location and asks to fly directly
      over / above / through it (e.g. "fly over the grain silo", "cross the
      barn", "pass through both NFZs"), OR
    * the command exceeds the drone's rated endurance, OR
    * the command asks you to ignore the geofence, disable safety, fly above
      the ceiling, or otherwise override the hard constraints.
  When you pick REJECTED you MUST populate rejection_reasons with at least one
  concrete reason. Do NOT pick NEEDS_CLARIFICATION as a way to hedge — if the
  command is unsafe as stated, reject it.
- status=NEEDS_CLARIFICATION is for cases where the command is so vague that
  you genuinely cannot pick a target, sensor, altitude, or pattern (e.g. "go
  check something out there", "find stuff"). It is NOT for unsafe commands.
- status=READY_FOR_APPROVAL is for everything else — produce a real plan.

LEG TYPE RULES — picking the wrong leg type is the #1 cause of kernel rejection:
- Use leg_type=TRANSIT for moving between points, including PERIMETER PATROLS.
  A perimeter patrol is flying AROUND the boundary, not sweeping it for
  coverage. Sample 4–8 waypoints along the boundary corners at the requested
  altitude. The kernel does NOT check sensor coverage for TRANSIT legs.
- Use leg_type=SEARCH_PATTERN ONLY for grid-coverage sweeps where you need to
  see every square meter of an area (e.g. "search the yard", "sweep the
  field"). The kernel WILL check that consecutive parallel tracks are spaced
  no further apart than the sensor's swath width at the chosen altitude —
  if you pick SEARCH_PATTERN, lay out tracks tightly (e.g. spacing ≤ 30m at
  60m altitude for EO).
- Use leg_type=LOITER for hovering in place.
- Use leg_type=RETURN_TO_BASE for the final leg only.

OTHER GUIDANCE:
- Always end with a RETURN_TO_BASE leg going to (home_lat, home_lon) at a
  descent altitude (e.g. 30m).
- Pick reasonable durations: a small-yard perimeter patrol is ~2–5 minutes.
- Round coordinates to 6 decimal places.
"""


_llm = None


def _llm_lazy() -> Any:
    """Build (and cache) a chat model routed to whichever provider is configured.

    Direct LLM call — no agent abstraction, no tool calling. Simpler, faster,
    fewer moving parts. Geo + drone context is inlined into the prompt so the
    model has everything it needs in one shot.
    """
    global _llm
    if _llm is not None:
        return _llm
    settings = get_settings()
    provider = settings.llm_provider.lower()
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "model_provider": provider,
        "temperature": settings.llm_temperature,
    }
    if provider == "google_genai":
        kwargs.update(
            max_output_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            google_api_key=settings.api_key,
        )
    elif provider in ("anthropic", "openai"):
        kwargs.update(
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            api_key=settings.api_key,
        )
    else:
        kwargs["api_key"] = settings.api_key
    _llm = init_chat_model(**kwargs)
    return _llm


def _strip_to_json(text: str) -> str:
    """Pull the first {...} JSON object out of the model's response, tolerating
    leading prose or ```json fences."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)
        s = s[1] if len(s) >= 2 else text
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text  # let the JSON parser raise a meaningful error
    return s[start : end + 1]


def plan_node(state: CompileState) -> dict[str, Any]:
    """Direct LLM call that returns a MissionPlan as JSON.

    On repair passes, the deterministic kernel's violation list is injected so
    the model can correct course. We keep the loop cap from the graph builder.
    """
    log.info("→ plan (repair_attempts=%s)", state.get("repair_attempts", 0))
    llm = _llm_lazy()
    repair_attempts = state.get("repair_attempts", 0)
    errors = state.get("validation_errors") or []
    geo = state["geo_context"]

    user_msg_parts = [
        f"COMMAND: {state['raw_command']}",
        f"AREA_ID: {state['area_id']}",
        f"OPERATOR_CLEARANCE: {state.get('operator_clearance', 'STANDARD')}",
        f"DRONE_STATE: {state.get('drone_state')}",
        "GEO_CONTEXT:",
        f"  boundary_lonlat (polygon vertices): {geo['boundary_lonlat']}",
        f"  nfz_polygons:                       {geo['nfzs']}",
        f"  altitude_ceiling_m:                 {geo['altitude_ceiling_m']}",
        f"  home_lon, home_lat:                 {geo['home_lon']}, {geo['home_lat']}",
    ]
    if repair_attempts > 0 and errors:
        user_msg_parts.append(
            "\nREPAIR PASS — the deterministic safety kernel rejected your previous "
            "plan with these violations. Fix every one of them:\n  - "
            + "\n  - ".join(errors)
        )
    user_msg = "\n".join(user_msg_parts)

    settings = get_settings()
    provider, model = settings.llm_provider, settings.llm_model
    start = time.perf_counter()
    try:
        ai_msg = llm.invoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
    except Exception as e:  # noqa: BLE001
        log.exception("llm.invoke raised: %s", e)
        outcome = "rate_limited" if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) else "error"
        LLM_CALLS_TOTAL.labels(provider, model, outcome).inc()
        LLM_DURATION.labels(provider, model).observe(time.perf_counter() - start)
        return {
            "draft_plan": None,
            "validation_errors": [f"llm error: {type(e).__name__}: {e}"],
        }
    LLM_CALLS_TOTAL.labels(provider, model, "ok").inc()
    LLM_DURATION.labels(provider, model).observe(time.perf_counter() - start)

    raw_text = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
    log.info("← plan got %d chars", len(raw_text) if isinstance(raw_text, str) else -1)

    try:
        payload = json.loads(_strip_to_json(raw_text))
    except Exception as e:  # noqa: BLE001
        log.warning("plan JSON parse failed: %s; first 200 chars: %r", e, raw_text[:200])
        return {
            "draft_plan": None,
            "validation_errors": [f"plan was not valid JSON: {e}"],
        }

    payload.setdefault("mission_id", str(uuid.uuid4()))
    payload["area_id"] = state["area_id"]

    try:
        structured = MissionPlan.model_validate(payload)
    except Exception as e:  # noqa: BLE001
        log.warning("MissionPlan validation failed: %s", e)
        return {
            "draft_plan": None,
            "validation_errors": [f"plan failed schema: {e}"],
        }
    return {"draft_plan": structured.model_dump()}


# ---- validate ---------------------------------------------------------------


def validate_node(state: CompileState) -> dict[str, Any]:
    """Deterministic kernel pass. No LLM. Source of truth on safety."""
    log.info("→ validate")
    raw = state.get("draft_plan")
    if raw is None:
        return {"validation_errors": state.get("validation_errors") or ["no draft_plan to validate"]}

    plan = MissionPlan.model_validate(raw)

    # Routes that came from intake/plan as REJECTED or NEEDS_CLARIFICATION are
    # not safety-validated — they bypass to finalize/clarify.
    if plan.status in (MissionStatus.REJECTED, MissionStatus.NEEDS_CLARIFICATION):
        return {
            "draft_plan": plan.model_dump(),
            "validation_errors": [],
            "constraints_report": None,
        }

    engine = get_engine()
    geo = load_geo_context(engine, state["area_id"])
    drone_id = state["drone_state"]["drone_profile_id"]
    profile = load_drone_profile(engine, drone_id)
    starting_pct = float(state["drone_state"].get("battery_pct", 100.0))

    violations, report = validate_plan(plan, geo, profile, starting_battery_pct=starting_pct)

    # Record each FAILED check so the dashboard can show which safety checks
    # trip most often (e.g. coverage-gap dominates → planner's pattern logic
    # is weak; battery dominates → planner is over-budgeting).
    for check_name in (
        "inside_geofence",
        "avoids_nfz",
        "battery_within_budget",
        "within_endurance",
        "sensor_coverage_adequate",
        "ends_with_rtb",
    ):
        if not getattr(report, check_name):
            VALIDATION_VIOLATIONS.labels(check_name).inc()

    return {
        "draft_plan": plan.model_dump(),
        "validation_errors": violations,
        "constraints_report": report.model_dump(),
    }


# ---- repair / clarify -------------------------------------------------------


def repair_node(state: CompileState) -> dict[str, Any]:
    """Increment the repair counter; the conditional edge sends us back to plan."""
    log.info("→ repair (n=%s)", state.get("repair_attempts", 0) + 1)
    return {"repair_attempts": state.get("repair_attempts", 0) + 1}


def clarify_node(state: CompileState) -> dict[str, Any]:
    """If intake didn't already populate questions, generate generic ones from
    the violations."""
    questions = state.get("clarification_questions") or []
    if not questions:
        questions = [
            "Could you specify the target area or asset more precisely?",
            "Which sensor mode should be used (EO, IR, OFF)?",
            "What is the preferred altitude band?",
        ]
    return {
        "status": MissionStatus.NEEDS_CLARIFICATION.value,
        "clarification_questions": questions,
    }


def reject_node(state: CompileState) -> dict[str, Any]:
    reasons = state.get("rejection_reasons") or []
    if not reasons:
        reasons = (
            state.get("validation_errors")
            or ["planner could not produce a safe plan within the repair budget"]
        )
    return {
        "status": MissionStatus.REJECTED.value,
        "rejection_reasons": reasons,
    }


# ---- finalize ---------------------------------------------------------------


def finalize_node(state: CompileState) -> dict[str, Any]:  # noqa: D401
    log.info("→ finalize (status=%s)", state.get("status"))
    """Assemble the final MissionPlan, persist it, and mark ready for approval.

    The interrupt happens at the edge from finalize → END (configured in the
    builder) so the operator gates approval out-of-band.

    When intake/validate short-circuit to NEEDS_CLARIFICATION or REJECTED there
    may be no draft_plan; synthesize an empty plan envelope so callers always
    receive a consistent shape.
    """
    raw = state.get("draft_plan")
    if raw is None:
        plan = MissionPlan(
            mission_id=str(uuid.uuid4()),
            area_id=state.get("area_id", ""),
            status=MissionStatus.REJECTED,
        )
    else:
        plan = MissionPlan.model_validate(raw)

    state_status = state.get("status")
    if state_status == MissionStatus.NEEDS_CLARIFICATION.value:
        plan.status = MissionStatus.NEEDS_CLARIFICATION
        plan.clarification_questions = state.get("clarification_questions") or []
    elif state_status == MissionStatus.REJECTED.value:
        plan.status = MissionStatus.REJECTED
        plan.rejection_reasons = state.get("rejection_reasons") or []
    elif raw is None:
        plan.status = MissionStatus.REJECTED
        plan.rejection_reasons = ["no plan produced"]
    else:
        plan.status = MissionStatus.READY_FOR_APPROVAL

    # Record plan-shape metrics only for plans that actually flew through the
    # planner (skip NEEDS_CLARIFICATION/REJECTED-from-intake — those have no
    # legs and would skew the histograms toward zero).
    if plan.status == MissionStatus.READY_FOR_APPROVAL and plan.legs:
        PLAN_LEGS.observe(len(plan.legs))
        PLAN_BATTERY_PCT.observe(plan.total_battery_pct)
        PLAN_DURATION.observe(plan.total_duration_s)

    try:
        save_mission(get_engine(), plan.model_dump())
    except Exception as e:  # noqa: BLE001
        log.warning("save_mission skipped: %s", e)

    return {"draft_plan": plan.model_dump(), "status": plan.status.value}
