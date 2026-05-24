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

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model

from app.config import get_settings
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

_SYSTEM_PROMPT = """You are the planning agent for the Mission Tasking Service.

You receive a natural-language operator command, the operating-area geofence,
the drone's capability profile, and (on repair passes) the list of validation
violations from the previous attempt.

Your job is to produce a structured MissionPlan that:
  1. stays inside the geofence,
  2. avoids every no-fly-zone,
  3. respects the altitude ceiling and the 20m minimum AGL floor,
  4. fits within the drone's rated endurance with at least 20% battery reserve at landing,
  5. uses sensor coverage adequately for any SEARCH_PATTERN leg,
  6. ENDS with a RETURN_TO_BASE leg back to the home point.

Tools available:
- get_geofence(area_id)
- check_path_clear(area_id, polyline_lonlat)
- estimate_battery(drone_profile_id, legs)
- get_sensor_coverage(drone_profile_id, sensor_mode, altitude_m)
- lookup_search_pattern(pattern_name, center_lat, center_lon, altitude_m, ...)
- get_weather(area_id)

Always call get_geofence first. Use lookup_search_pattern to generate search
geometry — do NOT invent it. Call estimate_battery before finalizing.

If the command is genuinely ambiguous (e.g. unspecified target, unknown asset),
set status=NEEDS_CLARIFICATION and populate clarification_questions.

If the command demands an unsafe action (entering an NFZ, exceeding endurance,
disabling the geofence), set status=REJECTED with rejection_reasons.

Otherwise, set status=READY_FOR_APPROVAL and produce ordered legs. Provide a
clear reasoning_trace explaining the choices you made.

Never emit free-text plans — your response MUST conform to the MissionPlan schema."""


def _build_planning_agent() -> Any:
    """Build the planning agent, routing to whichever LLM provider is configured.

    `init_chat_model` selects the provider (anthropic | google_genai | openai)
    based on settings.llm_provider, so swapping providers is a config change,
    not a code change. The shared `API_KEY` env var is passed explicitly so we
    do not require per-provider env vars (ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    OPENAI_API_KEY) to be set separately.
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    # Per-provider kwargs: different providers spell timeout/max_tokens slightly
    # differently. We pass the shared knobs and the api_key under whichever name
    # that provider's chat-model class accepts.
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "model_provider": provider,
        "temperature": settings.llm_temperature,
    }
    if provider == "anthropic":
        kwargs.update(
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            api_key=settings.api_key,
        )
    elif provider == "google_genai":
        kwargs.update(
            max_output_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            google_api_key=settings.api_key,
        )
    elif provider == "openai":
        kwargs.update(
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            api_key=settings.api_key,
        )
    else:
        kwargs["api_key"] = settings.api_key

    llm = init_chat_model(**kwargs)

    # Model-retry middleware: exponential backoff on transient model errors.
    # Content-moderation middleware is available here too; omitted because
    # inputs come from authenticated operators, not the public web.
    return create_agent(
        model=llm,
        tools=PLANNING_TOOLS,
        system_prompt=_SYSTEM_PROMPT,
        response_format=MissionPlan,
        middleware=[ModelFallbackMiddleware(llm)],
    )


_agent = None


def _agent_lazy() -> Any:
    global _agent
    if _agent is None:
        _agent = _build_planning_agent()
    return _agent


def plan_node(state: CompileState) -> dict[str, Any]:
    """LLM tool-calling node. Produces a draft MissionPlan. On repair passes,
    the previous validation errors are injected so the agent can correct course.
    """
    log.info("→ plan (repair_attempts=%s)", state.get("repair_attempts", 0))
    agent = _agent_lazy()
    repair_attempts = state.get("repair_attempts", 0)
    errors = state.get("validation_errors") or []

    user_msg_parts = [
        f"COMMAND: {state['raw_command']}",
        f"AREA_ID: {state['area_id']}",
        f"OPERATOR_CLEARANCE: {state.get('operator_clearance', 'STANDARD')}",
        f"DRONE_STATE: {state.get('drone_state')}",
        f"GEO_CONTEXT (summary): ceiling={state['geo_context']['altitude_ceiling_m']}m, "
        f"nfz_count={len(state['geo_context']['nfzs'])}",
    ]
    if repair_attempts > 0 and errors:
        user_msg_parts.append(
            "REPAIR PASS — previous attempt had these violations; fix them:\n  - "
            + "\n  - ".join(errors)
        )

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "\n".join(user_msg_parts)}]}
        )
    except Exception as e:  # noqa: BLE001
        log.exception("agent.invoke raised: %s", e)
        return {
            "draft_plan": None,
            "validation_errors": [f"agent error: {type(e).__name__}: {e}"],
        }
    log.info("← plan returned keys=%s", list(result.keys()) if isinstance(result, dict) else type(result).__name__)
    structured: MissionPlan = result.get("structured_response") or result.get("response")
    if structured is None:
        # Defensive: if the agent failed to produce structured output, treat as
        # a violation so the repair loop can try again.
        return {
            "draft_plan": None,
            "validation_errors": ["agent did not produce a structured MissionPlan"],
        }
    if not structured.mission_id:
        structured.mission_id = str(uuid.uuid4())
    structured.area_id = state["area_id"]
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

    try:
        save_mission(get_engine(), plan.model_dump())
    except Exception as e:  # noqa: BLE001
        log.warning("save_mission skipped: %s", e)

    return {"draft_plan": plan.model_dump(), "status": plan.status.value}
