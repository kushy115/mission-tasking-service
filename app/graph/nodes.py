"""Graph nodes for the compile flow.

intake → plan → validate → advisor → finalize | clarify | REJECTED

The `plan` node is the only one that calls the LLM. Every other node is
deterministic. The repair loop is bounded by settings.repair_loop_cap (default
3) — a deliberate safety/cost ceiling; document changes here.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from app.geo.store import get_engine, load_drone_profile, load_geo_context, save_mission
from app.graph.state import CompileState
from app.observability.metrics import (
    LLM_CALLS_TOTAL,
    LLM_DURATION,
    PLAN_BATTERY_PCT,
    PLAN_DURATION,
    PLAN_LEGS,
    VALIDATION_VIOLATIONS,
)
from app.schemas.enums import MissionStatus
from app.schemas.plan import MissionPlan, OptimizationAdvisory
from app.tools.planning_tools import build_planning_tools
from app.validation.deconfliction import deconfliction_check
from app.validation.kernel import validate_plan
from app.weather import get_weather_for_area

log = logging.getLogger(__name__)


class MissionPlanSet(BaseModel):
    """Agent response schema for ALTERNATIVES mode — several candidate plans.

    Used as `response_format` when the operator asked for alternatives, so the
    tool-calling agent returns a typed set instead of a single MissionPlan.
    """

    plans: list[MissionPlan] = Field(default_factory=list)
    primary_idx: int = 0


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

    home_lon, home_lat = geo.home_point
    assigned_home = state.get("assigned_home")
    if assigned_home:
        try:
            home_lon, home_lat = float(assigned_home[0]), float(assigned_home[1])
        except (TypeError, ValueError, IndexError):
            home_lon, home_lat = geo.home_point

    geo_serialized = {
        "area_id": geo.area_id,
        "boundary_lonlat": list(geo.boundary.exterior.coords),
        "nfzs": [list(p.exterior.coords) for p in geo.nfz_polygons],
        "altitude_ceiling_m": geo.altitude_ceiling_m,
        "home_lon": home_lon,
        "home_lat": home_lat,
        "home_bases": [list(base) for base in (geo.home_bases or (geo.home_point,))],
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

    # Fetch weather once per compile (cached for 10 min in the provider layer).
    # Failures fall back to synthetic — never blocks intake.
    weather_obs = get_weather_for_area(geo.home_point[1], geo.home_point[0])
    weather_dict = {
        "wind_mps": weather_obs.wind_mps,
        "gust_mps": weather_obs.gust_mps,
        "wind_dir_deg": weather_obs.wind_dir_deg,
        "visibility_m": weather_obs.visibility_m,
        "precipitation_mmh": weather_obs.precipitation_mmh,
        "temperature_c": weather_obs.temperature_c,
        "source": weather_obs.source,
        "summary": weather_obs.summary_for_prompt(),
    }

    return {
        "geo_context": geo_serialized,
        "weather": weather_dict,
        "repair_attempts": 0,
    }


# ---- plan -------------------------------------------------------------------
#
# IMPORTANT ARCHITECTURE NOTE — read before changing this file.
#
# The `plan` node uses a DIRECT LLM call (one llm.invoke per attempt), NOT a
# LangChain `create_agent` tool-calling agent. This is a deliberate change from
# the original design. Why we moved away from the agent:
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

_SYSTEM_PROMPT = """You are the planner for Perception.

Given a natural-language operator command + the operating area's geofence + the
drone profile, produce ONE mission plan as a single JSON object matching the
schema below. No prose, no markdown — JSON only, starting with `{` and ending
with `}`.

╔═══════════════════════════════════════════════════════════════════════════╗
║ RULE #0 — NO-FLY-ZONE AVOIDANCE IS AUTOMATIC AND ABSOLUTE.                ║
║                                                                           ║
║ The `nfz_polygons` list in GEO_CONTEXT is the set of forbidden regions.   ║
║ The operator NEVER has to tell you to avoid them — you avoid them by      ║
║ default on every leg, every plan, including transits and search patterns. ║
║                                                                           ║
║ Concretely: BEFORE you emit any leg, treat every NFZ as an obstacle to    ║
║ route around. For transits between two points, if the straight line       ║
║ between them clips an NFZ, insert intermediate waypoints that skirt the   ║
║ NFZ with a safety margin of ~15m. For search patterns over an area that   ║
║ contains an NFZ, choose a pattern + orientation that doesn't lay tracks   ║
║ through it (e.g. shift the bounding box, split into two patterns on       ║
║ either side, or shorten tracks where they would otherwise intersect).     ║
║                                                                           ║
║ Crossing an NFZ is a hard kernel failure. The operator should never see   ║
║ a plan with an NFZ-clipping leg — fix it before emitting.                 ║
╚═══════════════════════════════════════════════════════════════════════════╝

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
2. NO leg may cross a no-fly-zone polygon. See RULE #0 above — route around
   them with intermediate waypoints by default; do NOT wait to be told.
3. Every altitude MUST be at or below the ceiling AND at or above 20m AGL.
4. Battery reserve at landing MUST be ≥ 20%.
5. Total duration MUST be ≤ the drone's rated endurance.
6. The LAST leg MUST be a RETURN_TO_BASE back to the home point.

SELECTION RULES — the operator made explicit UI selections; honor them:
- SELECTED_AREA is the operating area. The command is about THIS area by default.
  Generic references ("this area", "the area", "the current area", "here", or
  the selected area's own name) all mean SELECTED_AREA — just plan. ONLY if the
  command names a DIFFERENT, specific place/site that is clearly NOT the selected
  area (e.g. selected area is "west-city" but the command says "patrol the north
  farm"), set status=NEEDS_CLARIFICATION and ask the operator to confirm which
  area — do NOT silently plan in the selected area or invent the other one.
- SELECTED_DRONE is the drone to use. Plan for it. ONLY if the command explicitly
  names a DIFFERENT drone/airframe than the selected profile, set
  status=NEEDS_CLARIFICATION and ask which drone to use. A bare "multiple drones"
  is NOT a different drone — ignore it (fleet sizing is handled outside you).
- SELECTED_SENSORS is the allowed sensor set. Every leg's sensor_mode MUST be one
  of these (or OFF for transits). If empty/none, default to EO. Do NOT use a
  sensor the operator did not enable. If the command's wording conflicts with the
  checkboxes (e.g. command says "use IR" but only EO is checked), prefer the
  CHECKBOXES and note it in reasoning_trace — do NOT clarify over this.

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
- status=NEEDS_CLARIFICATION is a LAST RESORT, only when the command names NO
  target and NO intent so you genuinely cannot pick a region to fly over (e.g.
  "go check something out there", "find stuff", "do a thing"). It is NOT for
  unsafe commands, and NOT for commands that are merely under-detailed. If the
  command names the operating area (or says "this/the current area") and any
  verb of intent (patrol, search, sweep, inspect, cover, survey, monitor), you
  have enough to plan — DO NOT clarify; resolve every remaining detail yourself
  using these defaults:
    * "multiple drones" / "a fleet" / drone-count phrasing → IGNORE it for
      status purposes. Fleet sizing and per-drone tasking are handled OUTSIDE
      you; you plan for the ONE drone in DRONE_STATE (or your assigned slot if a
      MULTI-DRONE GROUP block is present). NEVER ask how many drones there are.
    * "all sensors" / "EO and IR" / unspecified sensor → pick the ONE most
      relevant sensor_mode per the SEARCH_PATTERN rule (IR for heat/night/low-
      visibility, otherwise EO). NEVER ask which sensor to use.
    * "patrol" / "as much as possible" / "cover the area" with no perimeter
      wording → treat as FULL-AREA COVERAGE (SEARCH_PATTERN). NEVER ask whether
      it means perimeter vs. interior — see the LEG TYPE RULES.
    * altitude / detail tradeoff → apply the ALTITUDE default (high for coverage
      unless fine detail is requested). NEVER ask about the altitude band.
  Only emit NEEDS_CLARIFICATION when NONE of the above can rescue the command.
- status=READY_FOR_APPROVAL is for everything else — produce a real plan.

LEG TYPE RULES — picking the wrong leg type is the #1 cause of kernel rejection:
- "patrol" is AMBIGUOUS — disambiguate by what is being patrolled:
    * "patrol the PERIMETER / boundary / fence line / edge" → fly AROUND the
      boundary. Use leg_type=TRANSIT, 4–8 waypoints along the boundary corners.
      The kernel does NOT check sensor coverage for TRANSIT legs.
    * "patrol the ENTIRE / whole / complete area" or "patrol all of <area>" or
      "cover the area" → this is FULL-AREA COVERAGE, NOT a perimeter loop. Use
      leg_type=SEARCH_PATTERN so every square meter is seen. Do NOT just trace
      the boundary — that would miss the interior.
- Use leg_type=TRANSIT for moving between points.
- Use leg_type=SEARCH_PATTERN for grid-coverage sweeps where you need to see
  every square meter of an area (e.g. "search the yard", "sweep the field",
  "patrol the entire area").
  CRITICAL OUTPUT RULE: for a SEARCH_PATTERN leg, emit AT MOST 4 waypoints —
  just the corner bounds of the area to sweep (at your chosen altitude). DO NOT
  enumerate the full serpentine — the service regenerates the exact track
  geometry deterministically (correctly spaced, fitted to the boundary, routed
  around no-fly zones). Emitting hundreds of waypoints overflows the response
  and breaks the plan. Pick pattern_name, altitude, sensor_mode, and the 4
  corner waypoints — nothing more.
  ONE PASS, ONE SENSOR: cover the area with a SINGLE search pass. The gimbal
  carries EO and IR together, so even if the operator says "all sensors" or
  "EO and IR", pick the ONE most relevant sensor_mode for the pass (IR for
  heat/night/low-visibility, otherwise EO). NEVER lay down two separate search
  passes (one per sensor) — that doubles flight time and blows the battery and
  endurance budget for no coverage benefit.
  ALTITUDE: prefer a higher altitude within the ceiling for coverage sweeps —
  higher altitude gives a WIDER sensor swath, so fewer tracks and LESS flight
  time. "high altitude" makes a sweep cheaper, not more expensive. Aim near the
  area ceiling (but at/under it) unless the operator asks for fine detail.
- Use leg_type=LOITER for hovering in place.
- Use leg_type=RETURN_TO_BASE for the final leg only.

OTHER GUIDANCE:
- Always end with a RETURN_TO_BASE leg going to (home_lat, home_lon) at a
  descent altitude (e.g. 30m).
- Pick reasonable durations: a small-yard perimeter patrol is ~2–5 minutes.
- Round coordinates to 6 decimal places.
"""


_LLM_CACHE: dict[str, Any] = {}


def _llm_for(model: str | None = None) -> Any:
    """Build (and cache) a chat model for a specific model id, routed to the
    configured provider. `model=None` uses the configured default model.

    The planning-effort tier (`_effort_config`) selects which model id is passed
    here — a light, rate-limit-friendly model for "fast", the most capable one
    for "thorough". All tiers share the one configured provider + API key.
    """
    settings = get_settings()
    model_id = model or settings.llm_model
    cached = _LLM_CACHE.get(model_id)
    if cached is not None:
        return cached
    provider = settings.llm_provider.lower()
    kwargs: dict[str, Any] = {
        "model": model_id,
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
    llm = init_chat_model(**kwargs)
    _LLM_CACHE[model_id] = llm
    return llm


def _llm_lazy() -> Any:
    """The configured default model (used by the advisor + chat endpoints)."""
    return _llm_for(None)


def _effort_config(requested: str | None) -> tuple[str, dict[str, Any]]:
    """Resolve a planning-effort tier to (name, {model, use_tools, recursion_limit}).

    The tier changes ONLY which model is used and whether the plan node runs the
    tool-calling agent. Every downstream calculation (physics, densification, the
    deterministic safety kernel) is identical across tiers. "fast" = one direct
    LLM call (lowest API volume / rate-limit friendly); "balanced" / "thorough" =
    the grounding agent on progressively more capable models.
    """
    s = get_settings()
    eff = (requested or s.default_planning_effort or "balanced").strip().lower()
    table = {
        "fast": {"model": s.plan_model_fast or s.llm_model, "use_tools": False, "recursion_limit": 0},
        "balanced": {"model": s.plan_model_balanced or s.llm_model, "use_tools": True, "recursion_limit": 30},
        "thorough": {"model": s.plan_model_thorough or s.llm_model, "use_tools": True, "recursion_limit": 60},
    }
    if eff not in table:
        eff = "balanced"
    return eff, table[eff]


def _strip_to_json(text: str) -> str:
    """Pull the first {...} JSON object out of the model's response, tolerating
    leading prose or ```json fences."""
    s = text.strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[1] if len(parts) >= 2 else text
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text  # let the JSON parser raise a meaningful error
    return s[start : end + 1]


# Appended to the system prompt ONLY in the agent tiers (balanced/thorough) so
# the model uses its tools sparingly. The fast tier makes no tool calls at all.
_TOOL_USE_NOTE = """

TOOL USE — BE EFFICIENT (this directly affects latency and API rate limits). You
have tools to ground your plan (geofence, path clearance, battery, sensor
coverage, search-pattern geometry). Call a tool ONLY when its answer changes a
decision, and at most once or twice in total. Do NOT re-check every leg, and do
NOT re-call a tool you already have an answer for. Make a few targeted calls,
then commit — the deterministic safety kernel validates the final plan
regardless, so you do not need to verify exhaustively yourself."""


def _direct_plan(
    llm: Any,
    history_msgs: list[Any],
    user_msg: str,
    area_id: str,
    provider: str,
    model: str,
    effort: str,
) -> tuple[MissionPlan | None, list[dict[str, Any]], str | None]:
    """One direct LLM call → MissionPlan (no tools). Used by the 'fast' tier AND
    as the graceful fallback when the agent loop stalls (e.g. GraphRecursionError)
    so a drone still gets a plan. Returns (plan, raw_alternatives, error_message).
    """
    start = time.perf_counter()
    try:
        ai_msg = llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), *history_msgs, HumanMessage(content=user_msg)]
        )
    except Exception as e:  # noqa: BLE001
        log.exception("planning LLM raised: %s", e)
        outcome = "rate_limited" if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) else "error"
        LLM_CALLS_TOTAL.labels(provider, model, outcome).inc()
        LLM_DURATION.labels(provider, model).observe(time.perf_counter() - start)
        return None, [], f"llm error: {type(e).__name__}: {e}"
    LLM_CALLS_TOTAL.labels(provider, model, "ok").inc()
    LLM_DURATION.labels(provider, model).observe(time.perf_counter() - start)
    raw_text = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
    log.info(
        "← plan direct [%s/%s]: %d chars",
        effort, model, len(raw_text) if isinstance(raw_text, str) else -1,
    )
    try:
        payload = json.loads(_strip_to_json(raw_text))
    except Exception as e:  # noqa: BLE001
        log.warning("plan JSON parse failed: %s; first 200 chars: %r", e, raw_text[:200])
        return None, [], f"plan was not valid JSON: {e}"
    raw_alts: list[dict[str, Any]] = []
    if isinstance(payload, dict) and "plans" in payload and isinstance(payload["plans"], list):
        plans = payload["plans"]
        if not plans:
            return None, [], "alternatives list was empty"
        primary_idx = max(0, min(int(payload.get("primary_idx") or 0), len(plans) - 1))
        raw_alts = [p for i, p in enumerate(plans) if i != primary_idx]
        payload = plans[primary_idx]
    payload.setdefault("mission_id", str(uuid.uuid4()))
    payload["area_id"] = area_id
    try:
        structured = MissionPlan.model_validate(payload)
    except Exception as e:  # noqa: BLE001
        log.warning("MissionPlan validation failed: %s", e)
        return None, [], f"plan failed schema: {e}"
    return structured, raw_alts, None


def plan_node(state: CompileState) -> dict[str, Any]:
    """Produce a draft MissionPlan via the planning-effort tier (see _effort_config).

    fast = one direct LLM call; balanced/thorough = a tool-calling agent. On
    repair passes the kernel's violation list is injected so the model can
    correct course. The loop cap comes from the graph builder.
    """
    log.info("→ plan (repair_attempts=%s)", state.get("repair_attempts", 0))
    repair_attempts = state.get("repair_attempts", 0)
    errors = state.get("validation_errors") or []
    geo = state["geo_context"]

    selected_sensors = [s.upper() for s in (state.get("selected_sensors") or [])]
    selected_drone = (state.get("drone_state") or {}).get("drone_profile_id", "")
    sensor_line = (
        f"SELECTED_SENSORS (operator-enabled, use ONLY these): {selected_sensors}"
        if selected_sensors
        else "SELECTED_SENSORS: none checked — default to EO (visual)"
    )
    user_msg_parts = [
        f"COMMAND: {state['raw_command']}",
        "",
        "OPERATOR UI SELECTIONS (authoritative context — see SELECTION RULES):",
        f"  SELECTED_AREA (the operating area to plan in): {state['area_id']}",
        f"  SELECTED_DRONE (the drone profile to use): {selected_drone}",
        f"  {sensor_line}",
        "",
        f"OPERATOR_CLEARANCE: {state.get('operator_clearance', 'STANDARD')}",
        f"DRONE_STATE: {state.get('drone_state')}",
        "GEO_CONTEXT:",
        f"  boundary_lonlat (polygon vertices): {geo['boundary_lonlat']}",
        f"  nfz_polygons:                       {geo['nfzs']}",
        f"  altitude_ceiling_m:                 {geo['altitude_ceiling_m']}",
        f"  home_lon, home_lat:                 {geo['home_lon']}, {geo['home_lat']}",
        f"  registered_home_bases:              {geo.get('home_bases') or []}",
    ]
    wx = state.get("weather")
    if wx:
        user_msg_parts.append(f"WEATHER: {wx.get('summary', '')}")
        user_msg_parts.append(
            "  Consider weather when choosing sensor (low visibility favors IR), "
            "orienting search patterns (cross-wind sweeps lose more battery), "
            "and selecting altitudes (higher altitudes feel stronger winds)."
        )
    # Multi-drone group context: planner is told which slot/layer it occupies.
    md = state.get("multi_drone_slot")
    if md:
        spatial = ""
        total = md.get("total", 1)
        idx = md.get("index", 0)
        if total > 1:
            lats = [pt[1] for pt in geo["boundary_lonlat"]]
            miny, maxy = min(lats), max(lats)
            band_h = (maxy - miny) / total
            lo = miny + idx * band_h
            hi = miny + (idx + 1) * band_h
            spatial = (
                f" SPATIAL ASSIGNMENT: cover ONLY the latitude band "
                f"{lo:.6f}–{hi:.6f} (slice {idx + 1} of {total}, dividing the area "
                f"north-to-south). Your search pattern is auto-fitted to this "
                f"band — transit from home to your band, sweep it, then return. "
                f"Do NOT plan to cover the whole area; siblings cover the rest."
            )
        user_msg_parts.append(
            f"\nMULTI-DRONE GROUP — you are planning drone {md['index'] + 1} of "
            f"{md['total']}. Your assigned altitude band is "
            f"{md['alt_min']:.0f}–{md['alt_max']:.0f}m. Siblings are on bands: "
            f"{md['sibling_bands']}. Stagger your first leg's est_duration_s by "
            f"{md['takeoff_offset_s']:.0f}s so the group doesn't collide at home."
            f"{spatial}"
        )
    if state.get("alternatives_requested"):
        user_msg_parts.append(
            "\nALTERNATIVES MODE — produce 2 or 3 alternative plans (e.g. a "
            "conservative-battery one, an aggressive-coverage one, a different "
            "search orientation). Return ONE JSON object with shape "
            '{"plans": [<plan>, <plan>, ...], "primary_idx": <int>}. Each '
            "<plan> matches the MissionPlan schema. Pick primary_idx for the "
            "one you most recommend."
        )
    if repair_attempts > 0 and errors:
        user_msg_parts.append(
            "\nREPAIR PASS — the deterministic safety kernel rejected your previous "
            "plan with these violations. Fix every one of them:\n  - " + "\n  - ".join(errors)
        )
    user_msg = "\n".join(user_msg_parts)

    # Conversation history (DD-005): when the operator answers a clarification
    # question from a prior turn, we feed the prior turns to the LLM as proper
    # LangChain message objects so the planner has full context, not just the
    # latest follow-up. Empty for fresh missions.
    convo = state.get("conversation_history") or []
    history_msgs: list[Any] = []
    for turn in convo:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            history_msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            history_msgs.append(AIMessage(content=content))

    provider = get_settings().llm_provider
    effort, cfg = _effort_config(state.get("planning_effort"))
    model = cfg["model"]
    llm = _llm_for(model)
    want_alts = bool(state.get("alternatives_requested"))
    raw_alternatives: list[dict[str, Any]] = []
    structured: MissionPlan | None = None

    if cfg["use_tools"]:
        # ---- Balanced / Thorough: real tool-calling agent (DD-016) ----------
        # The model GROUNDS its plan with the planning tools (geofence, path
        # clearance, battery, sensor coverage, pattern geometry) and self-corrects
        # inside its own loop, then returns a schema-valid plan via ToolStrategy
        # (a NON-strict tool call — Anthropic's strict native structured output
        # overflows its compiled-grammar limit on the large MissionPlan schema).
        try:
            profile = load_drone_profile(get_engine(), selected_drone or "long-endurance-quad")
        except Exception:  # noqa: BLE001 — profile renamed/removed; fall back
            profile = load_drone_profile(get_engine(), "long-endurance-quad")
        tools = build_planning_tools(geo, profile)
        response_schema: type = MissionPlanSet if want_alts else MissionPlan
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=_SYSTEM_PROMPT + _TOOL_USE_NOTE,
            response_format=ToolStrategy(response_schema),
        )
        start = time.perf_counter()
        result: Any = None
        try:
            result = agent.invoke(
                {"messages": [*history_msgs, HumanMessage(content=user_msg)]},
                config={"recursion_limit": cfg["recursion_limit"]},
            )
        except Exception as e:  # noqa: BLE001
            # Most common: GraphRecursionError — the agent looped on tool calls
            # without converging. Don't fail the drone (that produced an empty,
            # 0-battery "rejected" plan with no clear reason); fall through to a
            # single direct call below so this slot still gets a real plan.
            log.warning("planning agent failed (%s); will fall back to direct call", type(e).__name__)
        LLM_DURATION.labels(provider, model).observe(time.perf_counter() - start)
        if result is not None:
            agent_msgs = result.get("messages", []) if isinstance(result, dict) else []
            n_ai_turns = sum(1 for m in agent_msgs if isinstance(m, AIMessage))
            n_tool_calls = sum(1 for m in agent_msgs if isinstance(m, ToolMessage))
            for _ in range(max(n_ai_turns, 1)):
                LLM_CALLS_TOTAL.labels(provider, model, "ok").inc()
            log.info(
                "← plan agent [%s/%s]: %d model turns, %d tool calls",
                effort, model, n_ai_turns, n_tool_calls,
            )
            structured_obj = result.get("structured_response") if isinstance(result, dict) else None
            if want_alts and isinstance(structured_obj, MissionPlanSet) and structured_obj.plans:
                primary_idx = max(0, min(structured_obj.primary_idx, len(structured_obj.plans) - 1))
                structured = structured_obj.plans[primary_idx]
                raw_alternatives = [
                    p.model_dump() for i, p in enumerate(structured_obj.plans) if i != primary_idx
                ]
            elif isinstance(structured_obj, MissionPlan):
                structured = structured_obj

        # Graceful fallback: agent errored or returned no usable plan → one
        # direct LLM call so the drone still produces a plan instead of failing.
        if structured is None:
            log.info("agent yielded no plan for [%s]; using direct fallback", model)
            structured, raw_alternatives, err = _direct_plan(
                llm, history_msgs, user_msg, state["area_id"], provider, model, effort
            )
            if structured is None:
                return {"draft_plan": None, "validation_errors": [err or "planner produced no plan"]}
    else:
        # ---- Fast: ONE direct LLM call → JSON (no tools, no agent loop) ------
        # The original single-shot planner: lowest API volume, rate-limit
        # friendly. Identical downstream handling — only the call count differs.
        structured, raw_alternatives, err = _direct_plan(
            llm, history_msgs, user_msg, state["area_id"], provider, model, effort
        )
        if structured is None:
            return {"draft_plan": None, "validation_errors": [err or "planner produced no plan"]}

    if not structured.mission_id:
        structured.mission_id = str(uuid.uuid4())
    structured.area_id = state["area_id"]

    # ----- Deterministic sensor clamp: enforce the operator's checkbox selection.
    #
    # The prompt already constrains the planner to SELECTED_SENSORS, but we
    # mechanically guarantee it: any active (non-OFF) leg whose sensor_mode is
    # not in the allowed set is remapped to the first allowed sensor. Empty
    # selection defaults to EO. Runs before swath/densification so coverage math
    # uses the sensor we'll actually fly.
    if structured.status == MissionStatus.READY_FOR_APPROVAL:
        from app.schemas.enums import SensorMode as _SensorMode  # noqa: PLC0415

        allowed = [s for s in selected_sensors if s in ("EO", "IR")] or ["EO"]
        allowed_default = _SensorMode(allowed[0])
        for leg in structured.legs:
            if leg.sensor_mode is None or leg.sensor_mode == _SensorMode.OFF:
                continue
            if leg.sensor_mode.value not in allowed:
                leg.sensor_mode = allowed_default

    # ----- DD-010: deterministic geometry for SEARCH_PATTERN legs.
    #
    # The LLM is unreliable at producing correctly-spaced search patterns: it
    # writes confident reasoning ("16 parallel tracks at 60m spacing") but
    # emits only the 4 corner waypoints, leaving the kernel's coverage check
    # to reject the plan ("track spacing 622.7m exceeds sensor swath 75m").
    # Fix: keep the LLM's choices (pattern_name, altitude, sensor_mode) but
    # REPLACE the geometry with a deterministically-generated lawnmower fitted
    # to the area boundary at the appropriate spacing.
    try:
        from shapely.geometry import Polygon as _SPoly

        from app.geo.patterns import band_subregion, lawnmower_fit_to_boundary
        from app.schemas.enums import LegType
        from app.validation.physics import estimate_leg, sensor_swath_m

        boundary_coords = geo["boundary_lonlat"]
        boundary_poly = _SPoly(boundary_coords)
        nfz_polys_search = [_SPoly(c) for c in (geo.get("nfzs") or []) if c]
        drone_id = state["drone_state"]["drone_profile_id"]
        profile = load_drone_profile(get_engine(), drone_id)

        # Multi-drone spatial decomposition: each drone sweeps its own latitude
        # band so the group divides the area instead of every drone covering the
        # whole thing. Single-drone missions get sub_region=None (full area).
        md_slot = state.get("multi_drone_slot")
        sub_region = None
        if md_slot and md_slot.get("total", 1) > 1:
            sub_region = band_subregion(
                boundary_poly, md_slot.get("index", 0), md_slot["total"]
            )

        def _pick_sensor(mode_str: str | None) -> Any:
            target = (mode_str or "EO").upper()
            for s in profile.sensors:
                if s.mode.value == target:
                    return s
            return profile.sensors[0] if profile.sensors else None

        # ONE PASS: a real gimbal carries EO+IR together, so a single sweep
        # covers the area. If the LLM emitted multiple SEARCH_PATTERN legs (e.g.
        # one per sensor), that DOUBLES flight time and busts endurance. Collapse
        # to the first search leg and drop the rest. See agent-diagnosed root
        # cause — this was the #1 source of per-drone endurance rejections.
        search_idxs = [i for i, lg in enumerate(structured.legs) if lg.leg_type == LegType.SEARCH_PATTERN]
        if len(search_idxs) > 1:
            keep = search_idxs[0]
            structured.legs = [
                lg
                for i, lg in enumerate(structured.legs)
                if lg.leg_type != LegType.SEARCH_PATTERN or i == keep
            ]
            log.info("collapsed %d SEARCH_PATTERN legs to 1 (single-pass coverage)", len(search_idxs))

        # Deterministic altitude for coverage sweeps: fly as HIGH as legally
        # allowed (widest sensor swath → fewest tracks → least duration/battery).
        # This makes the per-drone budget a function of band size, not the LLM's
        # altitude guess — which is what made same-profile drones diverge (one
        # guessed low and busted endurance, a sibling guessed high and passed).
        # Clamp to the drone's assigned altitude band in multi-drone mode so
        # vertical deconfliction still holds.
        ceiling = float(geo.get("altitude_ceiling_m") or 120.0)
        coverage_alt = max(20.0, ceiling - 5.0)
        if md_slot and md_slot.get("total", 1) > 1:
            band_hi = float(md_slot.get("alt_max", coverage_alt))
            coverage_alt = max(20.0, min(coverage_alt, band_hi))
        # Respect an explicit low-altitude / fine-detail request — don't force
        # the swath-maximizing altitude when the operator wants close inspection.
        _cmd_lc = (state.get("raw_command") or "").lower()
        wants_detail = any(
            k in _cmd_lc
            for k in ("low altitude", "low alt", "close", "detail", "high resolution", "high-res", "fine")
        )

        for leg_idx, leg in enumerate(structured.legs):
            if leg.leg_type != LegType.SEARCH_PATTERN:
                continue
            sensor = _pick_sensor(leg.sensor_mode.value if leg.sensor_mode else None)
            if sensor is None:
                continue
            llm_alt = leg.geometry[0].alt_m if leg.geometry else coverage_alt
            alt = llm_alt if wants_detail else coverage_alt
            swath = sensor_swath_m(sensor, alt)
            new_pts = lawnmower_fit_to_boundary(
                boundary_poly,
                alt,
                swath,
                nfz_polygons=nfz_polys_search,
                sub_region=sub_region,
            )
            if not new_pts:
                continue
            leg.geometry = new_pts
            # Recompute leg duration/battery from the new geometry so the kernel's
            # endurance/battery checks line up with what we'll actually fly.
            energy = estimate_leg(leg, profile)
            leg.est_duration_s = energy.duration_s
            leg.est_battery_pct = energy.battery_pct
            log.info(
                "densified SEARCH_PATTERN leg[%d]: %d waypoints @ %.1fm spacing "
                "(swath=%.1fm, alt=%.0fm)",
                leg_idx,
                len(new_pts),
                swath * 0.85,
                swath,
                alt,
            )

        # Re-aggregate total_duration / total_battery if we changed anything.
        if any(leg.leg_type == LegType.SEARCH_PATTERN for leg in structured.legs):
            structured.total_duration_s = sum(leg.est_duration_s for leg in structured.legs)
            structured.total_battery_pct = sum(leg.est_battery_pct for leg in structured.legs)
            starting_pct = float(state.get("drone_state", {}).get("battery_pct", 100.0))
            structured.battery_reserve_pct = starting_pct - structured.total_battery_pct
    except Exception as e:  # noqa: BLE001 — densification is best-effort
        log.warning("search-pattern densification skipped: %s", e)

    # ----- Deterministic perimeter-patrol geometry.
    #
    # Same class of bug as SEARCH_PATTERN: the LLM hand-picks 4–8 boundary
    # corner waypoints for "patrol the perimeter" commands and routinely clips
    # NFZs or drifts outside the boundary, especially with multiple NFZs. Fix:
    # detect patrol intent from the command, find the TRANSIT leg(s) that look
    # like the perimeter trace, and REPLACE their geometry with a Shapely-
    # computed inset perimeter that has been routed around buffered NFZs.
    try:
        from shapely.geometry import Polygon as _SPoly  # noqa: PLC0415

        from app.geo.patterns import perimeter_patrol_fit_to_boundary  # noqa: PLC0415
        from app.schemas.enums import LegType  # noqa: PLC0415
        from app.validation.physics import estimate_leg  # noqa: PLC0415

        cmd_lc = (state.get("raw_command") or "").lower()
        # Distinguish a PERIMETER patrol (fly the boundary) from an AREA patrol
        # / full-coverage sweep ("patrol the ENTIRE area"). Only the former gets
        # the perimeter-trace replacement; the latter is a SEARCH_PATTERN and is
        # handled by the lawnmower densifier above. "patrol the entire west-city"
        # means cover everything, NOT trace the fence.
        coverage_markers = ("entire", "whole", "all of", "every", "complete coverage", "cover the")
        perimeter_markers = ("perimeter", "boundary", "around the", "fence line", "edge of")
        wants_coverage = any(m in cmd_lc for m in coverage_markers)
        wants_perimeter = any(m in cmd_lc for m in perimeter_markers)
        # Fire perimeter replacement only when it's clearly a boundary patrol:
        # explicit perimeter language, or a bare "patrol" with no coverage intent.
        is_patrol = wants_perimeter or ("patrol" in cmd_lc and not wants_coverage)
        if is_patrol:
            boundary_coords = geo["boundary_lonlat"]
            boundary_poly = _SPoly(boundary_coords)
            nfz_polys = [_SPoly(c) for c in (geo.get("nfzs") or []) if c]
            drone_id = state["drone_state"]["drone_profile_id"]
            profile = load_drone_profile(get_engine(), drone_id)
            patrol_replaced = False
            for leg in structured.legs:
                # Only replace the long perimeter-style TRANSIT (>= 3 waypoints,
                # not the takeoff hop or RTB). Heuristic: the longest TRANSIT
                # that isn't the final RTB.
                if leg.leg_type != LegType.TRANSIT or len(leg.geometry) < 3:
                    continue
                alt = leg.geometry[0].alt_m if leg.geometry else 60.0
                new_pts = perimeter_patrol_fit_to_boundary(
                    boundary_poly, nfz_polys, altitude_m=alt
                )
                if len(new_pts) < 3:
                    continue
                leg.geometry = new_pts
                energy = estimate_leg(leg, profile)
                leg.est_duration_s = energy.duration_s
                leg.est_battery_pct = energy.battery_pct
                patrol_replaced = True
                log.info(
                    "replaced TRANSIT-perimeter geometry: %d waypoints (NFZ-safe)",
                    len(new_pts),
                )
                break  # only replace the first/longest patrol TRANSIT

            if patrol_replaced:
                structured.total_duration_s = sum(
                    leg.est_duration_s for leg in structured.legs
                )
                structured.total_battery_pct = sum(
                    leg.est_battery_pct for leg in structured.legs
                )
                starting_pct = float(state.get("drone_state", {}).get("battery_pct", 100.0))
                structured.battery_reserve_pct = starting_pct - structured.total_battery_pct
    except Exception as e:  # noqa: BLE001 — replacement is best-effort
        log.warning("perimeter-patrol densification skipped: %s", e)

    # ----- Explicit home → work → home routing.
    #
    # Every mission must physically depart from a base and recover to a base —
    # even when redundant — so the flown route (and the live sim) shows the
    # transit out and back, not a drone teleporting onto its search band. We
    # rebuild the bookend legs deterministically rather than trusting the LLM:
    #   * strip any LLM-authored leading TRANSIT and trailing RETURN_TO_BASE,
    #   * keep the "core" work legs (search / loiter / sensor / mid-transits),
    #   * prepend a TRANSIT from the nearest base to the first work waypoint,
    #   * append a RETURN_TO_BASE to the nearest base to the LAST work waypoint.
    # Takeoff and recovery bases are chosen INDEPENDENTLY — a drone may land at
    # a different base than it launched from (whichever is closest to where it
    # finishes). NFZ routing + aggregate recompute below fix up the new legs.
    if structured.status == MissionStatus.READY_FOR_APPROVAL and structured.legs:
        try:
            from app.graph.multi_drone import nearest_home_base  # noqa: PLC0415
            from app.schemas.enums import LegType as _HLeg  # noqa: PLC0415
            from app.schemas.enums import SensorMode as _HSensor  # noqa: PLC0415
            from app.schemas.plan import MissionLeg as _MLeg  # noqa: PLC0415
            from app.schemas.plan import Waypoint as _Wp  # noqa: PLC0415

            bases = [list(b) for b in (geo.get("home_bases") or [])]
            if not bases:
                bases = [[float(geo["home_lon"]), float(geo["home_lat"])]]

            core = list(structured.legs)
            if core and core[0].leg_type == _HLeg.TRANSIT:
                core = core[1:]
            if core and core[-1].leg_type == _HLeg.RETURN_TO_BASE:
                core = core[:-1]

            if core and core[0].geometry and core[-1].geometry:
                first_wp = core[0].geometry[0]
                last_wp = core[-1].geometry[-1]
                takeoff = nearest_home_base(bases, (first_wp.lon, first_wp.lat))
                landing = nearest_home_base(bases, (last_wp.lon, last_wp.lat))
                base_alt = max(30.0, get_settings().min_agl_m + 5.0)

                takeoff_leg = _MLeg(
                    leg_type=_HLeg.TRANSIT,
                    geometry=[
                        _Wp(lat=takeoff[1], lon=takeoff[0], alt_m=base_alt),
                        _Wp(lat=first_wp.lat, lon=first_wp.lon, alt_m=first_wp.alt_m),
                    ],
                    sensor_mode=_HSensor.OFF,
                    est_duration_s=0.0,
                    est_battery_pct=0.0,
                )
                rtb_leg = _MLeg(
                    leg_type=_HLeg.RETURN_TO_BASE,
                    geometry=[
                        _Wp(lat=last_wp.lat, lon=last_wp.lon, alt_m=last_wp.alt_m),
                        _Wp(lat=landing[1], lon=landing[0], alt_m=base_alt),
                    ],
                    sensor_mode=_HSensor.OFF,
                    est_duration_s=0.0,
                    est_battery_pct=0.0,
                )
                structured.legs = [takeoff_leg, *core, rtb_leg]
                log.info(
                    "home routing: takeoff %.5f,%.5f → land %.5f,%.5f (%s base)",
                    takeoff[0], takeoff[1], landing[0], landing[1],
                    "same" if takeoff == landing else "different",
                )
        except Exception as e:  # noqa: BLE001 — best-effort; LLM legs still stand
            log.warning("home routing rebuild skipped: %s", e)

    # ----- Snap stray waypoints back inside the geofence.
    #
    # The LLM hand-picks TRANSIT / RETURN_TO_BASE / LOITER geometry and
    # occasionally places a waypoint a few meters outside the boundary, which
    # the kernel rejects ("waypoint outside geofence"). SEARCH_PATTERN legs are
    # already deterministically generated inside, so we leave them alone. Snap
    # any out-of-bounds waypoint to the nearest point a hair INSIDE the boundary.
    try:
        from shapely.geometry import Point as _SPoint  # noqa: PLC0415
        from shapely.geometry import Polygon as _SPoly2  # noqa: PLC0415

        from app.schemas.enums import LegType as _LegType  # noqa: PLC0415

        bpoly = _SPoly2(geo["boundary_lonlat"])
        inner = bpoly.buffer(-2e-5)  # ~2m inside, in degrees
        if inner.is_empty:
            inner = bpoly
        snapped = 0
        for leg in structured.legs:
            if leg.leg_type == _LegType.SEARCH_PATTERN:
                continue
            for wp in leg.geometry:
                pt = _SPoint(wp.lon, wp.lat)
                if not bpoly.covers(pt):
                    nearest = inner.exterior.interpolate(inner.exterior.project(pt))
                    wp.lon, wp.lat = float(nearest.x), float(nearest.y)
                    snapped += 1
        if snapped:
            log.info("snapped %d out-of-bounds waypoint(s) back inside geofence", snapped)

        # Route non-search legs around NFZs deterministically. The LLM authors
        # TRANSIT / RETURN_TO_BASE geometry (home → band → home) and its straight
        # lines routinely clip a no-fly zone. Reuse the same visibility-graph
        # router the lawnmower uses to insert NFZ-avoiding detour waypoints.
        nfz_polys_route = [_SPoly2(c) for c in (geo.get("nfzs") or []) if c]
        if nfz_polys_route:
            from app.geo.patterns import (  # noqa: PLC0415
                METERS_PER_DEG_LAT,
                _route_through_flyable,
            )

            lats_all = [pt[1] for pt in geo["boundary_lonlat"]]
            mlat = sum(lats_all) / len(lats_all)
            import math as _m  # noqa: PLC0415

            margin_deg = 15.0 * (
                1.0 / METERS_PER_DEG_LAT
                + 1.0 / max(METERS_PER_DEG_LAT * _m.cos(_m.radians(mlat)), 1e-6)
            ) / 2.0
            nfz_bufs_route = [p.buffer(margin_deg) for p in nfz_polys_route]
            routed = 0
            for leg in structured.legs:
                if leg.leg_type == _LegType.SEARCH_PATTERN or len(leg.geometry) < 2:
                    continue
                new_geo = [leg.geometry[0]]
                for prev, curr in zip(leg.geometry, leg.geometry[1:], strict=False):
                    detour = _route_through_flyable(
                        (prev.lon, prev.lat), (curr.lon, curr.lat), nfz_bufs_route
                    )
                    for lon, latd in detour:
                        new_geo.append(type(curr)(lat=latd, lon=lon, alt_m=curr.alt_m))
                        routed += 1
                    new_geo.append(curr)
                leg.geometry = new_geo
            if routed:
                log.info("inserted %d NFZ-avoidance waypoint(s) on non-search legs", routed)

        # Recompute aggregates since snapping/routing changed geometry.
        try:
            from app.validation.physics import estimate_leg as _est  # noqa: PLC0415

            _drone_id = state["drone_state"]["drone_profile_id"]
            _profile = load_drone_profile(get_engine(), _drone_id)
            for leg in structured.legs:
                e = _est(leg, _profile)
                leg.est_duration_s = e.duration_s
                leg.est_battery_pct = e.battery_pct
            structured.total_duration_s = sum(leg.est_duration_s for leg in structured.legs)
            structured.total_battery_pct = sum(leg.est_battery_pct for leg in structured.legs)
            _start_pct = float(state.get("drone_state", {}).get("battery_pct", 100.0))
            structured.battery_reserve_pct = _start_pct - structured.total_battery_pct
        except Exception as _e:  # noqa: BLE001
            log.warning("aggregate recompute after snap/route skipped: %s", _e)
    except Exception as e:  # noqa: BLE001 — snapping is best-effort
        log.warning("waypoint snap skipped: %s", e)

    out: dict[str, Any] = {"draft_plan": structured.model_dump()}
    if raw_alternatives:
        # Stash raw alternative dicts; validate_node will filter and convert.
        out["alternatives"] = raw_alternatives
    return out


# ---- validate ---------------------------------------------------------------


def validate_node(state: CompileState) -> dict[str, Any]:
    """Deterministic kernel pass. No LLM. Source of truth on safety."""
    log.info("→ validate")
    raw = state.get("draft_plan")
    if raw is None:
        return {
            "validation_errors": state.get("validation_errors") or ["no draft_plan to validate"]
        }

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

    # Re-hydrate the weather observation captured in intake (so cost doesn't
    # double-bill per repair attempt) and feed it through the kernel.
    weather_obs = None
    wx = state.get("weather")
    if wx:
        from app.weather import WeatherObservation

        weather_obs = WeatherObservation(
            wind_mps=wx["wind_mps"],
            gust_mps=wx["gust_mps"],
            wind_dir_deg=wx["wind_dir_deg"],
            visibility_m=wx["visibility_m"],
            precipitation_mmh=wx["precipitation_mmh"],
            temperature_c=wx["temperature_c"],
            source=wx.get("source", "cached"),
        )

    # Airspace deconfliction: compare against approved missions in the same
    # area within the active window. See DESIGN_DECISIONS §8. The state can
    # carry a precomputed value for the multi-drone group case where siblings
    # haven't been persisted yet.
    deconfliction = state.get("_deconfliction")
    if deconfliction is None:
        try:
            deconfliction = deconfliction_check(engine, plan, self_mission_id=plan.mission_id)
        except Exception as e:  # noqa: BLE001
            log.warning("deconfliction check failed: %s", e)
            deconfliction = (True, [])

    violations, report = validate_plan(
        plan,
        geo,
        profile,
        starting_battery_pct=starting_pct,
        weather=weather_obs,
        deconfliction=deconfliction,
    )

    # Filter alternatives independently: each must pass the kernel; violators
    # are dropped silently (operator never sees an unsafe "choice"). Repair
    # loop applies only to the primary plan. See DESIGN_DECISIONS §7.
    raw_alts = state.get("alternatives") or []
    accepted_alts: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_alts):
        try:
            raw.setdefault("mission_id", f"{plan.mission_id}-alt{i + 1}")
            raw["area_id"] = state["area_id"]
            alt_plan = MissionPlan.model_validate(raw)
        except Exception as e:  # noqa: BLE001
            log.info("dropping alternative %d: schema failure (%s)", i, e)
            continue
        alt_vios, _alt_report = validate_plan(
            alt_plan,
            geo,
            profile,
            starting_battery_pct=starting_pct,
            weather=weather_obs,
            deconfliction=deconfliction,
        )
        if alt_vios:
            log.info("dropping alternative %d: %d violations", i, len(alt_vios))
            continue
        accepted_alts.append(alt_plan.model_dump())

    # Capture this attempt for the mission-history diff view.
    drafts = list(state.get("repair_drafts") or [])
    drafts.append(
        {
            "attempt": state.get("repair_attempts", 0),
            "draft_plan": plan.model_dump(),
            "violations": list(violations),
        }
    )

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
        "weather_acceptable",
        "airspace_deconflicted",
    ):
        if not getattr(report, check_name):
            VALIDATION_VIOLATIONS.labels(check_name).inc()

    if violations:
        log.info("validate: %d violation(s): %s", len(violations), violations[:6])
    return {
        "draft_plan": plan.model_dump(),
        "validation_errors": violations,
        "constraints_report": report.model_dump(),
        "repair_drafts": drafts,
        "alternatives": accepted_alts,
    }


# ---- advisor (optimization suggestions) ------------------------------------
#
# Uses LangChain's `with_structured_output` so the LLM is forced to return an
# OptimizationAdvisory pydantic object — no free-text JSON parsing, no string
# surgery. The advisor sees the validated plan, weather, drone profile, fleet,
# and original command. It proposes concrete tweaks the operator could choose to
# adopt before approving. Failure is non-blocking — the plan still ships.

_ADVISOR_SYSTEM_PROMPT = """You are a senior drone-operations advisor reviewing
a mission plan that has ALREADY PASSED the deterministic safety kernel. Your
job is NOT safety — that is settled. Your job is to suggest concrete tactical
optimizations the operator should consider before approving the plan.

You will receive:
  - the operator's original natural-language command
  - the validated mission plan (legs, sensor choices, altitudes, durations)
  - current weather conditions at the operating area
  - the drone profile being used
  - other drone profiles available in the fleet
  - estimated battery reserve at landing

Produce exactly 2 specific, actionable suggestions. Each title and rationale
must be one crisp line. Each must be a slight modification the operator could
adopt while STILL ACCOMPLISHING the mission goal. Examples
of good suggestions:
  - "Rotate the lawnmower sweep 90° to run parallel to the 8 m/s northwest wind
     — saves ~6% battery vs. cross-wind."
  - "Swap the EO sensor for IR on leg 2 — visibility is 1800 m due to mist,
     and IR sees through it for the same coverage."
  - "Split the 200-acre sweep across two drones at staggered altitudes (40 m
     and 70 m) — cuts total wall-clock time roughly in half and leaves more
     battery margin per airframe."
  - "Drop the loiter from 4 min to 2 min — the EO will already have captured
     the asset; the extra time bleeds battery you might want as reserve."

If the plan looks RESOURCE-CONSTRAINED (battery reserve under 30%, duration
within 5 minutes of the rated endurance, or trying to cover too much area for
one drone), include a `resource_constrained_fallback`: one sentence describing
a scoped-down mission that still accomplishes the PRIMARY goal but fits
comfortably within resources. Otherwise leave it null.

Each suggestion needs:
  - title:      one-line headline
  - rationale:  one concise sentence explaining why this is better given conditions
  - impact:     "minor" | "moderate" | "major"
  - category:   "weather" | "resource" | "coverage" | "coordination" | "tactical"

Do NOT propose changes that would violate the safety kernel (geofence breach,
crossing NFZs, etc.). Do NOT assess safety — assume the plan is safe.
"""


def advisor_node(state: CompileState) -> dict[str, Any]:
    """LangChain structured-output advisor. Always advisory — never blocks."""
    log.info("→ advisor")
    raw = state.get("draft_plan") or {}
    plan_status = raw.get("status")
    if plan_status in (MissionStatus.NEEDS_CLARIFICATION.value, MissionStatus.REJECTED.value):
        return {}
    if not raw.get("legs"):
        return {}

    settings = get_settings()
    if not settings.api_key:
        log.info("advisor skipped: no API key configured")
        return {}

    # Load fleet + drone info so the advisor can recommend multi-drone splits
    # or different airframes when appropriate.
    try:
        from app.geo.store import list_drones

        fleet = [
            {
                "profile_id": d.get("profile_id"),
                "rated_endurance_min": round((d.get("rated_endurance_s") or 0) / 60.0, 1),
                "cruise_speed_mps": d.get("cruise_speed_mps"),
            }
            for d in (list_drones(get_engine()) or [])
        ]
    except Exception as e:  # noqa: BLE001
        log.warning("advisor: fleet lookup failed: %s", e)
        fleet = []

    drone_id = (state.get("drone_state") or {}).get("drone_profile_id")
    drone_profile_info = {}
    if drone_id:
        try:
            profile = load_drone_profile(get_engine(), drone_id)
            drone_profile_info = {
                "profile_id": drone_id,
                "rated_endurance_min": round(profile.rated_endurance_s / 60.0, 1),
                "cruise_speed_mps": profile.cruise_speed_mps,
                "sensors": [s.mode.value for s in profile.sensors],
            }
        except Exception as e:  # noqa: BLE001
            log.warning("advisor: drone lookup failed: %s", e)

    wx = state.get("weather") or {}
    plan_summary = {
        "legs": [
            {
                "leg_type": leg.get("leg_type"),
                "pattern": leg.get("pattern_name"),
                "sensor": leg.get("sensor_mode"),
                "alt_m": (leg.get("geometry") or [{}])[0].get("alt_m"),
                "duration_s": leg.get("est_duration_s"),
                "battery_pct": leg.get("est_battery_pct"),
            }
            for leg in (raw.get("legs") or [])
        ],
        "total_duration_s": raw.get("total_duration_s"),
        "total_battery_pct": raw.get("total_battery_pct"),
        "battery_reserve_pct": raw.get("battery_reserve_pct"),
    }

    user_msg = (
        f"OPERATOR COMMAND: {state.get('raw_command', '')}\n\n"
        f"DRONE IN USE: {drone_profile_info}\n"
        f"FLEET (other drones available): {fleet}\n\n"
        f"WEATHER: {wx.get('summary', 'unknown')}\n"
        f"  wind {wx.get('wind_mps', '?')} m/s from {wx.get('wind_dir_deg', '?')}°, "
        f"vis {wx.get('visibility_m', '?')} m, precip {wx.get('precipitation_mmh', 0)} mm/h\n\n"
        f"PLAN (already passed safety kernel):\n{json.dumps(plan_summary, indent=2)}\n\n"
        "Produce an OptimizationAdvisory with exactly 2 concrete tactical suggestions. "
        "For EACH suggestion, also fill `apply_command`: a complete, self-contained "
        "command the operator can re-send to the planner to apply that one tweak. "
        "It MUST restate the full original mission intent (the planner has no memory "
        "of this plan) with the single change folded in — not just the delta. "
        "Include resource_constrained_fallback ONLY if battery reserve is under "
        "30% OR total duration is within 5 minutes of the drone's rated endurance "
        "OR the mission scope clearly exceeds one drone's reach."
    )

    llm = _llm_lazy()
    provider, model = settings.llm_provider, settings.llm_model
    start = time.perf_counter()
    try:
        # Native LangChain structured-output: the chat model returns a populated
        # pydantic object. No JSON-string parsing, no fence-stripping.
        structured_llm = llm.with_structured_output(OptimizationAdvisory)
        advisory: OptimizationAdvisory = structured_llm.invoke(
            [
                SystemMessage(content=_ADVISOR_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
    except Exception as e:  # noqa: BLE001
        log.warning("advisor LLM call failed: %s — continuing without it", e)
        LLM_CALLS_TOTAL.labels(provider, model, "error").inc()
        LLM_DURATION.labels(provider, model).observe(time.perf_counter() - start)
        return {}
    LLM_CALLS_TOTAL.labels(provider, model, "ok").inc()
    LLM_DURATION.labels(provider, model).observe(time.perf_counter() - start)

    # Defensive: providers occasionally return a dict instead of the pydantic
    # object (or a test mock returns a MagicMock). Anything that isn't a real
    # OptimizationAdvisory gets dropped — advisory is advisory only.
    if not isinstance(advisory, OptimizationAdvisory):
        log.warning("advisor returned %s, not OptimizationAdvisory — skipping", type(advisory))
        return {}

    def _clip(text: str | None, max_chars: int) -> str:
        s = (text or "").strip()
        return s if len(s) <= max_chars else s[: max_chars - 1].rstrip() + "…"

    advisory.summary = _clip(advisory.summary, 180)
    advisory.suggestions = advisory.suggestions[:2]
    for suggestion in advisory.suggestions:
        suggestion.title = _clip(suggestion.title, 70)
        suggestion.rationale = _clip(suggestion.rationale, 180)
        suggestion.apply_command = _clip(suggestion.apply_command, 260)
    if advisory.resource_constrained_fallback:
        advisory.resource_constrained_fallback = _clip(advisory.resource_constrained_fallback, 180)
    return {"advisory": advisory.model_dump()}


# ---- repair / clarify -------------------------------------------------------


def repair_node(state: CompileState) -> dict[str, Any]:
    """Increment the repair counter; the conditional edge sends us back to plan."""
    log.info("→ repair (n=%s)", state.get("repair_attempts", 0) + 1)
    return {"repair_attempts": state.get("repair_attempts", 0) + 1}


def clarify_node(state: CompileState) -> dict[str, Any]:
    """If intake didn't already populate questions, generate generic ones from
    the violations."""
    questions = state.get("clarification_questions") or []
    # When the planner itself chose NEEDS_CLARIFICATION it usually wrote specific,
    # well-targeted questions onto the draft plan — prefer those over generics.
    if not questions:
        draft = state.get("draft_plan") or {}
        questions = draft.get("clarification_questions") or []
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
        reasons = state.get("validation_errors") or [
            "planner could not produce a safe plan within the repair budget"
        ]
    return {
        "status": MissionStatus.REJECTED.value,
        "rejection_reasons": reasons,
    }


def _grounded_plan_summary(plan: MissionPlan, state: CompileState) -> str:
    """Build the operator-facing summary from validated plan facts."""
    if plan.status == MissionStatus.NEEDS_CLARIFICATION:
        return "No flight plan was generated; the request needs the listed operator clarifications."
    if plan.status == MissionStatus.REJECTED:
        reasons = [r for r in (plan.rejection_reasons or []) if r]
        if reasons:
            return "No flight plan was approved. Reason(s): " + "; ".join(reasons[:3])
        return "No flight plan was approved; the listed rejection reasons come from planner policy or kernel validation."
    if not plan.legs:
        return "No flight legs were produced."

    md = state.get("multi_drone_slot") or {}
    total = int(md.get("total", 1) or 1)
    if total > 1:
        fleet_text = (
            f"Drone {int(md.get('index', 0)) + 1} of {total}, latitude band "
            f"{int(md.get('index', 0)) + 1}/{total}, altitude slot "
            f"{float(md.get('alt_min', 0)):.0f}-{float(md.get('alt_max', 0)):.0f}m"
        )
    else:
        fleet_text = "Single-drone plan"

    waypoint_alts = [wp.alt_m for leg in plan.legs for wp in leg.geometry]
    alt_text = f"{min(waypoint_alts):.0f}-{max(waypoint_alts):.0f}m" if waypoint_alts else "n/a"
    sensors = sorted(
        {
            leg.sensor_mode.value
            for leg in plan.legs
            if leg.sensor_mode is not None and leg.sensor_mode.value != "OFF"
        }
    )
    sensor_text = ", ".join(sensors) if sensors else "no active sensor"
    search_passes = sum(1 for leg in plan.legs if leg.leg_type.value == "SEARCH_PATTERN")

    checks = [
        "geofence",
        "NFZ avoidance",
        "altitude limits",
        "battery reserve",
        "endurance",
        "sensor coverage",
        "RTB endpoint",
    ]
    report = plan.constraints_satisfied
    if report and report.weather_acceptable_detail != "not evaluated":
        checks.append("weather")
    if report and report.airspace_deconflicted_detail != "not evaluated":
        checks.append("airspace deconfliction")

    return (
        f"{fleet_text}: {len(plan.legs)} legs, {search_passes} coverage pass"
        f"{'' if search_passes == 1 else 'es'}, {sensor_text}, altitude {alt_text}. "
        f"Estimated {plan.total_duration_s / 60.0:.1f} min, "
        f"{plan.total_battery_pct:.0f}% battery used, "
        f"{plan.battery_reserve_pct:.0f}% reserve; kernel passed "
        f"{', '.join(checks)}."
    )


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

    plan.reasoning_trace = _grounded_plan_summary(plan, state)
    if state.get("advisory"):
        from app.schemas.plan import OptimizationAdvisory

        try:
            plan.advisory = OptimizationAdvisory.model_validate(state["advisory"])
        except Exception as e:  # noqa: BLE001
            log.warning("advisor payload failed schema: %s", e)

    # Record plan-shape metrics only for plans that actually flew through the
    # planner (skip NEEDS_CLARIFICATION/REJECTED-from-intake — those have no
    # legs and would skew the histograms toward zero).
    if plan.status == MissionStatus.READY_FOR_APPROVAL and plan.legs:
        PLAN_LEGS.observe(len(plan.legs))
        PLAN_BATTERY_PCT.observe(plan.total_battery_pct)
        PLAN_DURATION.observe(plan.total_duration_s)

    try:
        save_mission(
            get_engine(),
            plan.model_dump(),
            command=state.get("raw_command", ""),
            drone_profile_id=(state.get("drone_state") or {}).get("drone_profile_id", ""),
            repair_drafts=state.get("repair_drafts") or [],
        )
    except Exception as e:  # noqa: BLE001
        log.warning("save_mission skipped: %s", e)

    return {"draft_plan": plan.model_dump(), "status": plan.status.value}
