from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.plan import MissionPlan


class DroneState(BaseModel):
    """Snapshot of the drone the operator wants to use."""

    drone_profile_id: str = Field(..., description="ID of YAML profile in data/drones/")
    battery_pct: float = Field(100.0, ge=0.0, le=100.0)
    home_lat: float | None = None
    home_lon: float | None = None


class CompileRequest(BaseModel):
    """Operator request to compile a natural-language command into a mission plan."""

    command: str = Field(..., min_length=1, description="Operator natural-language command")
    area_id: str = Field(..., description="Operating area ID; must exist in geo store")
    operator_clearance: str = Field("STANDARD", description="Auth level for safety gates")
    drone_state: DroneState
    request_id: str | None = Field(None, description="Client-supplied idempotency key")
    alternatives: bool = Field(
        False,
        description="If true, the planner emits 2–3 alternative plans (see DESIGN_DECISIONS §7).",
    )
    planning_effort: str = Field(
        "",
        description=(
            "Planner effort tier: 'fast' (one direct LLM call — rate-limit "
            "friendly), 'balanced' (tool-calling agent), or 'thorough' (agent on "
            "the strongest model). Empty = server default."
        ),
    )
    selected_sensors: list[str] = Field(
        default_factory=list,
        description=(
            "Sensors the operator explicitly enabled (subset of {EO, IR}). The "
            "planner constrains every leg's sensor_mode to this set. Empty = "
            "planner default (EO / visual)."
        ),
    )
    drone_ids: list[str] | None = Field(
        None,
        description="If set, compile a multi-drone group plan, one per id (see DESIGN_DECISIONS §9).",
    )
    auto_drones: bool = Field(
        False,
        description=(
            "If true, the LLM picks how many drones and which profiles to use "
            "from the available fleet (see app/graph/auto_drones.py). Ignored "
            "when `drone_ids` is also set — explicit IDs win."
        ),
    )
    drones_vs_time_bias: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Bias for auto_drones. 0.0 = minimize drone count, 1.0 = minimize "
            "mission wall-clock time. 0.5 is balanced."
        ),
    )
    # Chat-style clarification thread (DD-005). When the prior compile returned
    # NEEDS_CLARIFICATION, the UI sends the prior turns so the planner sees the
    # full conversation, not just the latest follow-up.
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Prior {role, content} turns from this clarification thread.",
    )


class DroneUpsertRequest(BaseModel):
    """Operator-authored drone profile. POST /v1/drones."""

    profile_id: str = Field(..., min_length=1, max_length=64)
    rated_endurance_s: float = Field(..., gt=0)
    cruise_speed_mps: float = Field(..., gt=0)
    climb_rate_mps: float = Field(..., gt=0)
    battery_wh: float = Field(..., gt=0)
    cruise_power_w: float = Field(..., gt=0)
    hover_power_w: float = Field(..., gt=0)
    description: str = ""
    sensors: list[dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "name": "nadir_eo",
                "mode": "EO",
                "power_w": 10.0,
                "half_angle_deg": 30.0,
                "ground_resolution_at_100m": 0.04,
            },
        ]
    )


class AreaResearchRequest(BaseModel):
    """Polygon submitted to the LLM-research endpoint (DD-007)."""

    boundary: dict[str, Any] = Field(..., description="GeoJSON Polygon")
    home_lon: float = Field(..., ge=-180.0, le=180.0)
    home_lat: float = Field(..., ge=-90.0, le=90.0)


class AreaResearchResponse(BaseModel):
    flight_permitted: bool
    ceiling_m: float
    suggested_nfzs: list[dict[str, Any]] = Field(default_factory=list)
    notes: str


class CompileResponse(BaseModel):
    """Response envelope wrapping the produced (or non-) plan."""

    plan: MissionPlan
    repair_loops: int = 0
    awaiting_approval: bool = False
    # Extra alternatives (besides `plan`) for the operator to choose from.
    # Each alternative has already passed the safety kernel.
    alternatives: list[MissionPlan] = Field(default_factory=list)
    # Multi-drone group response (one entry per drone_id; otherwise empty).
    group_plans: list[MissionPlan] = Field(default_factory=list)
    # When auto_drones was used, this is the LLM's fleet-pick justification
    # ("3 drones because the area is ~1.2 km² and bias favored speed"). Empty
    # for manually-specified or single-drone missions.
    auto_drones_reasoning: str = ""


class ApprovalRequest(BaseModel):
    mission_id: str
    approve: bool
    operator_note: str = ""


class ApprovalResponse(BaseModel):
    mission_id: str
    final_status: str
    operator_note: str = ""


class VerifyResponse(BaseModel):
    mission_id: str
    flown_ok: bool
    actual_duration_s: float
    actual_battery_pct: float
    deviations: list[str]


class ChatTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class MissionChatRequest(BaseModel):
    """Follow-up chat about an already-compiled mission.

    The frontend keeps the full history in memory and sends it on every turn so
    the backend can stay stateless. The LLM sees: a system prompt scoped to the
    plan, the prior turns as LangChain messages, and the new user message.
    """

    history: list[ChatTurn] = Field(
        default_factory=list,
        description="Prior turns including the new user message at the end.",
    )


class MissionChatResponse(BaseModel):
    reply: str
    mission_id: str
    # When the operator asks the chat to redo/retry the mission, the LLM emits a
    # sentinel and we surface a structured signal so the UI can offer a one-
    # click recompile (reusing the current area + drone selection). Empty when
    # the chat is just a Q&A turn.
    action: str = ""  # one of: "", "retry"
    refined_command: str = ""


class AreaUpsertRequest(BaseModel):
    """Request body for POST /v1/areas — upserts an operating area drawn in the UI."""

    area_id: str = Field(..., min_length=1, max_length=64)
    boundary: dict[str, Any] = Field(..., description="GeoJSON Polygon")
    nfzs: list[dict[str, Any]] = Field(default_factory=list, description="GeoJSON Polygons")
    ceiling_m: float = Field(120.0, gt=0.0, le=500.0)
    home_lon: float = Field(..., ge=-180.0, le=180.0)
    home_lat: float = Field(..., ge=-90.0, le=90.0)
    home_bases: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Optional list of [lon, lat] home/base points. Falls back to home_lon/home_lat.",
    )


class AreaUpsertResponse(BaseModel):
    area_id: str
    home_lon: float
    home_lat: float
    home_bases: list[tuple[float, float]] = Field(default_factory=list)
    nfz_count: int
    home_was_snapped: bool
