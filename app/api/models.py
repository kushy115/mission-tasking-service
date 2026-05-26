from __future__ import annotations

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
    drone_ids: list[str] | None = Field(
        None,
        description="If set, compile a multi-drone group plan, one per id (see DESIGN_DECISIONS §9).",
    )
    # Chat-style clarification thread (DD-005). When the prior compile returned
    # NEEDS_CLARIFICATION, the UI sends the prior turns so the planner sees the
    # full conversation, not just the latest follow-up.
    conversation_history: list[dict] = Field(
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
    sensors: list[dict] = Field(
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

    boundary: dict = Field(..., description="GeoJSON Polygon")
    home_lon: float = Field(..., ge=-180.0, le=180.0)
    home_lat: float = Field(..., ge=-90.0, le=90.0)


class AreaResearchResponse(BaseModel):
    flight_permitted: bool
    ceiling_m: float
    suggested_nfzs: list[dict] = Field(default_factory=list)
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


class AreaUpsertRequest(BaseModel):
    """Request body for POST /v1/areas — upserts an operating area drawn in the UI."""

    area_id: str = Field(..., min_length=1, max_length=64)
    boundary: dict = Field(..., description="GeoJSON Polygon")
    nfzs: list[dict] = Field(default_factory=list, description="GeoJSON Polygons")
    ceiling_m: float = Field(120.0, gt=0.0, le=500.0)
    home_lon: float = Field(..., ge=-180.0, le=180.0)
    home_lat: float = Field(..., ge=-90.0, le=90.0)


class AreaUpsertResponse(BaseModel):
    area_id: str
    home_lon: float
    home_lat: float
    nfz_count: int
    home_was_snapped: bool


class SupervisorInjectRequest(BaseModel):
    """Inject an in-flight event into a live mission session.

    `session_id` is the per-WS client identifier (UUID-ish; UI generates it
    before opening the WebSocket and passes it both in the WS query string
    and on this POST). The supervisor's policy reads the event off the queue
    on its next tick. See DESIGN_DECISIONS DD-014.
    """

    session_id: str = Field(..., min_length=1, max_length=64)
    type: str = Field(..., description="EventType value (e.g. WIND_SPIKE)")
    payload: dict = Field(default_factory=dict)
    note: str = Field("", max_length=256)


class SupervisorInjectResponse(BaseModel):
    accepted: bool
    queue_size: int
    detail: str = ""
