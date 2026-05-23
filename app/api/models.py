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


class CompileResponse(BaseModel):
    """Response envelope wrapping the produced (or non-) plan."""

    plan: MissionPlan
    repair_loops: int = 0
    awaiting_approval: bool = False


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
