from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.enums import LegType, MissionStatus, PatternName, SensorMode


class Waypoint(BaseModel):
    """A single 3D point on the mission path."""

    lat: float = Field(..., ge=-90.0, le=90.0, description="WGS84 latitude in degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="WGS84 longitude in degrees")
    alt_m: float = Field(..., ge=0.0, description="Altitude AGL in meters")


class MissionLeg(BaseModel):
    """One ordered segment of a mission. Geometry is leg-typed (transit vs pattern)."""

    leg_type: LegType
    geometry: list[Waypoint] = Field(..., min_length=1)
    pattern_name: PatternName | None = None
    sensor_mode: SensorMode | None = None
    est_duration_s: float = Field(..., ge=0.0)
    est_battery_pct: float = Field(..., ge=0.0, le=100.0)


class ConstraintReport(BaseModel):
    """Result of running a draft plan through the deterministic safety kernel."""

    inside_geofence: bool
    inside_geofence_detail: str = ""
    avoids_nfz: bool
    avoids_nfz_detail: str = ""
    battery_within_budget: bool
    battery_within_budget_detail: str = ""
    within_endurance: bool
    within_endurance_detail: str = ""
    sensor_coverage_adequate: bool
    sensor_coverage_adequate_detail: str = ""
    ends_with_rtb: bool
    ends_with_rtb_detail: str = ""

    @property
    def all_satisfied(self) -> bool:
        return (
            self.inside_geofence
            and self.avoids_nfz
            and self.battery_within_budget
            and self.within_endurance
            and self.sensor_coverage_adequate
            and self.ends_with_rtb
        )


class MissionPlan(BaseModel):
    """The output contract: a validated mission plan or a structured non-plan response.

    The planning LLM is constrained to this schema (response_format=MissionPlan).
    The deterministic validator (validation/kernel.py) is the source of truth on
    safety — the LLM does not get to decide constraints_satisfied.
    """

    mission_id: str
    area_id: str
    status: MissionStatus
    legs: list[MissionLeg] = Field(default_factory=list)
    total_duration_s: float = 0.0
    total_battery_pct: float = 0.0
    battery_reserve_pct: float = 0.0
    constraints_satisfied: ConstraintReport | None = None
    reasoning_trace: str = ""
    clarification_questions: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
