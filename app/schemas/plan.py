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
    # NOTE: no upper bound — this is *consumption*, not "remaining". A leg whose
    # geometry can't be flown on a full charge legitimately reports >100, and the
    # safety kernel needs that real number to reject via battery_within_budget.
    # Capping here would silently hide infeasible plans.
    est_battery_pct: float = Field(..., ge=0.0)


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
    # Weather and deconfliction are optional checks: True by default so old
    # records without those fields still appear satisfied.
    weather_acceptable: bool = True
    weather_acceptable_detail: str = "not evaluated"
    airspace_deconflicted: bool = True
    airspace_deconflicted_detail: str = "not evaluated"

    @property
    def all_satisfied(self) -> bool:
        return (
            self.inside_geofence
            and self.avoids_nfz
            and self.battery_within_budget
            and self.within_endurance
            and self.sensor_coverage_adequate
            and self.ends_with_rtb
            and self.weather_acceptable
            and self.airspace_deconflicted
        )


class OptimizationSuggestion(BaseModel):
    """One concrete tweak the advisor recommends to the operator.

    Suggestions are advisory, additive, and never block approval. The advisor
    is told the plan already passed the safety kernel — its job is tactical
    judgment ("rotate the sweep cross-wind", "swap to IR for the dawn haze",
    "add a second drone on layer 70 m") not safety.
    """

    title: str = Field(..., description="One-line headline, e.g. 'Sweep parallel to wind'.")
    rationale: str = Field(..., description="Why this is better given current conditions.")
    impact: str = Field(
        "moderate",
        description="One of: minor | moderate | major. Used to sort suggestions in the UI.",
    )
    category: str = Field(
        "tactical",
        description="One of: weather | resource | coverage | coordination | tactical.",
    )


class OptimizationAdvisory(BaseModel):
    """Bundle of LLM-authored optimizations attached to a READY_FOR_APPROVAL plan."""

    summary: str = Field("", description="One paragraph framing the advisor's overall take.")
    suggestions: list[OptimizationSuggestion] = Field(default_factory=list)
    resource_constrained_fallback: str | None = Field(
        None,
        description=(
            "If the plan is strained by resource limits (battery margin, endurance, "
            "single-drone capacity vs. area size), a 1–3 sentence sketch of a "
            "scoped-down mission that still accomplishes the primary goal."
        ),
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
    # Advisory critique (LLM, see DESIGN_DECISIONS.md §6). None = not run.
    confidence_score: float | None = None
    critique_notes: str = ""
    # Optimization advisor: LangChain structured-output pass that proposes
    # tactical tweaks (weather-aware re-orientations, multi-drone splits,
    # resource-constrained scope reductions). Advisory only — never blocks.
    advisory: OptimizationAdvisory | None = None
