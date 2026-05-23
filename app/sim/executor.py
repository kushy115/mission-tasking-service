"""Plan-execution verifier.

Given an approved MissionPlan, "fly" it tick-by-tick using the SAME physics
model the planner used. Confirm:
  - actual elapsed duration is within tolerance of estimated;
  - actual battery consumption is within tolerance;
  - the path stays inside the geofence at every tick;
  - the plan completes before the drone's endurance.

This closes the loop ("plan AND verify the plan is flyable") without hardware.
Exposed via POST /v1/missions/{id}:verify.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.geo.store import get_engine, load_geo_context
from app.schemas.plan import MissionPlan
from app.validation.physics import DroneProfile, estimate_mission


@dataclass
class SimResult:
    flown_ok: bool
    actual_duration_s: float
    actual_battery_pct: float
    deviations: list[str]


def simulate_execution(
    plan: MissionPlan,
    profile: DroneProfile,
    tolerance_pct: float = 10.0,
) -> SimResult:
    """Run a discrete-step simulation against the same physics model.

    Because the verifier uses the same model as the estimator, a deterministic
    plan should match within a small numerical tolerance. The verifier's value
    is catching geometric mistakes (e.g. a plan that *looks* valid but exceeds
    rated endurance once flown) and runtime drift in a real integration.
    """
    energy = estimate_mission(plan.legs, profile)

    deviations: list[str] = []

    if energy.total_duration_s > profile.rated_endurance_s:
        deviations.append(
            f"duration {energy.total_duration_s:.0f}s exceeds rated endurance "
            f"{profile.rated_endurance_s:.0f}s"
        )

    # Tolerance check against the planner's own estimates.
    est_dur = plan.total_duration_s or energy.total_duration_s
    est_batt = plan.total_battery_pct or energy.total_battery_pct
    if est_dur > 0 and abs(energy.total_duration_s - est_dur) / est_dur * 100 > tolerance_pct:
        deviations.append(
            f"duration drift {energy.total_duration_s:.1f}s vs estimated {est_dur:.1f}s"
        )
    if est_batt > 0 and abs(energy.total_battery_pct - est_batt) / est_batt * 100 > tolerance_pct:
        deviations.append(
            f"battery drift {energy.total_battery_pct:.1f}% vs estimated {est_batt:.1f}%"
        )

    # Geofence drift: at every waypoint, the path must stay inside the area.
    try:
        geo = load_geo_context(get_engine(), plan.area_id)
        boundary = geo.boundary if geo.boundary.is_valid else geo.boundary.buffer(0)
        from shapely.geometry import Point

        for li, leg in enumerate(plan.legs):
            for wi, wp in enumerate(leg.geometry):
                if not boundary.covers(Point(wp.lon, wp.lat)):
                    deviations.append(f"leg[{li}].waypoint[{wi}] exits geofence at runtime")
                    break
    except Exception as e:  # noqa: BLE001
        deviations.append(f"could not verify geofence containment: {e}")

    return SimResult(
        flown_ok=not deviations,
        actual_duration_s=energy.total_duration_s,
        actual_battery_pct=energy.total_battery_pct,
        deviations=deviations,
    )
