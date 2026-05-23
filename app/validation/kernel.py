"""Deterministic safety-kernel validator.

This module is the SAFETY CORE. It is pure Python — no LLM, no network — and
its tests in tests/test_validation.py are mandatory. The validator is the
source of truth for whether a plan is safe to approve; the LLM does not get
a vote.

Inputs:
- a MissionPlan produced by the planner;
- a GeoContext: operating-area polygon, NFZ polygons, altitude ceiling;
- a DroneProfile (physics).

Outputs:
- a list of human-readable violation strings (empty == safe);
- a ConstraintReport with a boolean + detail per check.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon

from app.config import get_settings
from app.schemas.enums import LegType, SensorMode
from app.schemas.plan import ConstraintReport, MissionLeg, MissionPlan, Waypoint
from app.validation.physics import (
    DroneProfile,
    estimate_mission,
    sensor_swath_m,
)


@dataclass(frozen=True)
class GeoContext:
    """Spatial constraints for an operating area. Polygons are in lon/lat."""

    area_id: str
    boundary: Polygon
    nfz_polygons: tuple[Polygon, ...]
    altitude_ceiling_m: float
    home_point: tuple[float, float]  # (lon, lat)


def _wp_point(wp: Waypoint) -> Point:
    return Point(wp.lon, wp.lat)


def _leg_line(leg: MissionLeg) -> LineString | Point:
    if len(leg.geometry) == 1:
        return _wp_point(leg.geometry[0])
    return LineString([(wp.lon, wp.lat) for wp in leg.geometry])


def _waypoints_inside(boundary: Polygon, plan: MissionPlan) -> tuple[bool, list[str]]:
    bad: list[str] = []
    # buffer(0) repairs any minor self-intersection issues
    safe_boundary = boundary if boundary.is_valid else boundary.buffer(0)
    for li, leg in enumerate(plan.legs):
        for wi, wp in enumerate(leg.geometry):
            if not safe_boundary.covers(_wp_point(wp)):
                bad.append(f"leg[{li}].waypoint[{wi}] ({wp.lat:.5f},{wp.lon:.5f}) outside geofence")
    return (not bad, bad)


def _avoids_nfz(nfzs: tuple[Polygon, ...], plan: MissionPlan) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for li, leg in enumerate(plan.legs):
        line = _leg_line(leg)
        for ni, nfz in enumerate(nfzs):
            poly = nfz if nfz.is_valid else nfz.buffer(0)
            if line.intersects(poly):
                bad.append(f"leg[{li}] intersects no-fly-zone[{ni}]")
    return (not bad, bad)


def _altitude_ok(
    ceiling_m: float, min_agl_m: float, plan: MissionPlan
) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for li, leg in enumerate(plan.legs):
        for wi, wp in enumerate(leg.geometry):
            if wp.alt_m > ceiling_m:
                bad.append(f"leg[{li}].waypoint[{wi}] alt {wp.alt_m}m exceeds ceiling {ceiling_m}m")
            if wp.alt_m < min_agl_m:
                bad.append(
                    f"leg[{li}].waypoint[{wi}] alt {wp.alt_m}m below min AGL {min_agl_m}m"
                )
    return (not bad, bad)


def _battery_ok(
    plan: MissionPlan, profile: DroneProfile, reserve_min_pct: float, starting_pct: float
) -> tuple[bool, str, float, float]:
    energy = estimate_mission(plan.legs, profile)
    reserve_pct = starting_pct - energy.total_battery_pct
    ok = reserve_pct >= reserve_min_pct
    detail = (
        f"estimated_use={energy.total_battery_pct:.1f}%; "
        f"reserve={reserve_pct:.1f}% (min {reserve_min_pct}%)"
    )
    return ok, detail, energy.total_battery_pct, reserve_pct


def _endurance_ok(plan: MissionPlan, profile: DroneProfile) -> tuple[bool, str, float]:
    energy = estimate_mission(plan.legs, profile)
    ok = energy.total_duration_s <= profile.rated_endurance_s
    detail = (
        f"total_duration={energy.total_duration_s:.0f}s; "
        f"rated_endurance={profile.rated_endurance_s:.0f}s"
    )
    return ok, detail, energy.total_duration_s


def _sensor_coverage_ok(plan: MissionPlan, profile: DroneProfile) -> tuple[bool, list[str]]:
    """For SEARCH_PATTERN legs, leg.geometry should be a serpentine of parallel
    tracks; consecutive parallel tracks must be no farther apart than the
    sensor swath at their altitude to guarantee no coverage gap.
    Heuristic: check the minimum altitude in the leg gives a swath at least
    equal to the mean track spacing.
    """
    bad: list[str] = []
    for li, leg in enumerate(plan.legs):
        if leg.leg_type != LegType.SEARCH_PATTERN:
            continue
        if leg.sensor_mode is None or leg.sensor_mode == SensorMode.OFF:
            bad.append(f"leg[{li}] is SEARCH_PATTERN but sensor_mode is OFF")
            continue
        sensor = profile.sensor(leg.sensor_mode)
        if sensor is None:
            bad.append(f"leg[{li}] requests sensor {leg.sensor_mode} not on drone")
            continue
        if len(leg.geometry) < 4:
            bad.append(f"leg[{li}] SEARCH_PATTERN has <4 waypoints, cannot ensure coverage")
            continue
        min_alt = min(wp.alt_m for wp in leg.geometry)
        swath = sensor_swath_m(sensor, min_alt)
        # Estimate track spacing: average distance between every other consecutive
        # waypoint pair (the perpendicular hop between adjacent tracks).
        # This is a rough but principled measure for a serpentine.
        from app.validation.physics import haversine_m

        hops = [
            haversine_m(leg.geometry[i], leg.geometry[i + 1])
            for i in range(0, len(leg.geometry) - 1, 2)
        ]
        if not hops:
            continue
        mean_spacing = sum(hops) / len(hops)
        if mean_spacing > swath:
            bad.append(
                f"leg[{li}] track spacing {mean_spacing:.1f}m exceeds sensor swath "
                f"{swath:.1f}m at {min_alt:.0f}m AGL — coverage gap"
            )
    return (not bad, bad)


def _ends_with_rtb(plan: MissionPlan) -> tuple[bool, str]:
    if not plan.legs:
        return False, "plan has zero legs"
    last = plan.legs[-1]
    if last.leg_type != LegType.RETURN_TO_BASE:
        return False, f"last leg is {last.leg_type.value}, expected RETURN_TO_BASE"
    return True, "ok"


def validate_plan(
    plan: MissionPlan,
    geo: GeoContext,
    profile: DroneProfile,
    starting_battery_pct: float = 100.0,
) -> tuple[list[str], ConstraintReport]:
    """Run every safety check. Returns (violations, structured report)."""
    settings = get_settings()
    violations: list[str] = []

    geo_ok, geo_bad = _waypoints_inside(geo.boundary, plan)
    violations.extend(geo_bad)

    nfz_ok, nfz_bad = _avoids_nfz(geo.nfz_polygons, plan)
    violations.extend(nfz_bad)

    alt_ok, alt_bad = _altitude_ok(geo.altitude_ceiling_m, settings.min_agl_m, plan)
    violations.extend(alt_bad)

    batt_ok, batt_detail, batt_used, batt_reserve = _battery_ok(
        plan, profile, settings.battery_reserve_min_pct, starting_battery_pct
    )
    if not batt_ok:
        violations.append(batt_detail)

    end_ok, end_detail, total_dur = _endurance_ok(plan, profile)
    if not end_ok:
        violations.append(end_detail)

    cov_ok, cov_bad = _sensor_coverage_ok(plan, profile)
    violations.extend(cov_bad)

    rtb_ok, rtb_detail = _ends_with_rtb(plan)
    if not rtb_ok:
        violations.append(rtb_detail)

    report = ConstraintReport(
        inside_geofence=geo_ok,
        inside_geofence_detail="; ".join(geo_bad) if geo_bad else "ok",
        avoids_nfz=nfz_ok,
        avoids_nfz_detail="; ".join(nfz_bad) if nfz_bad else "ok",
        battery_within_budget=batt_ok,
        battery_within_budget_detail=batt_detail,
        within_endurance=end_ok,
        within_endurance_detail=end_detail,
        sensor_coverage_adequate=cov_ok,
        sensor_coverage_adequate_detail="; ".join(cov_bad) if cov_bad else "ok",
        ends_with_rtb=rtb_ok,
        ends_with_rtb_detail=rtb_detail,
    )

    # Stamp aggregate fields back on the plan for downstream consumers.
    energy = estimate_mission(plan.legs, profile)
    plan.total_duration_s = energy.total_duration_s
    plan.total_battery_pct = energy.total_battery_pct
    plan.battery_reserve_pct = max(0.0, starting_battery_pct - energy.total_battery_pct)
    plan.constraints_satisfied = report

    return violations, report
