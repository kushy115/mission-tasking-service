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
    haversine_m,
    sensor_swath_m,
)
from app.weather import WeatherObservation


@dataclass(frozen=True)
class GeoContext:
    """Spatial constraints for an operating area. Polygons are in lon/lat."""

    area_id: str
    boundary: Polygon
    nfz_polygons: tuple[Polygon, ...]
    altitude_ceiling_m: float
    home_point: tuple[float, float]  # (lon, lat)
    home_bases: tuple[tuple[float, float], ...] = ()  # registered bases, each (lon, lat)


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


def _altitude_ok(ceiling_m: float, min_agl_m: float, plan: MissionPlan) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for li, leg in enumerate(plan.legs):
        for wi, wp in enumerate(leg.geometry):
            if wp.alt_m > ceiling_m:
                bad.append(f"leg[{li}].waypoint[{wi}] alt {wp.alt_m}m exceeds ceiling {ceiling_m}m")
            if wp.alt_m < min_agl_m:
                bad.append(f"leg[{li}].waypoint[{wi}] alt {wp.alt_m}m below min AGL {min_agl_m}m")
    return (not bad, bad)


def _battery_ok(
    plan: MissionPlan,
    profile: DroneProfile,
    reserve_min_pct: float,
    starting_pct: float,
    *,
    wind_mps: float = 0.0,
    wind_coeff: float = 0.0,
) -> tuple[bool, str, float, float]:
    energy = estimate_mission(plan.legs, profile, wind_mps=wind_mps, wind_coeff=wind_coeff)
    reserve_pct = starting_pct - energy.total_battery_pct
    ok = reserve_pct >= reserve_min_pct
    wind_note = f"; wind_penalty=+{wind_coeff * wind_mps * 100:.1f}% power" if wind_mps > 0 else ""
    detail = (
        f"estimated_use={energy.total_battery_pct:.1f}%; "
        f"reserve={reserve_pct:.1f}% (min {reserve_min_pct}%){wind_note}"
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
        # Measure track spacing by clustering waypoints into east-west ROWS by
        # latitude, then taking the median gap between consecutive row centers.
        #
        # The old method assumed a clean 2-waypoints-per-track serpentine and
        # read odd-indexed hops as the perpendicular spacing. That breaks for
        # NFZ-aware patterns: when a track is split by a no-fly zone (or detour
        # waypoints are inserted to route around one) the per-row waypoint count
        # is variable, so the odd/even indices no longer line up with the
        # perpendicular hops and it reports a bogus spacing. Clustering by
        # latitude is robust to variable-length rows and NFZ detours. We use the
        # MEDIAN gap so a few large jumps (e.g. skirting an NFZ) don't dominate —
        # coverage gaps that fall inside a no-fly zone are unavoidable and not a
        # real sensing failure. See DD-010.
        from app.geo.patterns import METERS_PER_DEG_LAT  # noqa: PLC0415

        lats = sorted({round(wp.lat, 6) for wp in leg.geometry})
        # Merge lats within ~2m into the same row.
        row_merge_deg = 2.0 / METERS_PER_DEG_LAT
        rows: list[float] = []
        for lat in lats:
            if not rows or (lat - rows[-1]) > row_merge_deg:
                rows.append(lat)
        if len(rows) < 2:
            continue
        gaps_m = sorted((rows[i + 1] - rows[i]) * METERS_PER_DEG_LAT for i in range(len(rows) - 1))
        mid = len(gaps_m) // 2
        median_spacing = (
            gaps_m[mid] if len(gaps_m) % 2 == 1 else (gaps_m[mid - 1] + gaps_m[mid]) / 2.0
        )
        if median_spacing > swath:
            bad.append(
                f"leg[{li}] track spacing {median_spacing:.1f}m exceeds sensor swath "
                f"{swath:.1f}m at {min_alt:.0f}m AGL — coverage gap"
            )
    return (not bad, bad)


def _ends_with_rtb(
    plan: MissionPlan, home_bases: tuple[tuple[float, float], ...]
) -> tuple[bool, str]:
    if not plan.legs:
        return False, "plan has zero legs"
    last = plan.legs[-1]
    if last.leg_type != LegType.RETURN_TO_BASE:
        return False, f"last leg is {last.leg_type.value}, expected RETURN_TO_BASE"
    if not last.geometry:
        return False, "RETURN_TO_BASE leg has no waypoints"
    final_wp = last.geometry[-1]
    if not home_bases:
        return False, "no registered home bases available for RETURN_TO_BASE check"
    distances = [
        haversine_m(final_wp, Waypoint(lat=lat, lon=lon, alt_m=final_wp.alt_m))
        for lon, lat in home_bases
    ]
    nearest_m = min(distances)
    tolerance_m = 5.0
    if nearest_m > tolerance_m:
        return (
            False,
            f"final RETURN_TO_BASE waypoint is {nearest_m:.1f}m from nearest registered home base",
        )
    return True, "ok"


def _weather_ok(plan: MissionPlan, obs: WeatherObservation) -> tuple[bool, list[str], str]:
    """Hard reject conditions only (see DESIGN_DECISIONS.md §4). Anything below
    these thresholds is just a battery penalty in the physics model."""
    settings = get_settings()
    bad: list[str] = []
    if obs.wind_mps > settings.weather_max_wind_mps:
        bad.append(
            f"sustained wind {obs.wind_mps:.1f} m/s exceeds drone tolerance "
            f"{settings.weather_max_wind_mps:.1f} m/s"
        )
    if obs.gust_mps > settings.weather_max_gust_mps:
        bad.append(
            f"wind gusts {obs.gust_mps:.1f} m/s exceed drone tolerance "
            f"{settings.weather_max_gust_mps:.1f} m/s"
        )
    if obs.precipitation_mmh > settings.weather_max_precip_mmh:
        bad.append(
            f"precipitation {obs.precipitation_mmh:.1f} mm/h exceeds limit "
            f"{settings.weather_max_precip_mmh:.1f} mm/h (drone not waterproof)"
        )
    if obs.visibility_m < settings.weather_min_visibility_m:
        uses_eo = any(leg.sensor_mode == SensorMode.EO for leg in plan.legs)
        if uses_eo:
            bad.append(
                f"visibility {obs.visibility_m:.0f}m below {settings.weather_min_visibility_m:.0f}m "
                "and plan uses EO sensor (switch to IR or wait)"
            )
    detail = obs.summary_for_prompt() if not bad else "; ".join(bad)
    return (not bad, bad, detail)


def validate_plan(
    plan: MissionPlan,
    geo: GeoContext,
    profile: DroneProfile,
    starting_battery_pct: float = 100.0,
    weather: WeatherObservation | None = None,
    deconfliction: tuple[bool, list[str]] | None = None,
) -> tuple[list[str], ConstraintReport]:
    """Run every safety check. Returns (violations, structured report).

    `weather` and `deconfliction` are optional; when not supplied those checks
    pass trivially. See docs/DESIGN_DECISIONS.md §4 and §8.
    """
    settings = get_settings()
    violations: list[str] = []

    geo_ok, geo_bad = _waypoints_inside(geo.boundary, plan)
    violations.extend(geo_bad)

    nfz_ok, nfz_bad = _avoids_nfz(geo.nfz_polygons, plan)
    violations.extend(nfz_bad)

    alt_ok, alt_bad = _altitude_ok(geo.altitude_ceiling_m, settings.min_agl_m, plan)
    violations.extend(alt_bad)

    # Wind penalty propagates into the energy estimate when weather is supplied.
    wind_mps = weather.wind_mps if weather is not None else 0.0
    wind_coeff = settings.weather_wind_power_coeff if weather is not None else 0.0
    batt_ok, batt_detail, batt_used, batt_reserve = _battery_ok(
        plan,
        profile,
        settings.battery_reserve_min_pct,
        starting_battery_pct,
        wind_mps=wind_mps,
        wind_coeff=wind_coeff,
    )
    if not batt_ok:
        violations.append(batt_detail)

    end_ok, end_detail, total_dur = _endurance_ok(plan, profile)
    if not end_ok:
        violations.append(end_detail)

    cov_ok, cov_bad = _sensor_coverage_ok(plan, profile)
    violations.extend(cov_bad)

    home_bases = geo.home_bases or (geo.home_point,)
    rtb_ok, rtb_detail = _ends_with_rtb(plan, home_bases)
    if not rtb_ok:
        violations.append(rtb_detail)

    if weather is not None:
        wx_ok, wx_bad, wx_detail = _weather_ok(plan, weather)
        violations.extend(wx_bad)
    else:
        wx_ok, wx_detail = True, "not evaluated"

    if deconfliction is not None:
        dec_ok, dec_bad = deconfliction
        violations.extend(dec_bad)
        dec_detail = "; ".join(dec_bad) if dec_bad else "ok"
    else:
        dec_ok, dec_detail = True, "not evaluated"

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
        weather_acceptable=wx_ok,
        weather_acceptable_detail=wx_detail,
        airspace_deconflicted=dec_ok,
        airspace_deconflicted_detail=dec_detail,
    )

    # Stamp aggregate fields back on the plan for downstream consumers.
    energy = estimate_mission(plan.legs, profile, wind_mps=wind_mps, wind_coeff=wind_coeff)
    plan.total_duration_s = energy.total_duration_s
    plan.total_battery_pct = energy.total_battery_pct
    plan.battery_reserve_pct = max(0.0, starting_battery_pct - energy.total_battery_pct)
    plan.constraints_satisfied = report

    return violations, report
