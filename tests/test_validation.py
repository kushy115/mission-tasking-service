"""Tests for the deterministic safety kernel. Safety core — keep thorough."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from app.schemas.enums import LegType, MissionStatus, SensorMode
from app.schemas.plan import MissionLeg, MissionPlan, Waypoint
from app.validation.kernel import GeoContext, validate_plan
from app.validation.physics import DroneProfile, SensorSpec


@pytest.fixture
def profile() -> DroneProfile:
    return DroneProfile(
        profile_id="test-quad",
        rated_endurance_s=1800.0,
        cruise_speed_mps=15.0,
        climb_rate_mps=3.0,
        battery_wh=300.0,
        cruise_power_w=300.0,
        hover_power_w=400.0,
        sensors=(
            SensorSpec(
                name="eo",
                mode=SensorMode.EO,
                power_w=10.0,
                half_angle_deg=30.0,
                ground_resolution_at_100m=0.05,
            ),
        ),
    )


@pytest.fixture
def geo() -> GeoContext:
    boundary = Polygon(
        [(-74.01, 40.0), (-74.0, 40.0), (-74.0, 40.01), (-74.01, 40.01), (-74.01, 40.0)]
    )
    nfz = Polygon(
        [(-74.005, 40.005), (-74.004, 40.005), (-74.004, 40.006), (-74.005, 40.006)]
    )
    return GeoContext(
        area_id="area-test",
        boundary=boundary,
        nfz_polygons=(nfz,),
        altitude_ceiling_m=120.0,
        home_point=(-74.008, 40.001),
    )


def _wp(lat: float, lon: float, alt: float = 50.0) -> Waypoint:
    return Waypoint(lat=lat, lon=lon, alt_m=alt)


def _plan_with(legs: list[MissionLeg]) -> MissionPlan:
    return MissionPlan(
        mission_id="m-1",
        area_id="area-test",
        status=MissionStatus.READY_FOR_APPROVAL,
        legs=legs,
    )


def test_valid_plan_passes(geo, profile):
    legs = [
        MissionLeg(
            leg_type=LegType.TRANSIT,
            geometry=[_wp(40.001, -74.008), _wp(40.002, -74.007)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
        MissionLeg(
            leg_type=LegType.RETURN_TO_BASE,
            geometry=[_wp(40.002, -74.007), _wp(40.001, -74.008, 20)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
    ]
    plan = _plan_with(legs)
    violations, report = validate_plan(plan, geo, profile)
    assert not violations, violations
    assert report.all_satisfied


def test_outside_geofence_flagged(geo, profile):
    # Waypoint south of the boundary
    legs = [
        MissionLeg(
            leg_type=LegType.TRANSIT,
            geometry=[_wp(40.001, -74.008), _wp(39.99, -74.005)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
        MissionLeg(
            leg_type=LegType.RETURN_TO_BASE,
            geometry=[_wp(39.99, -74.005), _wp(40.001, -74.008, 20)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
    ]
    plan = _plan_with(legs)
    violations, report = validate_plan(plan, geo, profile)
    assert not report.inside_geofence
    assert any("outside geofence" in v for v in violations)


def test_nfz_intersection_flagged(geo, profile):
    # Path goes diagonally across the NFZ corner
    legs = [
        MissionLeg(
            leg_type=LegType.TRANSIT,
            geometry=[_wp(40.0045, -74.0055), _wp(40.0065, -74.0035)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
        MissionLeg(
            leg_type=LegType.RETURN_TO_BASE,
            geometry=[_wp(40.0065, -74.0035), _wp(40.001, -74.008, 20)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
    ]
    plan = _plan_with(legs)
    violations, report = validate_plan(plan, geo, profile)
    assert not report.avoids_nfz
    assert any("no-fly-zone" in v for v in violations)


def test_above_ceiling_flagged(geo, profile):
    legs = [
        MissionLeg(
            leg_type=LegType.TRANSIT,
            geometry=[_wp(40.001, -74.008, alt=200.0), _wp(40.002, -74.007, alt=200.0)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
        MissionLeg(
            leg_type=LegType.RETURN_TO_BASE,
            geometry=[_wp(40.002, -74.007, 200.0), _wp(40.001, -74.008, 20)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
    ]
    plan = _plan_with(legs)
    violations, _ = validate_plan(plan, geo, profile)
    assert any("exceeds ceiling" in v for v in violations)


def test_below_min_agl_flagged(geo, profile):
    legs = [
        MissionLeg(
            leg_type=LegType.TRANSIT,
            geometry=[_wp(40.001, -74.008, alt=5.0), _wp(40.002, -74.007, alt=5.0)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
        MissionLeg(
            leg_type=LegType.RETURN_TO_BASE,
            geometry=[_wp(40.002, -74.007, 50.0), _wp(40.001, -74.008, 30)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
    ]
    plan = _plan_with(legs)
    violations, _ = validate_plan(plan, geo, profile)
    assert any("below min AGL" in v for v in violations)


def test_missing_rtb_flagged(geo, profile):
    legs = [
        MissionLeg(
            leg_type=LegType.TRANSIT,
            geometry=[_wp(40.001, -74.008), _wp(40.002, -74.007)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
    ]
    plan = _plan_with(legs)
    violations, report = validate_plan(plan, geo, profile)
    assert not report.ends_with_rtb
    assert any("RETURN_TO_BASE" in v for v in violations)


def test_battery_exhaustion_flagged(profile, geo):
    # Lots of long loiter at hover power
    legs = [
        MissionLeg(
            leg_type=LegType.LOITER,
            geometry=[_wp(40.005, -74.008, 50)],
            sensor_mode=SensorMode.EO,
            est_duration_s=3000.0,
            est_battery_pct=0,
        ),
        MissionLeg(
            leg_type=LegType.RETURN_TO_BASE,
            geometry=[_wp(40.005, -74.008, 50), _wp(40.001, -74.008, 20)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
    ]
    plan = _plan_with(legs)
    violations, report = validate_plan(plan, geo, profile)
    assert not report.battery_within_budget or not report.within_endurance
    assert violations


def test_search_pattern_coverage_gap(profile, geo):
    # Tracks placed 200 m apart at 30 m altitude — swath = 2*30*tan(30°) ≈ 34.6 m, gap
    legs = [
        MissionLeg(
            leg_type=LegType.SEARCH_PATTERN,
            geometry=[
                _wp(40.001, -74.008, 30),
                _wp(40.001, -74.006, 30),
                _wp(40.003, -74.006, 30),
                _wp(40.003, -74.008, 30),
            ],
            pattern_name=None,
            sensor_mode=SensorMode.EO,
            est_duration_s=0,
            est_battery_pct=0,
        ),
        MissionLeg(
            leg_type=LegType.RETURN_TO_BASE,
            geometry=[_wp(40.003, -74.008, 30), _wp(40.001, -74.008, 20)],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
    ]
    plan = _plan_with(legs)
    violations, report = validate_plan(plan, geo, profile)
    assert not report.sensor_coverage_adequate
    assert any("coverage gap" in v for v in violations)
