"""Tests for the deterministic physics model. Safety core — keep thorough."""

from __future__ import annotations

import math

import pytest

from app.schemas.enums import LegType, SensorMode
from app.schemas.plan import MissionLeg, Waypoint
from app.validation.physics import (
    DroneProfile,
    SensorSpec,
    estimate_leg,
    estimate_mission,
    haversine_m,
    sensor_ground_resolution_m_per_px,
    sensor_swath_m,
)


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
            SensorSpec(
                name="ir",
                mode=SensorMode.IR,
                power_w=15.0,
                half_angle_deg=20.0,
                ground_resolution_at_100m=0.20,
            ),
        ),
    )


def test_haversine_zero():
    a = Waypoint(lat=40.0, lon=-74.0, alt_m=50)
    assert haversine_m(a, a) == pytest.approx(0.0)


def test_haversine_one_degree_lat():
    a = Waypoint(lat=40.0, lon=-74.0, alt_m=50)
    b = Waypoint(lat=41.0, lon=-74.0, alt_m=50)
    # 1 degree latitude is ~111_195 m on a 6_371_000 m sphere
    assert haversine_m(a, b) == pytest.approx(111_195.0, rel=1e-3)


def test_estimate_transit_leg(profile):
    leg = MissionLeg(
        leg_type=LegType.TRANSIT,
        geometry=[
            Waypoint(lat=40.0, lon=-74.0, alt_m=50),
            Waypoint(lat=40.0, lon=-74.0, alt_m=100),
        ],
        sensor_mode=SensorMode.OFF,
        est_duration_s=0,
        est_battery_pct=0,
    )
    e = estimate_leg(leg, profile)
    # 50 m climb at 3 m/s = 16.67 s
    assert e.vertical_time_s == pytest.approx(50 / 3, rel=1e-3)
    assert e.horizontal_time_s == pytest.approx(0.0)
    assert e.energy_wh > 0
    assert e.battery_pct > 0


def test_estimate_mission_aggregates(profile):
    legs = [
        MissionLeg(
            leg_type=LegType.TRANSIT,
            geometry=[
                Waypoint(lat=40.0, lon=-74.0, alt_m=50),
                Waypoint(lat=40.001, lon=-74.0, alt_m=50),
            ],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
        MissionLeg(
            leg_type=LegType.RETURN_TO_BASE,
            geometry=[
                Waypoint(lat=40.001, lon=-74.0, alt_m=50),
                Waypoint(lat=40.0, lon=-74.0, alt_m=0),
            ],
            sensor_mode=SensorMode.OFF,
            est_duration_s=0,
            est_battery_pct=0,
        ),
    ]
    m = estimate_mission(legs, profile)
    assert m.total_duration_s > 0
    assert m.total_battery_pct > 0
    assert math.isclose(sum(l.duration_s for l in m.legs), m.total_duration_s)


def test_sensor_swath_increases_with_altitude(profile):
    eo = profile.sensor(SensorMode.EO)
    assert eo is not None
    s100 = sensor_swath_m(eo, 100)
    s200 = sensor_swath_m(eo, 200)
    assert s200 == pytest.approx(2 * s100)


def test_sensor_resolution_scales_with_altitude(profile):
    eo = profile.sensor(SensorMode.EO)
    assert eo is not None
    assert sensor_ground_resolution_m_per_px(eo, 100) == pytest.approx(0.05)
    assert sensor_ground_resolution_m_per_px(eo, 200) == pytest.approx(0.10)


def test_loiter_uses_hover_power(profile):
    leg = MissionLeg(
        leg_type=LegType.LOITER,
        geometry=[Waypoint(lat=40.0, lon=-74.0, alt_m=80)],
        sensor_mode=SensorMode.IR,
        est_duration_s=120.0,
        est_battery_pct=0,
    )
    e = estimate_leg(leg, profile)
    # hover + IR sensor power over 120 s
    expected_wh = (profile.hover_power_w + 15.0) * 120 / 3600
    assert e.energy_wh == pytest.approx(expected_wh, rel=1e-6)
