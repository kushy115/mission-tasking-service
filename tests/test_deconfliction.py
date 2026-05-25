"""Tests for airspace deconfliction (no DB needed; we exercise the geometry math)."""

from __future__ import annotations

import pytest

from app.schemas.enums import LegType, MissionStatus, SensorMode
from app.schemas.plan import MissionLeg, MissionPlan, Waypoint
from app.validation.deconfliction import (
    LATERAL_SEP_M,
    VERTICAL_SEP_M,
    _bands_overlap,
    _plan_altitude_band,
    _plan_to_geometry,
)


def _plan(legs):
    return MissionPlan(
        mission_id="m", area_id="a", status=MissionStatus.READY_FOR_APPROVAL, legs=legs
    )


def _leg(coords, alt=50.0, lt=LegType.TRANSIT):
    return MissionLeg(
        leg_type=lt,
        sensor_mode=SensorMode.OFF,
        geometry=[Waypoint(lat=la, lon=lo, alt_m=alt) for la, lo in coords],
        est_duration_s=10.0,
        est_battery_pct=2.0,
    )


def test_plan_altitude_band_picks_min_and_max():
    plan = _plan([
        _leg([(40.7, -74.1), (40.71, -74.11)], alt=30),
        _leg([(40.71, -74.11), (40.72, -74.12)], alt=80),
    ])
    band = _plan_altitude_band(plan)
    assert band == (30.0, 80.0)


def test_plan_to_geometry_handles_single_leg():
    plan = _plan([_leg([(40.7, -74.1), (40.71, -74.11)])])
    geom = _plan_to_geometry(plan)
    assert geom is not None
    assert geom.geom_type == "LineString"


def test_plan_to_geometry_handles_multi_leg():
    plan = _plan([
        _leg([(40.7, -74.1), (40.71, -74.11)]),
        _leg([(40.71, -74.11), (40.72, -74.12)]),
    ])
    geom = _plan_to_geometry(plan)
    assert geom.geom_type == "MultiLineString"


def test_plan_to_geometry_drops_single_point_legs():
    plan = _plan([_leg([(40.7, -74.1)])])  # only one point — no LineString
    assert _plan_to_geometry(plan) is None


def test_band_overlap_true_when_close():
    # Bands abut at 60m, separation is 30m → overlap.
    assert _bands_overlap((30.0, 60.0), (60.0, 90.0))


def test_band_overlap_false_when_far_apart():
    # 30m apart, sep requirement also 30m → comparison: lo = max(60,100)-15=85, hi = min(80,120)+15=95
    # 85 <= 95 -> True. Need bigger gap. Let's check 30-80 vs 200-300:
    assert not _bands_overlap((30.0, 80.0), (200.0, 300.0))


def test_constants_present():
    assert LATERAL_SEP_M > 0
    assert VERTICAL_SEP_M > 0
