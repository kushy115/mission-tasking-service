"""Tests for the telemetry stream simulator."""

from __future__ import annotations

import asyncio

import pytest
from shapely.geometry import Polygon

from app.schemas.enums import LegType, MissionStatus, SensorMode
from app.schemas.plan import MissionLeg, MissionPlan, Waypoint
from app.sim.stream import telemetry_stream
from app.validation.kernel import GeoContext
from app.validation.physics import DroneProfile, SensorSpec


@pytest.fixture
def profile() -> DroneProfile:
    return DroneProfile(
        profile_id="t",
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
                power_w=10,
                half_angle_deg=30,
                ground_resolution_at_100m=0.05,
            ),
        ),
    )


@pytest.fixture
def geo() -> GeoContext:
    return GeoContext(
        area_id="t",
        boundary=Polygon([(-74.18, 40.71), (-74.16, 40.71), (-74.16, 40.73), (-74.18, 40.73)]),
        nfz_polygons=(),
        altitude_ceiling_m=120.0,
        home_point=(-74.17, 40.72),
    )


def _plan() -> MissionPlan:
    return MissionPlan(
        mission_id="m",
        area_id="t",
        status=MissionStatus.READY_FOR_APPROVAL,
        legs=[
            MissionLeg(
                leg_type=LegType.TRANSIT,
                sensor_mode=SensorMode.EO,
                geometry=[
                    Waypoint(lat=40.72, lon=-74.17, alt_m=40),
                    Waypoint(lat=40.725, lon=-74.168, alt_m=60),
                ],
                est_duration_s=10,
                est_battery_pct=2,
            ),
            MissionLeg(
                leg_type=LegType.RETURN_TO_BASE,
                sensor_mode=SensorMode.OFF,
                geometry=[
                    Waypoint(lat=40.725, lon=-74.168, alt_m=60),
                    Waypoint(lat=40.72, lon=-74.17, alt_m=30),
                ],
                est_duration_s=10,
                est_battery_pct=2,
            ),
        ],
    )


def test_stream_yields_frames_and_terminates(profile, geo):
    plan = _plan()

    # speed=1000 makes the sleeps near-instant so the test runs fast.
    async def collect():
        frames = []
        async for f in telemetry_stream(plan, profile, geo, speed=1000.0):
            frames.append(f)
        return frames

    frames = asyncio.run(collect())
    assert len(frames) >= 2
    # First frame is mid-flight, last is terminal
    assert frames[-1].status in ("done", "aborted")
    # Battery should monotonically decrease
    batts = [f.battery_pct for f in frames]
    for i in range(len(batts) - 1):
        assert batts[i + 1] <= batts[i] + 1e-6
    # Elapsed time should be monotonic
    ts = [f.t for f in frames]
    for i in range(len(ts) - 1):
        assert ts[i + 1] >= ts[i] - 1e-6


def test_stream_aborts_on_geofence_breach(profile):
    """Tiny boundary nowhere near the plan → first interpolated frame exits it."""
    plan = _plan()
    tiny_box = GeoContext(
        area_id="t",
        boundary=Polygon([(0, 0), (0.0001, 0), (0.0001, 0.0001), (0, 0.0001)]),
        nfz_polygons=(),
        altitude_ceiling_m=120.0,
        home_point=(0, 0),
    )

    async def collect():
        return [f async for f in telemetry_stream(plan, profile, tiny_box, speed=1000.0)]

    frames = asyncio.run(collect())
    assert frames[-1].status == "aborted"
    assert "geofence" in frames[-1].detail.lower()


def test_empty_plan_yields_single_done_frame(profile, geo):
    empty = MissionPlan(mission_id="m", area_id="t", status=MissionStatus.READY_FOR_APPROVAL)

    async def collect():
        return [f async for f in telemetry_stream(empty, profile, geo, speed=1000.0)]

    frames = asyncio.run(collect())
    assert len(frames) == 1
    assert frames[0].status == "done"
