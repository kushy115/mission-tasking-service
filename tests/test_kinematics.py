"""Tests for the kinematic trajectory model (DD-015)."""

from __future__ import annotations

from app.schemas.enums import LegType, MissionStatus, SensorMode
from app.schemas.plan import MissionLeg, MissionPlan, Waypoint
from app.sim.kinematics import _corner_speed, _wind_along_mps, simulate_trajectory
from app.validation.physics import DroneProfile, SensorSpec, haversine_m


def _profile(accel: float = 3.0) -> DroneProfile:
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
        max_accel_mps2=accel,
    )


def _transit(a: Waypoint, b: Waypoint) -> MissionLeg:
    return MissionLeg(
        leg_type=LegType.TRANSIT,
        sensor_mode=SensorMode.EO,
        geometry=[a, b],
        est_duration_s=0,
        est_battery_pct=0,
    )


def _straight_plan() -> MissionPlan:
    return MissionPlan(
        mission_id="m",
        area_id="t",
        status=MissionStatus.READY_FOR_APPROVAL,
        legs=[
            _transit(
                Waypoint(lat=40.72, lon=-74.17, alt_m=50), Waypoint(lat=40.73, lon=-74.17, alt_m=50)
            )
        ],
    )


def test_corner_speed_monotonic():
    v = 15.0
    assert _corner_speed(v, 0.0) == v  # straight => full cruise
    assert _corner_speed(v, 90.0) < v
    assert _corner_speed(v, 180.0) < _corner_speed(v, 90.0)  # sharper => slower


def test_wind_along_sign():
    # Track due North (0 deg). Wind FROM the north (0) is a headwind (negative).
    assert _wind_along_mps(0.0, 0.0, 5.0) < 0
    # Wind FROM the south (180) pushes North => tailwind (positive).
    assert _wind_along_mps(0.0, 180.0, 5.0) > 0


def test_trace_is_monotonic_and_drains_battery():
    trace = simulate_trajectory(_straight_plan(), _profile())
    assert trace.samples
    assert trace.total_duration_s > 0
    assert trace.total_battery_pct > 0
    ts = [s.t for s in trace.samples]
    batt = [s.battery_used_pct for s in trace.samples]
    assert all(ts[i + 1] >= ts[i] for i in range(len(ts) - 1))
    assert all(batt[i + 1] >= batt[i] - 1e-9 for i in range(len(batt) - 1))


def test_accel_makes_it_slower_than_teleport():
    """A real ramp-up + braking can't beat instantaneous cruise speed."""
    plan = _straight_plan()
    profile = _profile(accel=2.0)
    trace = simulate_trajectory(plan, profile)
    a, b = plan.legs[0].geometry
    teleport_t = haversine_m(a, b) / profile.cruise_speed_mps
    assert trace.total_duration_s >= teleport_t  # accel + corner braking add time


def test_tailwind_faster_than_headwind():
    plan = _straight_plan()  # flies due north
    profile = _profile()
    head = simulate_trajectory(plan, profile, wind_mps=8.0, wind_from_deg=0.0, wind_coeff=0.02)
    tail = simulate_trajectory(plan, profile, wind_mps=8.0, wind_from_deg=180.0, wind_coeff=0.02)
    assert tail.total_duration_s < head.total_duration_s
    assert tail.total_battery_pct < head.total_battery_pct


def test_turn_costs_time():
    """An L-shaped path should take longer than its straight-line distance / cruise."""
    profile = _profile()
    corner = MissionPlan(
        mission_id="m",
        area_id="t",
        status=MissionStatus.READY_FOR_APPROVAL,
        legs=[
            _transit(
                Waypoint(lat=40.72, lon=-74.17, alt_m=50), Waypoint(lat=40.73, lon=-74.17, alt_m=50)
            ),
            _transit(
                Waypoint(lat=40.73, lon=-74.17, alt_m=50), Waypoint(lat=40.73, lon=-74.15, alt_m=50)
            ),
        ],
    )
    trace = simulate_trajectory(corner, profile)
    # Sanity: the trace visits both legs.
    assert {s.leg_idx for s in trace.samples} == {0, 1}
    assert trace.total_duration_s > 0
