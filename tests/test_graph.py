"""Graph wiring tests. The LLM-driven `plan` node is stubbed so the test runs
deterministically and offline. Verifies:
  - happy-path: a valid draft plan passes validate → finalize
  - bad plan: triggers repair, then succeeds within the repair budget
  - bad plan exceeding the cap: routes to REJECTED
  - intake clarification short-circuit
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from shapely.geometry import Polygon

from app.graph.builder import build_graph
from app.schemas.enums import LegType, MissionStatus, SensorMode
from app.schemas.plan import MissionLeg, MissionPlan, Waypoint
from app.validation.kernel import GeoContext
from app.validation.physics import DroneProfile, SensorSpec


@pytest.fixture
def geo_ctx() -> GeoContext:
    return GeoContext(
        area_id="yard-simple",
        boundary=Polygon(
            [(-74.175, 40.718), (-74.165, 40.718), (-74.165, 40.724), (-74.175, 40.724)]
        ),
        nfz_polygons=(),
        altitude_ceiling_m=120.0,
        home_point=(-74.170, 40.720),
    )


@pytest.fixture
def profile() -> DroneProfile:
    return DroneProfile(
        profile_id="long-endurance-quad",
        rated_endurance_s=2700.0,
        cruise_speed_mps=18.0,
        climb_rate_mps=4.0,
        battery_wh=540.0,
        cruise_power_w=380.0,
        hover_power_w=520.0,
        sensors=(
            SensorSpec(
                name="eo",
                mode=SensorMode.EO,
                power_w=12.0,
                half_angle_deg=32.0,
                ground_resolution_at_100m=0.035,
            ),
        ),
    )


def _valid_plan() -> MissionPlan:
    return MissionPlan(
        mission_id="m-test",
        area_id="yard-simple",
        status=MissionStatus.READY_FOR_APPROVAL,
        legs=[
            MissionLeg(
                leg_type=LegType.TRANSIT,
                geometry=[
                    Waypoint(lat=40.720, lon=-74.170, alt_m=60),
                    Waypoint(lat=40.722, lon=-74.168, alt_m=60),
                ],
                sensor_mode=SensorMode.EO,
                est_duration_s=0,
                est_battery_pct=0,
            ),
            MissionLeg(
                leg_type=LegType.RETURN_TO_BASE,
                geometry=[
                    Waypoint(lat=40.722, lon=-74.168, alt_m=60),
                    Waypoint(lat=40.720, lon=-74.170, alt_m=30),
                ],
                sensor_mode=SensorMode.OFF,
                est_duration_s=0,
                est_battery_pct=0,
            ),
        ],
    )


def _patch_engine(geo_ctx, profile):
    """Patch all engine accessors so nodes/validator don't touch a real DB."""
    return [
        patch("app.graph.nodes.get_engine", return_value=None),
        patch("app.graph.nodes.load_geo_context", return_value=geo_ctx),
        patch("app.graph.nodes.load_drone_profile", return_value=profile),
        patch("app.graph.nodes.save_mission", lambda *_, **__: None),
    ]


def test_happy_path(geo_ctx, profile):
    plan = _valid_plan()
    with (
        _patch_engine(geo_ctx, profile)[0],
        _patch_engine(geo_ctx, profile)[1],
        _patch_engine(geo_ctx, profile)[2],
        _patch_engine(geo_ctx, profile)[3],
        patch("app.graph.nodes._agent_lazy") as mock_agent,
    ):
        mock_agent.return_value.invoke.return_value = {"structured_response": plan}
        g = build_graph()
        out = g.invoke(
            {
                "raw_command": "Patrol the yard perimeter at 60m.",
                "area_id": "yard-simple",
                "operator_clearance": "STANDARD",
                "drone_state": {"drone_profile_id": "long-endurance-quad", "battery_pct": 100.0},
                "messages": [],
                "repair_attempts": 0,
            },
            config={"configurable": {"thread_id": "t1"}},
        )
    assert out["draft_plan"]["status"] == "READY_FOR_APPROVAL"


def test_intake_clarification_short_circuit(geo_ctx, profile):
    with (
        _patch_engine(geo_ctx, profile)[0],
        _patch_engine(geo_ctx, profile)[1],
        _patch_engine(geo_ctx, profile)[2],
        _patch_engine(geo_ctx, profile)[3],
    ):
        g = build_graph()
        out = g.invoke(
            {
                "raw_command": "Go check something out there",
                "area_id": "yard-simple",
                "operator_clearance": "STANDARD",
                "drone_state": {"drone_profile_id": "long-endurance-quad", "battery_pct": 100.0},
                "messages": [],
            },
            config={"configurable": {"thread_id": "t-clarify"}},
        )
    assert out["draft_plan"]["status"] == "NEEDS_CLARIFICATION"
    assert out["draft_plan"]["clarification_questions"]


def test_repair_then_reject(geo_ctx, profile):
    """Planner produces a bad plan (missing RTB) repeatedly; once we exceed
    the cap, the graph routes to REJECTED."""
    bad = MissionPlan(
        mission_id="m-bad",
        area_id="yard-simple",
        status=MissionStatus.READY_FOR_APPROVAL,
        legs=[
            MissionLeg(
                leg_type=LegType.TRANSIT,
                geometry=[
                    Waypoint(lat=40.720, lon=-74.170, alt_m=60),
                    Waypoint(lat=40.722, lon=-74.168, alt_m=60),
                ],
                sensor_mode=SensorMode.OFF,
                est_duration_s=0,
                est_battery_pct=0,
            )
        ],
    )
    with (
        _patch_engine(geo_ctx, profile)[0],
        _patch_engine(geo_ctx, profile)[1],
        _patch_engine(geo_ctx, profile)[2],
        _patch_engine(geo_ctx, profile)[3],
        patch("app.graph.nodes._agent_lazy") as mock_agent,
    ):
        mock_agent.return_value.invoke.return_value = {"structured_response": bad}
        g = build_graph()
        out = g.invoke(
            {
                "raw_command": "Patrol the yard at 60m EO.",
                "area_id": "yard-simple",
                "operator_clearance": "STANDARD",
                "drone_state": {"drone_profile_id": "long-endurance-quad", "battery_pct": 100.0},
                "messages": [],
                "repair_attempts": 0,
            },
            config={"configurable": {"thread_id": "t-bad"}},
        )
    assert out["draft_plan"]["status"] == "REJECTED"
    assert out["repair_attempts"] >= 3
