"""Tests for plan-export formats. Pure-string output, deterministic."""

from __future__ import annotations

import pytest

from app.export import EXPORT_FORMATS, render_export
from app.schemas.enums import LegType, MissionStatus, SensorMode
from app.schemas.plan import MissionLeg, MissionPlan, Waypoint


@pytest.fixture
def plan() -> MissionPlan:
    return MissionPlan(
        mission_id="abc-123",
        area_id="yard-simple",
        status=MissionStatus.READY_FOR_APPROVAL,
        legs=[
            MissionLeg(
                leg_type=LegType.TRANSIT,
                geometry=[
                    Waypoint(lat=40.720, lon=-74.170, alt_m=30.0),
                    Waypoint(lat=40.722, lon=-74.168, alt_m=60.0),
                ],
                sensor_mode=SensorMode.EO,
                est_duration_s=45.0,
                est_battery_pct=5.0,
            ),
            MissionLeg(
                leg_type=LegType.RETURN_TO_BASE,
                geometry=[
                    Waypoint(lat=40.722, lon=-74.168, alt_m=60.0),
                    Waypoint(lat=40.720, lon=-74.170, alt_m=30.0),
                ],
                sensor_mode=SensorMode.OFF,
                est_duration_s=45.0,
                est_battery_pct=5.0,
            ),
        ],
    )


def test_kml_contains_coordinates_and_relative_alt(plan: MissionPlan):
    body, spec = render_export(plan, "kml")
    assert spec.extension == "kml"
    assert spec.media_type == "application/vnd.google-earth.kml+xml"
    assert "<kml" in body and "</kml>" in body
    assert "relativeToGround" in body  # altitudes are AGL
    assert "-74.170000,40.720000,30.0" in body
    # one Placemark per waypoint (4) + one LineString per leg (2) = 6 placemarks
    assert body.count("<Placemark>") == 6


def test_gpx_contains_routes_and_waypoints(plan: MissionPlan):
    body, spec = render_export(plan, "gpx")
    assert spec.extension == "gpx"
    assert '<gpx version="1.1"' in body
    # 4 wpt and 2 rte
    assert body.count("<wpt") == 4
    assert body.count("<rte>") == 2
    assert "<ele>30.0</ele>" in body


def test_dji_csv_has_header_and_one_row_per_waypoint(plan: MissionPlan):
    body, spec = render_export(plan, "dji")
    assert spec.extension == "csv"
    lines = [ln for ln in body.strip().splitlines() if ln and not ln.startswith("#")]
    # 1 header + 4 waypoint rows
    assert len(lines) == 5
    assert lines[0].startswith("latitude,longitude,altitude")


def test_unknown_format_raises():
    plan = MissionPlan(mission_id="m", area_id="a", status=MissionStatus.READY_FOR_APPROVAL)
    with pytest.raises(KeyError):
        render_export(plan, "bogus")


def test_all_formats_registered():
    assert set(EXPORT_FORMATS.keys()) == {"kml", "gpx", "dji"}
