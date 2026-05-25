"""Weather provider + kernel integration tests."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from app.schemas.enums import LegType, SensorMode
from app.schemas.plan import MissionLeg, MissionPlan, Waypoint
from app.validation.kernel import GeoContext, validate_plan
from app.validation.physics import DroneProfile, SensorSpec, estimate_mission
from app.weather import WeatherObservation, get_weather_for_area
from app.weather.provider import SyntheticWeather, clear_cache


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
        sensors=(SensorSpec(name="eo", mode=SensorMode.EO, power_w=10,
                            half_angle_deg=30, ground_resolution_at_100m=0.05),),
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


def _simple_plan() -> MissionPlan:
    return MissionPlan(
        mission_id="m",
        area_id="t",
        status="READY_FOR_APPROVAL",
        legs=[
            MissionLeg(
                leg_type=LegType.TRANSIT, sensor_mode=SensorMode.EO,
                geometry=[
                    Waypoint(lat=40.72, lon=-74.17, alt_m=40),
                    Waypoint(lat=40.725, lon=-74.165, alt_m=60),
                ],
                est_duration_s=10, est_battery_pct=2,
            ),
            MissionLeg(
                leg_type=LegType.RETURN_TO_BASE, sensor_mode=SensorMode.OFF,
                geometry=[
                    Waypoint(lat=40.725, lon=-74.165, alt_m=60),
                    Waypoint(lat=40.72, lon=-74.17, alt_m=30),
                ],
                est_duration_s=10, est_battery_pct=2,
            ),
        ],
    )


def test_synthetic_weather_is_deterministic():
    a = SyntheticWeather(seed=42).observe(40.0, -74.0)
    b = SyntheticWeather(seed=42).observe(40.0, -74.0)
    assert a.wind_mps == b.wind_mps
    assert a.source == "synthetic"


def test_provider_cache_dedupes(monkeypatch):
    clear_cache()
    obs1 = get_weather_for_area(40.0, -74.0)
    obs2 = get_weather_for_area(40.0, -74.0)
    assert obs1 == obs2


def test_kernel_rejects_high_wind(profile, geo):
    plan = _simple_plan()
    high_wind = WeatherObservation(
        wind_mps=20.0, gust_mps=22.0, wind_dir_deg=180,
        visibility_m=10000, precipitation_mmh=0, temperature_c=15, source="t",
    )
    violations, report = validate_plan(plan, geo, profile, weather=high_wind)
    assert not report.weather_acceptable
    assert any("wind" in v for v in violations)


def test_kernel_rejects_low_visibility_with_eo(profile, geo):
    plan = _simple_plan()
    foggy = WeatherObservation(
        wind_mps=2, gust_mps=3, wind_dir_deg=0,
        visibility_m=800, precipitation_mmh=0, temperature_c=12, source="t",
    )
    violations, report = validate_plan(plan, geo, profile, weather=foggy)
    assert not report.weather_acceptable
    assert any("visibility" in v for v in violations)


def test_kernel_passes_mild_weather(profile, geo):
    plan = _simple_plan()
    mild = WeatherObservation(
        wind_mps=2, gust_mps=4, wind_dir_deg=90,
        visibility_m=10000, precipitation_mmh=0, temperature_c=18, source="t",
    )
    violations, report = validate_plan(plan, geo, profile, weather=mild)
    assert report.weather_acceptable
    # mild weather shouldn't add weather violations
    assert not any("wind" in v or "visibility" in v or "precipitation" in v for v in violations)


def test_wind_increases_battery_usage(profile):
    plan = _simple_plan()
    base = estimate_mission(plan.legs, profile)
    windy = estimate_mission(plan.legs, profile, wind_mps=10.0, wind_coeff=0.05)
    assert windy.total_battery_pct > base.total_battery_pct


def test_no_weather_passes_trivially(profile, geo):
    plan = _simple_plan()
    _, report = validate_plan(plan, geo, profile)  # no weather arg
    assert report.weather_acceptable
    assert "not evaluated" in report.weather_acceptable_detail
