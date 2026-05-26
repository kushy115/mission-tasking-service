"""LangChain @tool suite the planner agent uses to ground its reasoning.

Tools are deterministic where possible (critical for repeatable evals). They
read from the PostGIS store, the drone profile, and the deterministic physics
model — they do NOT call LLMs and have no side effects beyond DB reads.

Each docstring is read by the LLM as the tool spec, so keep them precise.
"""

from __future__ import annotations

import hashlib
import random
from typing import Annotated

from langchain_core.tools import tool
from shapely.geometry import LineString
from sqlalchemy.engine import Engine

from app.config import get_settings
from app.geo import patterns as patterns_mod
from app.geo.store import get_engine, load_drone_profile, load_geo_context
from app.schemas.enums import SensorMode
from app.schemas.plan import MissionLeg
from app.validation.physics import (
    estimate_mission,
    sensor_ground_resolution_m_per_px,
    sensor_swath_m,
)

# Engine resolution is lazy so unit tests can monkeypatch.
_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


@tool
def get_geofence(area_id: Annotated[str, "Operating area ID"]) -> dict:
    """Return the geofence for an operating area: outer boundary polygon, all
    no-fly-zone polygons, altitude ceiling, and home/base point.

    Use this BEFORE planning any waypoints so you know what bounds and obstacles
    you are working with. All coordinates are WGS84 lon/lat.
    """
    geo = load_geo_context(_get_engine(), area_id)
    return {
        "area_id": geo.area_id,
        "boundary_lonlat": list(geo.boundary.exterior.coords),
        "nfzs": [list(p.exterior.coords) for p in geo.nfz_polygons],
        "altitude_ceiling_m": geo.altitude_ceiling_m,
        "home_lon": geo.home_point[0],
        "home_lat": geo.home_point[1],
    }


@tool
def check_path_clear(
    area_id: Annotated[str, "Operating area ID"],
    polyline_lonlat: Annotated[
        list[list[float]],
        "List of [lon, lat] pairs describing the proposed flight path.",
    ],
) -> dict:
    """Check whether a proposed flight path intersects any no-fly-zone in the
    operating area. Returns clear/blocked and, if blocked, which NFZ indices.
    Use after generating geometry and BEFORE finalizing a plan.
    """
    if len(polyline_lonlat) < 2:
        return {"clear": True, "intersections": []}
    geo = load_geo_context(_get_engine(), area_id)
    line = LineString(polyline_lonlat)
    hits = []
    for i, nfz in enumerate(geo.nfz_polygons):
        poly = nfz if nfz.is_valid else nfz.buffer(0)
        if line.intersects(poly):
            hits.append(i)
    return {"clear": not hits, "intersections": hits}


@tool
def estimate_battery(
    drone_profile_id: Annotated[str, "Drone profile ID"],
    legs: Annotated[
        list[dict],
        "List of leg dicts matching MissionLeg schema (leg_type, geometry, sensor_mode, ...).",
    ],
) -> dict:
    """Estimate total flight time and battery consumption for a candidate set
    of legs using the deterministic physics model. Returns total_duration_s,
    total_battery_pct, and per-leg breakdown.
    Use this BEFORE finalizing to confirm the plan fits the battery budget.
    """
    profile = load_drone_profile(_get_engine(), drone_profile_id)
    parsed_legs = [MissionLeg.model_validate(d) for d in legs]
    energy = estimate_mission(parsed_legs, profile)
    return {
        "total_duration_s": energy.total_duration_s,
        "total_battery_pct": energy.total_battery_pct,
        "rated_endurance_s": profile.rated_endurance_s,
        "per_leg": [
            {
                "duration_s": e.duration_s,
                "battery_pct": e.battery_pct,
                "distance_m": e.distance_m,
            }
            for e in energy.legs
        ],
    }


@tool
def get_sensor_coverage(
    drone_profile_id: Annotated[str, "Drone profile ID"],
    sensor_mode: Annotated[str, "EO or IR"],
    altitude_m: Annotated[float, "AGL altitude in meters"],
) -> dict:
    """Return ground swath width (m) and ground resolution (m/px) for the named
    sensor at the given altitude. Use this to pick a search-pattern track
    spacing such that swath >= spacing (no coverage gap).
    """
    profile = load_drone_profile(_get_engine(), drone_profile_id)
    sensor = profile.sensor(SensorMode(sensor_mode))
    if sensor is None:
        return {"error": f"sensor {sensor_mode} not on drone {drone_profile_id}"}
    return {
        "swath_m": sensor_swath_m(sensor, altitude_m),
        "ground_resolution_m_per_px": sensor_ground_resolution_m_per_px(sensor, altitude_m),
    }


@tool
def lookup_search_pattern(
    pattern_name: Annotated[str, "lawnmower | expanding_square | sector"],
    center_lat: float,
    center_lon: float,
    altitude_m: float,
    width_m: Annotated[float, "Lawnmower: rectangle width. Other patterns: ignored."] = 0.0,
    height_m: Annotated[float, "Lawnmower: rectangle height. Other patterns: ignored."] = 0.0,
    spacing_m: Annotated[float, "Lawnmower track spacing in meters."] = 0.0,
    radius_m: Annotated[float, "Sector / expanding_square: radius or initial leg in meters."] = 0.0,
    legs_count: Annotated[int, "Expanding-square legs."] = 8,
) -> dict:
    """Generate the waypoint geometry for a named search pattern. Returns a
    list of {lat, lon, alt_m}. Use this so you don't have to invent geometry
    by hand — the generator is guaranteed parametric and consistent.
    """
    if pattern_name == "lawnmower":
        wps = patterns_mod.lawnmower(
            center_lat, center_lon, width_m, height_m, spacing_m, altitude_m
        )
    elif pattern_name == "expanding_square":
        wps = patterns_mod.expanding_square(
            center_lat, center_lon, radius_m or 50.0, 25.0, legs_count, altitude_m
        )
    elif pattern_name == "sector":
        wps = patterns_mod.sector(center_lat, center_lon, radius_m or 100.0, legs_count, altitude_m)
    else:
        return {"error": f"unknown pattern: {pattern_name}"}
    return {"pattern_name": pattern_name, "waypoints": [w.model_dump() for w in wps]}


@tool
def get_weather(area_id: Annotated[str, "Operating area ID"]) -> dict:
    """Return current weather snapshot for the area. Deterministic synthetic
    provider keyed by (area_id, seed) — same area always returns the same value
    across a run, so evals are reproducible.
    """
    settings = get_settings()
    if settings.weather_provider != "synthetic":
        # Real-provider integration intentionally left as a feature-flag stub.
        return {"error": "non-synthetic weather provider not configured"}
    seed_str = f"{area_id}:{settings.weather_seed}"
    seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)
    return {
        "wind_mps": round(rng.uniform(0.0, 10.0), 1),
        "wind_dir_deg": rng.randint(0, 359),
        "visibility_m": rng.choice([2000, 5000, 10000, 10000]),
        "precip": rng.choice(["none", "none", "none", "light_rain"]),
        "temp_c": round(rng.uniform(-5.0, 30.0), 1),
    }


PLANNING_TOOLS = [
    get_geofence,
    check_path_clear,
    estimate_battery,
    get_sensor_coverage,
    lookup_search_pattern,
    get_weather,
]
