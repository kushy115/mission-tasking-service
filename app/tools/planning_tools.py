"""LangChain `@tool` suite for the tool-calling planning agent (DD-016).

These tools let the planning agent GROUND its reasoning in the real operating
area, the selected drone's physics model, and validated pattern geometry —
instead of hallucinating coordinates or feasibility. The agent calls them inside
its own reasoning loop (e.g. "is this transit clear? → check_path_clear", "how
wide is the EO swath at 90 m? → get_sensor_coverage") and self-corrects before
emitting the final plan.

`build_planning_tools(geo, profile)` builds the suite per-compile, closing over
the area's geo context (loaded from PostGIS) and the chosen `DroneProfile`. That
way the model only supplies the *meaningful* arguments — a polyline, an altitude,
a pattern name — and never has to echo the bulky boundary / NFZ polygons. Each
tool returns a compact JSON string the model can read.

Pure and deterministic; the only I/O is the geo read already done upstream.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool
from shapely.geometry import LineString, Polygon

from app.geo.patterns import expanding_square, lawnmower_fit_to_boundary, sector
from app.schemas.enums import LegType, SensorMode
from app.schemas.plan import MissionLeg, Waypoint
from app.validation.physics import (
    DroneProfile,
    estimate_mission,
    sensor_ground_resolution_m_per_px,
    sensor_swath_m,
)


def build_planning_tools(geo: dict[str, Any], profile: DroneProfile) -> list[BaseTool]:
    """Build the planning-tool suite bound to one compile's area + drone."""
    boundary_coords = geo.get("boundary_lonlat") or []
    boundary = Polygon(boundary_coords) if len(boundary_coords) >= 3 else None
    nfz_polys = [Polygon(c) for c in (geo.get("nfzs") or []) if c and len(c) >= 3]
    ceiling = float(geo.get("altitude_ceiling_m") or 120.0)

    def _centroid_lat_lon() -> tuple[float, float]:
        if boundary is not None:
            c = boundary.centroid
            return (c.y, c.x)
        return (float(geo.get("home_lat") or 0.0), float(geo.get("home_lon") or 0.0))

    @tool
    def get_geofence() -> str:
        """Return the operating-area geofence the plan must stay inside: the
        boundary bounding box (lon/lat), the number and bounding boxes of no-fly
        zones, the altitude ceiling in meters, and the registered home bases.
        Call this first to understand the airspace you are planning in."""
        return json.dumps(
            {
                "boundary_bounds_lonlat": list(boundary.bounds) if boundary else None,
                "num_no_fly_zones": len(nfz_polys),
                "no_fly_zone_bounds_lonlat": [list(p.bounds) for p in nfz_polys],
                "altitude_ceiling_m": ceiling,
                "home_lon_lat": [geo.get("home_lon"), geo.get("home_lat")],
                "registered_home_bases_lonlat": geo.get("home_bases") or [],
            }
        )

    @tool
    def check_path_clear(polyline_lonlat: list[list[float]]) -> str:
        """Check whether a proposed flight path is legal BEFORE committing to it.
        `polyline_lonlat` is an ordered list of [lon, lat] points. Returns whether
        the path stays inside the geofence and clear of every no-fly zone; if not,
        it names the offending zone and its bounds so you can re-route around it."""
        if not polyline_lonlat or len(polyline_lonlat) < 2:
            return json.dumps({"clear": False, "reason": "need at least 2 points"})
        try:
            line = LineString([(float(x), float(y)) for x, y in polyline_lonlat])
        except Exception as e:  # noqa: BLE001
            return json.dumps({"clear": False, "reason": f"bad polyline: {e}"})
        problems: list[str] = []
        if boundary is not None and not boundary.buffer(0).covers(line):
            problems.append("path leaves the operating-area geofence")
        for i, nfz in enumerate(nfz_polys):
            if line.intersects(nfz.buffer(0)):
                problems.append(f"path intersects no-fly-zone[{i}] (bounds {list(nfz.bounds)})")
        return json.dumps({"clear": not problems, "violations": problems})

    @tool
    def estimate_battery(legs: list[dict]) -> str:
        """Estimate flight time and battery for a candidate set of legs using the
        SELECTED drone's physics model — use it to confirm a plan fits before you
        finalize it. Each leg is
        {"leg_type": "TRANSIT|SEARCH_PATTERN|LOITER|SENSOR_TASK|RETURN_TO_BASE",
         "sensor_mode": "EO"|"IR"|"OFF"|null,
         "waypoints": [[lat, lon, alt_m], ...]}.
        Returns total_duration_s, total_battery_pct consumed, reserve_pct from a
        full pack, the rated endurance, and whether it fits endurance + the 20%
        reserve rule."""
        try:
            mission_legs: list[MissionLeg] = []
            for lg in legs:
                wps = [
                    Waypoint(lat=float(w[0]), lon=float(w[1]), alt_m=float(w[2]))
                    for w in lg.get("waypoints", [])
                ]
                sm = lg.get("sensor_mode")
                mission_legs.append(
                    MissionLeg(
                        leg_type=LegType(lg["leg_type"]),
                        sensor_mode=SensorMode(sm) if sm else None,
                        geometry=wps,
                        est_duration_s=0.0,
                        est_battery_pct=0.0,
                    )
                )
            energy = estimate_mission(mission_legs, profile)
            return json.dumps(
                {
                    "total_duration_s": round(energy.total_duration_s, 1),
                    "total_battery_pct": round(energy.total_battery_pct, 1),
                    "reserve_pct": round(100.0 - energy.total_battery_pct, 1),
                    "rated_endurance_s": profile.rated_endurance_s,
                    "fits_endurance": energy.total_duration_s <= profile.rated_endurance_s,
                    "fits_reserve_20pct": (100.0 - energy.total_battery_pct) >= 20.0,
                }
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"could not estimate: {type(e).__name__}: {e}"})

    @tool
    def get_sensor_coverage(altitude_m: float, sensor_mode: str) -> str:
        """Return the ground swath width and resolution for a sensor at a given
        altitude, plus the recommended search-track spacing (85% of swath, for
        overlap so there are no coverage gaps). Use this to pick an altitude and
        to space lawnmower tracks correctly. `sensor_mode` is "EO" or "IR"."""
        try:
            spec = profile.sensor(SensorMode(sensor_mode.upper()))
        except Exception:  # noqa: BLE001
            spec = None
        if spec is None:
            return json.dumps(
                {
                    "error": f"this drone has no {sensor_mode!r} sensor",
                    "available_sensors": [s.mode.value for s in profile.sensors],
                }
            )
        swath = sensor_swath_m(spec, altitude_m)
        return json.dumps(
            {
                "altitude_m": altitude_m,
                "sensor": sensor_mode.upper(),
                "swath_width_m": round(swath, 1),
                "ground_resolution_m_per_px": round(sensor_ground_resolution_m_per_px(spec, altitude_m), 4),
                "recommended_track_spacing_m": round(swath * 0.85, 1),
            }
        )

    @tool
    def lookup_search_pattern(pattern_name: str, altitude_m: float, spacing_m: float = 0.0) -> str:
        """Generate valid, NFZ-aware search-pattern geometry fitted to the
        operating area, so you don't have to invent coordinates. Returns an
        ordered list of [lat, lon, alt_m] waypoints. `pattern_name` is
        "lawnmower" (parallel tracks — best for area coverage; pass spacing_m from
        get_sensor_coverage), "expanding_square" (SAR outward from the area
        center), or "sector" (spokes from the center). Choose the pattern; the
        tool lays out the geometry."""
        name = pattern_name.lower().strip()
        clat, clon = _centroid_lat_lon()
        try:
            if name in ("lawnmower", "boustrophedon", "parallel"):
                if boundary is None:
                    return json.dumps({"error": "no area boundary available"})
                swath = spacing_m if spacing_m > 0 else 75.0
                pts = lawnmower_fit_to_boundary(boundary, altitude_m, swath, nfz_polygons=nfz_polys)
            elif name in ("expanding_square", "expanding square", "square"):
                leg = spacing_m if spacing_m > 0 else 60.0
                pts = expanding_square(clat, clon, leg, leg, legs_count=10, altitude_m=altitude_m)
            elif name == "sector":
                pts = sector(clat, clon, radius_m=200.0, sectors=8, altitude_m=altitude_m)
            else:
                return json.dumps(
                    {
                        "error": f"unknown pattern {pattern_name!r}",
                        "supported": ["lawnmower", "expanding_square", "sector"],
                    }
                )
            return json.dumps(
                {
                    "pattern": name,
                    "num_waypoints": len(pts),
                    "waypoints_lat_lon_alt": [
                        [round(w.lat, 6), round(w.lon, 6), round(w.alt_m, 1)] for w in pts
                    ],
                }
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"could not generate pattern: {type(e).__name__}: {e}"})

    return [get_geofence, check_path_clear, estimate_battery, get_sensor_coverage, lookup_search_pattern]
