"""Search-pattern geometry generators.

Each generator returns a list of Waypoint that, when flown in order, executes
the named pattern. Pure functions — no LLM, no network. Used by the
`lookup_search_pattern` tool so the agent does not invent geometry.
"""

from __future__ import annotations

import math

from app.schemas.plan import Waypoint

METERS_PER_DEG_LAT = 111_320.0


def _meters_per_deg_lon(lat_deg: float) -> float:
    return METERS_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def lawnmower(
    center_lat: float,
    center_lon: float,
    width_m: float,
    height_m: float,
    spacing_m: float,
    altitude_m: float,
    bearing_deg: float = 0.0,  # noqa: ARG001 — informational; rotation is a future extension
) -> list[Waypoint]:
    """Boustrophedon parallel-track pattern. East-aligned by default.

    Tracks run east-west; consecutive tracks are `spacing_m` apart north-south.
    The pattern hits each corner of a `width_m x height_m` rectangle centered on
    (center_lat, center_lon). Bearing is currently informational; for an
    axis-aligned envelope this is sufficient for evals.
    """
    half_w_deg = (width_m / 2.0) / _meters_per_deg_lon(center_lat)
    half_h_deg = (height_m / 2.0) / METERS_PER_DEG_LAT
    spacing_deg = spacing_m / METERS_PER_DEG_LAT

    points: list[Waypoint] = []
    lat = center_lat - half_h_deg
    direction = 1  # 1: east, -1: west
    while lat <= center_lat + half_h_deg + 1e-9:
        west_lon = center_lon - half_w_deg
        east_lon = center_lon + half_w_deg
        if direction == 1:
            points.append(Waypoint(lat=lat, lon=west_lon, alt_m=altitude_m))
            points.append(Waypoint(lat=lat, lon=east_lon, alt_m=altitude_m))
        else:
            points.append(Waypoint(lat=lat, lon=east_lon, alt_m=altitude_m))
            points.append(Waypoint(lat=lat, lon=west_lon, alt_m=altitude_m))
        direction *= -1
        lat += spacing_deg
    return points


def lawnmower_fit_to_boundary(
    boundary: Polygon,  # noqa: F821 — shapely Polygon, deferred import
    altitude_m: float,
    swath_m: float,
    spacing_factor: float = 0.85,
    inset_factor: float = 0.30,
) -> list[Waypoint]:
    """Lay out a lawnmower pattern that fits the boundary's bbox at `altitude_m`.

    Track spacing is `spacing_factor * swath_m` (default 85% — leaves a safety
    margin so the kernel's coverage check passes). The bbox is inset by
    `inset_factor * swath_m` on each side so the tracks stay safely inside the
    boundary polygon (cheap proxy for proper polygon-clipped pattern fill).

    This is the deterministic counterpart that lets the LLM pick the pattern
    while the geometry is generated correctly. See DESIGN_DECISIONS §10.
    """
    minx, miny, maxx, maxy = boundary.bounds
    inset_lon = (swath_m * inset_factor) / _meters_per_deg_lon((miny + maxy) / 2)
    inset_lat = (swath_m * inset_factor) / METERS_PER_DEG_LAT
    minx += inset_lon
    maxx -= inset_lon
    miny += inset_lat
    maxy -= inset_lat
    if minx >= maxx or miny >= maxy:
        # Boundary too narrow to inset — degrade gracefully with a tiny pattern.
        minx, miny, maxx, maxy = boundary.bounds
    width_m = (maxx - minx) * _meters_per_deg_lon((miny + maxy) / 2)
    height_m = (maxy - miny) * METERS_PER_DEG_LAT
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    spacing_m = max(10.0, swath_m * spacing_factor)
    return lawnmower(center_lat, center_lon, width_m, height_m, spacing_m, altitude_m)


def expanding_square(
    center_lat: float,
    center_lon: float,
    initial_leg_m: float,
    growth_m: float,
    legs_count: int,
    altitude_m: float,
) -> list[Waypoint]:
    """Classic search-and-rescue expanding-square pattern. Heading rotates 90°
    each leg; leg length grows by `growth_m` every two legs.
    """
    points = [Waypoint(lat=center_lat, lon=center_lon, alt_m=altitude_m)]
    heading_deg = 0.0  # north
    leg_m = initial_leg_m
    cur_lat, cur_lon = center_lat, center_lon
    for i in range(legs_count):
        # convert leg_m + heading into delta lat/lon
        rad = math.radians(heading_deg)
        d_lat = (leg_m * math.cos(rad)) / METERS_PER_DEG_LAT
        d_lon = (leg_m * math.sin(rad)) / _meters_per_deg_lon(cur_lat)
        cur_lat += d_lat
        cur_lon += d_lon
        points.append(Waypoint(lat=cur_lat, lon=cur_lon, alt_m=altitude_m))
        heading_deg = (heading_deg + 90.0) % 360.0
        if i % 2 == 1:
            leg_m += growth_m
    return points


def sector(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    sectors: int,
    altitude_m: float,
) -> list[Waypoint]:
    """Sector search: spokes radiating from center at evenly spaced bearings."""
    points = [Waypoint(lat=center_lat, lon=center_lon, alt_m=altitude_m)]
    for i in range(sectors):
        bearing = (360.0 * i) / sectors
        rad = math.radians(bearing)
        d_lat = (radius_m * math.cos(rad)) / METERS_PER_DEG_LAT
        d_lon = (radius_m * math.sin(rad)) / _meters_per_deg_lon(center_lat)
        points.append(Waypoint(lat=center_lat + d_lat, lon=center_lon + d_lon, alt_m=altitude_m))
        points.append(Waypoint(lat=center_lat, lon=center_lon, alt_m=altitude_m))
    return points
