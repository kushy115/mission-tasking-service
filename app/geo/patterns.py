"""Search-pattern geometry generators.

Each generator returns a list of Waypoint that, when flown in order, executes
the named pattern. Pure functions — no LLM, no network. Used by the
`lookup_search_pattern` tool so the agent does not invent geometry.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from app.schemas.plan import Waypoint

if TYPE_CHECKING:
    from shapely.geometry import Polygon

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


def _segment_is_safe(
    a: tuple[float, float],
    b: tuple[float, float],
    obstacles: list[Polygon],
) -> bool:
    """True if the straight segment a→b does not pass through any obstacle's
    interior. Endpoints/edges touching an obstacle boundary are allowed (that's
    where clipped track ends legitimately sit)."""
    from shapely.geometry import LineString  # noqa: PLC0415

    seg = LineString([a, b])
    for obs in obstacles:
        # Negative-buffer the obstacle by a hair so boundary-touching segments
        # (clipped track endpoints sit exactly on the buffer edge) don't count
        # as intersections — only true interior passage does.
        shrunk = obs.buffer(-1e-7)
        if not shrunk.is_empty and seg.intersects(shrunk):
            return False
    return True


def _route_through_flyable(
    p1: tuple[float, float],
    p2: tuple[float, float],
    obstacles: list[Polygon],
) -> list[tuple[float, float]]:
    """Visibility-graph shortest path from p1→p2 that avoids every obstacle's
    interior. Returns the intermediate waypoints (excluding p1 and p2).

    Nodes are p1, p2, and every obstacle-buffer exterior vertex. An edge exists
    between two nodes when the straight segment between them is obstacle-free
    (`_segment_is_safe`). Dijkstra finds the shortest safe polyline. Small graph
    (a few dozen nodes), so the O(n²) edge build is cheap. Pure Shapely + heapq;
    no external graph dependency.
    """
    import heapq  # noqa: PLC0415
    import math as _math  # noqa: PLC0415

    if _segment_is_safe(p1, p2, obstacles):
        return []

    nodes: list[tuple[float, float]] = [p1, p2]
    for obs in obstacles:
        nodes.extend((x, y) for x, y in obs.exterior.coords)
    # De-dup nodes (buffers can share vertices).
    seen: set[tuple[float, float]] = set()
    uniq: list[tuple[float, float]] = []
    for node in nodes:
        key = (round(node[0], 9), round(node[1], 9))
        if key not in seen:
            seen.add(key)
            uniq.append(node)
    nodes = uniq

    def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return _math.hypot(a[0] - b[0], a[1] - b[1])

    n = len(nodes)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _segment_is_safe(nodes[i], nodes[j], obstacles):
                w = dist(nodes[i], nodes[j])
                adj[i].append((j, w))
                adj[j].append((i, w))

    src, dst = 0, 1
    best = [float("inf")] * n
    prev = [-1] * n
    best[src] = 0.0
    pq: list[tuple[float, int]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > best[u]:
            continue
        if u == dst:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < best[v]:
                best[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if best[dst] == float("inf"):
        return []  # no safe path found; caller keeps the (unsafe) direct line
    path: list[int] = []
    cur = dst
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    # Drop endpoints; return only intermediates.
    return [nodes[i] for i in path[1:-1]]


def lawnmower_fit_to_boundary(
    boundary: Polygon,
    altitude_m: float,
    swath_m: float,
    spacing_factor: float = 0.85,
    inset_factor: float = 0.30,
    nfz_polygons: list[Polygon] | None = None,
    nfz_margin_m: float = 15.0,
    sub_region: Polygon | None = None,
) -> list[Waypoint]:
    """Lay out a lawnmower pattern that fits the boundary's bbox at `altitude_m`.

    Track spacing is `spacing_factor * swath_m` (default 85% — leaves a safety
    margin so the kernel's coverage check passes). The bbox is inset by
    `inset_factor * swath_m` on each side so the tracks stay safely inside the
    boundary polygon (cheap proxy for proper polygon-clipped pattern fill).

    When `nfz_polygons` is provided, every generated track is intersected with
    `boundary − union(nfz buffered by nfz_margin_m)` so tracks split around
    NFZs instead of laying across them. Required for areas with interior
    no-fly zones — without it the kernel rejects on NFZ intersection.

    When `sub_region` is provided (multi-drone spatial decomposition), the
    pattern is fitted to `boundary ∩ sub_region` instead of the whole boundary,
    so each drone in a group sweeps only its assigned slice of the area.

    This is the deterministic counterpart that lets the LLM pick the pattern
    while the geometry is generated correctly. See DESIGN_DECISIONS §10.
    """
    from shapely.geometry import LineString  # noqa: PLC0415
    from shapely.geometry.base import BaseGeometry  # noqa: PLC0415
    from shapely.ops import unary_union  # noqa: PLC0415

    if sub_region is not None:
        clipped_boundary = boundary.intersection(sub_region)
        if not clipped_boundary.is_empty and clipped_boundary.area > 0:
            boundary = clipped_boundary

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

    spacing_m = max(10.0, swath_m * spacing_factor)
    spacing_deg = spacing_m / METERS_PER_DEG_LAT

    mean_lat = (miny + maxy) / 2.0
    deg_per_m_lat = 1.0 / METERS_PER_DEG_LAT
    deg_per_m_lon = 1.0 / max(_meters_per_deg_lon(mean_lat), 1e-6)
    nfz_buf_deg = nfz_margin_m * (deg_per_m_lat + deg_per_m_lon) / 2.0
    boundary_inset_deg = (swath_m * inset_factor) * (deg_per_m_lat + deg_per_m_lon) / 2.0

    # Real boundary-aware inset so clipped track endpoints stay strictly inside
    # the geofence (bbox-only inset isn't enough for concave shapes).
    inset_boundary: BaseGeometry = boundary.buffer(-boundary_inset_deg)
    if inset_boundary.is_empty:
        inset_boundary = boundary

    nfz_bufs: list[Polygon] = []
    flyable: BaseGeometry = inset_boundary
    if nfz_polygons:
        nfz_bufs = [p.buffer(nfz_buf_deg) for p in nfz_polygons if p and not p.is_empty]
        if nfz_bufs:
            nfz_union = unary_union(nfz_bufs)
            flyable = inset_boundary.difference(nfz_union)
            if flyable.is_empty:
                flyable = inset_boundary  # fall back; pattern won't fit

    points: list[Waypoint] = []
    lat = miny
    direction = 1  # 1: west→east, -1: east→west
    while lat <= maxy + 1e-9:
        west = (minx, lat)
        east = (maxx, lat)
        track = LineString([west, east])
        segments: list[LineString] = []
        clipped = track.intersection(flyable)
        if not clipped.is_empty:
            geoms = getattr(clipped, "geoms", [clipped])
            for g in geoms:
                if isinstance(g, LineString) and g.length > 0:
                    segments.append(g)
        if not segments:
            lat += spacing_deg
            continue

        # Order segments so we sweep in the current direction.
        segments.sort(key=lambda s: s.coords[0][0], reverse=(direction < 0))
        for seg in segments:
            coords = list(seg.coords)
            if direction < 0:
                coords = list(reversed(coords))
            for x, y in coords:
                points.append(Waypoint(lat=y, lon=x, alt_m=altitude_m))
        direction *= -1
        lat += spacing_deg

    # ----- Detour pass: between every consecutive waypoint pair whose direct
    # line clips an NFZ buffer, insert a visibility-graph shortest path that
    # routes around the obstacle(s). Without this the kernel rejects on NFZ
    # intersection even though individual track segments are NFZ-free.
    if nfz_bufs and len(points) >= 2:
        fixed: list[Waypoint] = [points[0]]
        for prev, curr in zip(points, points[1:], strict=False):
            p1 = (prev.lon, prev.lat)
            p2 = (curr.lon, curr.lat)
            detour = _route_through_flyable(p1, p2, nfz_bufs)
            for lon, latd in detour:
                fixed.append(Waypoint(lat=latd, lon=lon, alt_m=altitude_m))
            fixed.append(curr)
        points = fixed

    return points


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


def band_subregion(boundary: Polygon, index: int, total: int) -> Polygon | None:
    """Return the `index`-th EQUAL-AREA horizontal slice of `total` bands, as a
    rectangle Polygon. Used for multi-drone spatial decomposition so each drone
    sweeps a non-overlapping strip carrying roughly equal coverage load.

    Bands are split along latitude (north-south) because lawnmower tracks run
    east-west — splitting perpendicular to the tracks keeps each drone's tracks
    full-width and minimizes partial tracks. The split LINES are chosen so each
    band contains ~1/total of the boundary's AREA (not equal latitude height) —
    a concave/irregular boundary otherwise gives some drones far larger bands,
    which is what made per-drone durations (and rejection risk) uneven. A small
    overlap is added between adjacent bands so seams are double-covered.
    """
    from shapely.geometry import box  # noqa: PLC0415

    if total <= 1 or index < 0 or index >= total:
        return None
    minx, miny, maxx, maxy = boundary.bounds
    span = maxy - miny
    if span <= 0:
        return None
    total_area = boundary.area
    if total_area <= 0:
        return None

    # Find latitude split lines so cumulative area below each line is k/total of
    # the whole. Sample the cumulative-area curve and invert it by interpolation.
    samples = max(64, total * 16)
    cum: list[tuple[float, float]] = [(miny, 0.0)]  # (lat, area-below)
    for s in range(1, samples + 1):
        lat = miny + span * s / samples
        strip = boundary.intersection(box(minx, miny, maxx, lat))
        cum.append((lat, strip.area))

    def _lat_at_area(target: float) -> float:
        for (la, aa), (lb, ab) in zip(cum, cum[1:], strict=False):
            if ab >= target:
                if ab == aa:
                    return lb
                frac = (target - aa) / (ab - aa)
                return la + (lb - la) * frac
        return float(maxy)

    lo_lat = _lat_at_area(total_area * index / total)
    hi_lat = _lat_at_area(total_area * (index + 1) / total)
    band_h = hi_lat - lo_lat
    overlap = band_h * 0.05
    lo = lo_lat - (overlap if index > 0 else 0.0)
    hi = hi_lat + (overlap if index < total - 1 else 0.0)
    pad = (maxx - minx) * 0.01 + 1e-6
    return box(minx - pad, lo, maxx + pad, hi)


def perimeter_patrol_fit_to_boundary(
    boundary: Polygon,
    nfz_polygons: list[Polygon],
    altitude_m: float,
    samples: int = 16,
    inset_m: float = 12.0,
    nfz_margin_m: float = 18.0,
) -> list[Waypoint]:
    """Deterministic perimeter-patrol waypoints that respect the boundary AND NFZs.

    Used to replace LLM-emitted TRANSIT geometry for commands like "patrol the
    perimeter" — the LLM is unreliable at hand-picking corner waypoints when
    NFZs are present (it routinely clips them or drifts outside the boundary).

    Strategy:
      1. Negative-buffer the boundary by `inset_m` (keeps the patrol track
         inside the geofence with margin).
      2. Subtract every NFZ (positive-buffered by `nfz_margin_m`) from the
         inset region — that leaves the *flyable* corridor.
      3. Walk the largest resulting polygon's exterior, sampling `samples`
         points at even arc-length intervals.

    All math via Shapely; pure function; no LLM in the loop.
    """
    from shapely.geometry import MultiPolygon  # noqa: PLC0415
    from shapely.geometry import Polygon as _SPoly
    from shapely.geometry.base import BaseGeometry  # noqa: PLC0415

    if boundary.is_empty or samples < 3:
        return []

    centroid = boundary.centroid
    mean_lat = centroid.y if centroid and centroid.y is not None else 0.0
    deg_per_m_lat = 1.0 / METERS_PER_DEG_LAT
    deg_per_m_lon = 1.0 / max(_meters_per_deg_lon(mean_lat), 1e-6)

    # Use a degree-space buffer that approximates the requested meter margin.
    # Buffer-by-degrees is inhomogeneous in lat/lon but at small scales (a few
    # tens of meters over 1 km² polygons) the error is well under our margins.
    inset_deg = inset_m * (deg_per_m_lat + deg_per_m_lon) / 2.0
    nfz_buf_deg = nfz_margin_m * (deg_per_m_lat + deg_per_m_lon) / 2.0

    inset: BaseGeometry = boundary.buffer(-inset_deg)
    if inset.is_empty:
        inset = boundary  # boundary too narrow to inset — fall back

    flyable: BaseGeometry = inset
    for nfz in nfz_polygons or []:
        if nfz is None or nfz.is_empty:
            continue
        flyable = flyable.difference(nfz.buffer(nfz_buf_deg))
        if flyable.is_empty:
            break

    if flyable.is_empty:
        return []

    # Pick the largest connected piece — that's the "main" flyable corridor.
    if isinstance(flyable, MultiPolygon):
        polys = sorted(flyable.geoms, key=lambda p: p.area, reverse=True)
        if not polys:
            return []
        flyable = polys[0]
    if not isinstance(flyable, _SPoly):
        return []

    exterior = flyable.exterior
    if exterior is None or exterior.length == 0:
        return []

    points: list[Waypoint] = []
    for i in range(samples):
        frac = i / float(samples)
        p = exterior.interpolate(frac, normalized=True)
        points.append(Waypoint(lat=p.y, lon=p.x, alt_m=altitude_m))
    # Close the loop so the last waypoint is the same as the first — readable
    # patrol path and the kernel doesn't care about duplicates.
    if points:
        points.append(Waypoint(lat=points[0].lat, lon=points[0].lon, alt_m=altitude_m))
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
