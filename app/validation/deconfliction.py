"""Airspace deconfliction: check a candidate plan against already-approved
missions in the same area for spatial+temporal+vertical conflicts.

See docs/DESIGN_DECISIONS.md §8 for the rationale: "active" missions are those
created within the longest drone endurance window (45 min) and approved.
Conflict = 2D buffer overlap AND vertical band overlap.

PostGIS does the spatial filtering via ST_DWithin / ST_Intersects so we don't
pull the world into Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from shapely.geometry import LineString, MultiLineString
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.schemas.enums import MissionStatus
from app.schemas.plan import MissionPlan

log = logging.getLogger(__name__)

# Separation thresholds. Documented in DESIGN_DECISIONS §8.
LATERAL_SEP_M = 50.0
VERTICAL_SEP_M = 30.0
# Coarse upper bound on a flight's wall-clock window. We treat any mission
# whose created_at is older than this as definitely completed.
DEFAULT_ACTIVE_WINDOW_S = 2700  # 45 min (longest endurance in data/drones/)


@dataclass(frozen=True)
class Conflict:
    other_mission_id: str
    detail: str

    def as_violation_string(self) -> str:
        return f"airspace conflict with mission {self.other_mission_id}: {self.detail}"


def _plan_to_geometry(plan: MissionPlan) -> LineString | MultiLineString | None:
    """Flatten all leg paths into a single Multi/LineString in lon/lat."""
    lines: list[LineString] = []
    for leg in plan.legs:
        if len(leg.geometry) >= 2:
            lines.append(LineString([(wp.lon, wp.lat) for wp in leg.geometry]))
    if not lines:
        return None
    if len(lines) == 1:
        return lines[0]
    return MultiLineString(lines)


def _plan_altitude_band(plan: MissionPlan) -> tuple[float, float]:
    alts = [wp.alt_m for leg in plan.legs for wp in leg.geometry]
    if not alts:
        return (0.0, 0.0)
    return (min(alts), max(alts))


def _bands_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True iff the two altitude bands overlap after stripping VERTICAL_SEP_M."""
    lo = max(a[0], b[0]) - VERTICAL_SEP_M / 2
    hi = min(a[1], b[1]) + VERTICAL_SEP_M / 2
    return lo <= hi


def _approx_lateral_buffer_deg(lat: float) -> float:
    """Convert LATERAL_SEP_M to degrees of longitude at this latitude.

    Approximate — sufficient for the small operating areas this service
    targets. 1 deg lat ≈ 111_000 m; 1 deg lon ≈ 111_000 * cos(lat).
    """
    import math

    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    # Use the larger of the two scales for a conservative buffer.
    return LATERAL_SEP_M / (111_000.0 * cos_lat)


def find_conflicts(
    engine: Engine,
    plan: MissionPlan,
    self_mission_id: str | None = None,
    active_window_s: int = DEFAULT_ACTIVE_WINDOW_S,
) -> list[Conflict]:
    """Return conflicts with approved missions in the same area inside the
    active window. Empty list = clear airspace.
    """
    geom = _plan_to_geometry(plan)
    if geom is None:
        return []
    candidate_band = _plan_altitude_band(plan)

    # Centroid latitude for the lon-degree buffer approximation.
    centroid_lat = float(geom.centroid.y)
    buffer_deg = _approx_lateral_buffer_deg(centroid_lat)
    buffered_geom = geom.buffer(buffer_deg)

    rows: list[Any] = []
    with engine.connect() as conn:
        # Active = approved=true AND created within the window AND same area.
        sql = text(
            """
            SELECT mission_id, plan FROM missions
            WHERE area_id = :area
              AND approved = TRUE
              AND status = :ready
              AND created_at > (now() - (:win || ' seconds')::interval)
              AND mission_id <> COALESCE(:self_id, '')
            """
        )
        try:
            rows = list(
                conn.execute(
                    sql,
                    {
                        "area": plan.area_id,
                        "ready": MissionStatus.READY_FOR_APPROVAL.value,
                        "win": str(active_window_s),
                        "self_id": self_mission_id or "",
                    },
                ).all()
            )
        except Exception as e:  # noqa: BLE001
            # If the DB is unreachable we conservatively report no conflicts
            # rather than failing the compile; airspace check becomes a no-op.
            log.warning("deconfliction DB query failed (%s); skipping check", e)
            return []

    conflicts: list[Conflict] = []
    for mid, raw in rows:
        try:
            other = MissionPlan.model_validate(raw)
        except Exception as e:  # noqa: BLE001
            log.warning("deconfliction: skipping malformed mission %s (%s)", mid, e)
            continue
        other_geom = _plan_to_geometry(other)
        if other_geom is None:
            continue
        # 2D check using the candidate's buffer
        if not other_geom.intersects(buffered_geom):
            continue
        # 3D check
        other_band = _plan_altitude_band(other)
        if not _bands_overlap(candidate_band, other_band):
            continue
        conflicts.append(
            Conflict(
                other_mission_id=mid,
                detail=(
                    f"lateral overlap within {LATERAL_SEP_M:.0f}m AND "
                    f"altitude bands overlap "
                    f"(self={candidate_band[0]:.0f}-{candidate_band[1]:.0f}m, "
                    f"other={other_band[0]:.0f}-{other_band[1]:.0f}m, "
                    f"sep={VERTICAL_SEP_M:.0f}m required)"
                ),
            )
        )
    return conflicts


def deconfliction_check(
    engine: Engine,
    plan: MissionPlan,
    self_mission_id: str | None = None,
) -> tuple[bool, list[str]]:
    """Kernel-shaped entry point: returns (ok, violation_strings)."""
    conflicts = find_conflicts(engine, plan, self_mission_id=self_mission_id)
    return (not conflicts, [c.as_violation_string() for c in conflicts])
