"""Live mission-execution telemetry stream.

See docs/DESIGN_DECISIONS.md §10. A server-side discrete-time stepper walks the
plan at fixed `dt = 0.2s` ticks using the same physics model the validator uses,
and emits flat-JSON frames the UI can render directly.

Time compression default = 10× wall-clock (so a 5-min mission finishes in 30s).
Override with `?speed=N`. Auto-aborts on geofence breach.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

from shapely.geometry import Point

from app.schemas.enums import LegType
from app.schemas.plan import MissionPlan, Waypoint
from app.validation.kernel import GeoContext
from app.validation.physics import (
    DroneProfile,
    estimate_leg,
    haversine_m,
)

log = logging.getLogger(__name__)

SIM_DT_S = 0.2  # simulation tick — small enough for smooth animation


@dataclass
class TelemetryFrame:
    t: float  # simulated elapsed seconds since takeoff
    lat: float
    lon: float
    alt_m: float
    battery_pct: float  # remaining battery
    leg_idx: int
    leg_type: str
    sensor_mode: str | None
    status: str  # "flying" | "done" | "aborted"
    detail: str = ""  # human reason on abort

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _interp_waypoint(a: Waypoint, b: Waypoint, frac: float) -> tuple[float, float, float]:
    """Linear interpolation along (lat, lon, alt) between two waypoints."""
    return (
        a.lat + (b.lat - a.lat) * frac,
        a.lon + (b.lon - a.lon) * frac,
        a.alt_m + (b.alt_m - a.alt_m) * frac,
    )


def _segment_distance_m(a: Waypoint, b: Waypoint) -> float:
    """3D distance: horizontal haversine + vertical, RSS combined."""
    horiz = haversine_m(a, b)
    vert = abs(b.alt_m - a.alt_m)
    return math.sqrt(horiz * horiz + vert * vert)


async def telemetry_stream(
    plan: MissionPlan,
    profile: DroneProfile,
    geo: GeoContext | None,
    starting_battery_pct: float = 100.0,
    speed: float = 10.0,
) -> AsyncIterator[TelemetryFrame]:
    """Walk the plan and yield TelemetryFrames in simulated real-time.

    `speed` is the time-compression factor (10.0 = 10x faster than wall clock).
    The stream always ends with a terminal frame whose status is "done" or
    "aborted" so the consumer can close cleanly.
    """
    if not plan.legs:
        yield TelemetryFrame(
            t=0.0,
            lat=0.0,
            lon=0.0,
            alt_m=0.0,
            battery_pct=starting_battery_pct,
            leg_idx=0,
            leg_type="NONE",
            sensor_mode=None,
            status="done",
            detail="empty plan",
        )
        return

    boundary = None
    if geo is not None:
        boundary = geo.boundary if geo.boundary.is_valid else geo.boundary.buffer(0)

    elapsed = 0.0
    battery_remaining = float(starting_battery_pct)

    for li, leg in enumerate(plan.legs):
        # Cumulative leg distance for interpolation. For a LOITER with one
        # waypoint we synthesize a self-loop so we still emit frames.
        if len(leg.geometry) == 1 and leg.leg_type == LegType.LOITER:
            wp = leg.geometry[0]
            leg_duration = max(leg.est_duration_s, 1.0)
            ticks = max(1, int(leg_duration * speed / (speed * SIM_DT_S)))
            energy = estimate_leg(leg, profile)
            batt_per_tick = energy.battery_pct / max(ticks, 1)
            for _ in range(ticks):
                await asyncio.sleep(SIM_DT_S / max(speed, 0.01))
                elapsed += SIM_DT_S
                battery_remaining = max(0.0, battery_remaining - batt_per_tick)
                yield TelemetryFrame(
                    t=elapsed,
                    lat=wp.lat,
                    lon=wp.lon,
                    alt_m=wp.alt_m,
                    battery_pct=battery_remaining,
                    leg_idx=li,
                    leg_type=leg.leg_type.value,
                    sensor_mode=leg.sensor_mode.value if leg.sensor_mode else None,
                    status="flying",
                )
            continue

        # Per-leg energy budget — distributed proportionally to per-segment
        # distance so battery drain matches the physics model.
        energy = estimate_leg(leg, profile)
        seg_dists = [
            _segment_distance_m(leg.geometry[i], leg.geometry[i + 1])
            for i in range(len(leg.geometry) - 1)
        ]
        total_dist = sum(seg_dists) or 1.0

        for si, (a, b) in enumerate(zip(leg.geometry[:-1], leg.geometry[1:], strict=True)):
            seg_d = seg_dists[si] or 1.0
            seg_share = seg_d / total_dist
            seg_duration_s = energy.duration_s * seg_share
            seg_battery = energy.battery_pct * seg_share
            ticks = max(1, int(math.ceil(seg_duration_s / SIM_DT_S)))
            for tick in range(ticks):
                frac = (tick + 1) / ticks
                lat, lon, alt = _interp_waypoint(a, b, frac)
                await asyncio.sleep(SIM_DT_S / max(speed, 0.01))
                elapsed += seg_duration_s / ticks
                battery_remaining = max(0.0, battery_remaining - seg_battery / ticks)

                # Geofence breach check — terminate the stream.
                if boundary is not None and not boundary.covers(Point(lon, lat)):
                    yield TelemetryFrame(
                        t=elapsed,
                        lat=lat,
                        lon=lon,
                        alt_m=alt,
                        battery_pct=battery_remaining,
                        leg_idx=li,
                        leg_type=leg.leg_type.value,
                        sensor_mode=leg.sensor_mode.value if leg.sensor_mode else None,
                        status="aborted",
                        detail="exited geofence",
                    )
                    return

                yield TelemetryFrame(
                    t=elapsed,
                    lat=lat,
                    lon=lon,
                    alt_m=alt,
                    battery_pct=battery_remaining,
                    leg_idx=li,
                    leg_type=leg.leg_type.value,
                    sensor_mode=leg.sensor_mode.value if leg.sensor_mode else None,
                    status="flying",
                )

    # Terminal frame: identical to the last "flying" position with status=done.
    last_leg = plan.legs[-1]
    last_wp = last_leg.geometry[-1]
    yield TelemetryFrame(
        t=elapsed,
        lat=last_wp.lat,
        lon=last_wp.lon,
        alt_m=last_wp.alt_m,
        battery_pct=battery_remaining,
        leg_idx=len(plan.legs) - 1,
        leg_type=last_leg.leg_type.value,
        sensor_mode=last_leg.sensor_mode.value if last_leg.sensor_mode else None,
        status="done",
    )
