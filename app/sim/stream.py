"""Live mission-execution telemetry stream.

See docs/DESIGN_DECISIONS.md §10 and DD-015. The stream walks the SAME kinematic
trajectory the verifier uses (`sim/kinematics.py`) — so the live animation now
slows into turns, ramps up/brakes for accel limits, and reacts to wind, instead
of sliding between waypoints at a flat speed. It emits flat-JSON frames the UI
renders directly.

Time compression default = 10x wall-clock (a 5-min mission finishes in 30s).
Override with `?speed=N`. Auto-aborts on geofence breach or NFZ entry.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

from shapely.geometry import Point

from app.config import get_settings
from app.schemas.plan import MissionPlan
from app.sim.kinematics import simulate_trajectory
from app.validation.kernel import GeoContext
from app.validation.physics import DroneProfile

log = logging.getLogger(__name__)

SIM_DT_S = 0.2  # fine tick for smooth animation


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


async def telemetry_stream(
    plan: MissionPlan,
    profile: DroneProfile,
    geo: GeoContext | None,
    starting_battery_pct: float = 100.0,
    speed: float = 10.0,
) -> AsyncIterator[TelemetryFrame]:
    """Walk the kinematic trajectory and yield TelemetryFrames in sim-time.

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

    settings = get_settings()
    boundary = None
    nfzs: list[Any] = []
    wind_mps = 0.0
    wind_from_deg = 0.0
    if geo is not None:
        boundary = geo.boundary if geo.boundary.is_valid else geo.boundary.buffer(0)
        nfzs = [(n if n.is_valid else n.buffer(0)) for n in geo.nfz_polygons]
        try:
            from app.weather import get_weather_for_area

            home_lon, home_lat = geo.home_point
            wx = get_weather_for_area(home_lat, home_lon)
            wind_mps = wx.wind_mps
            wind_from_deg = wx.wind_dir_deg
        except Exception:  # noqa: BLE001 — calm-air fallback
            wind_mps = 0.0

    trace = simulate_trajectory(
        plan,
        profile,
        wind_mps=wind_mps,
        wind_from_deg=wind_from_deg,
        wind_coeff=settings.weather_wind_power_coeff,
        dt=SIM_DT_S,
    )

    last = None
    for s in trace.samples:
        await asyncio.sleep(SIM_DT_S / max(speed, 0.01))
        battery_remaining = max(0.0, starting_battery_pct - s.battery_used_pct)
        last = s
        pt = Point(s.lon, s.lat)

        if boundary is not None and not boundary.covers(pt):
            yield TelemetryFrame(
                t=s.t,
                lat=s.lat,
                lon=s.lon,
                alt_m=s.alt_m,
                battery_pct=battery_remaining,
                leg_idx=s.leg_idx,
                leg_type=s.leg_type,
                sensor_mode=s.sensor_mode,
                status="aborted",
                detail="exited geofence",
            )
            return
        breached_nfz = next((ni for ni, nfz in enumerate(nfzs) if nfz.covers(pt)), None)
        if breached_nfz is not None:
            yield TelemetryFrame(
                t=s.t,
                lat=s.lat,
                lon=s.lon,
                alt_m=s.alt_m,
                battery_pct=battery_remaining,
                leg_idx=s.leg_idx,
                leg_type=s.leg_type,
                sensor_mode=s.sensor_mode,
                status="aborted",
                detail=f"entered no-fly-zone[{breached_nfz}]",
            )
            return

        yield TelemetryFrame(
            t=s.t,
            lat=s.lat,
            lon=s.lon,
            alt_m=s.alt_m,
            battery_pct=battery_remaining,
            leg_idx=s.leg_idx,
            leg_type=s.leg_type,
            sensor_mode=s.sensor_mode,
            status="flying",
        )

    # Terminal frame: same position as the last tick, status=done.
    if last is not None:
        yield TelemetryFrame(
            t=last.t,
            lat=last.lat,
            lon=last.lon,
            alt_m=last.alt_m,
            battery_pct=max(0.0, starting_battery_pct - last.battery_used_pct),
            leg_idx=last.leg_idx,
            leg_type=last.leg_type,
            sensor_mode=last.sensor_mode,
            status="done",
        )
