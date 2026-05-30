"""Deterministic decision policy for the inflight supervisor.

Given current telemetry + a list of pending events + the remaining plan, return
one `SupervisorDecision` plus a human-readable reason.

This is deliberately rule-based, not LLM-based, for three reasons:

1. **Latency.** A live mission ticks at 5 Hz. An LLM round-trip per tick would
   queue up and the drone would drift past the deadline.
2. **Determinism.** Safety-critical decisions should be reproducible across runs
   and unit-testable without mocking an LLM.
3. **Cost.** The compile graph already burns one LLM call per repair attempt;
   adding another per supervisor tick is wasteful when the decision rules are
   plain conjunctions of measurable state.

An LLM-flavored "tactical advisor" pass on top of this is a future extension
(see DESIGN_DECISIONS DD-014 trade-off section) and would be invoked at most
once per event, not per tick.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from shapely.geometry import LineString, Point, Polygon, shape

from app.config import get_settings
from app.supervisor.events import Event, EventType
from app.supervisor.state import SupervisorDecision

# ---- thresholds ------------------------------------------------------------
#
# All thresholds are CONST-by-default; the kernel's existing settings are the
# source of truth for hard limits (e.g. weather_max_wind_mps). The supervisor
# is allowed to be slightly more conservative than the planner — a flying
# drone has less margin than one still in pre-flight.

# Reserve margin enforced by the supervisor (in addition to the 20% the
# planner already budgeted). If projected battery at home is below this,
# RTB immediately.
_SUPERVISOR_RESERVE_MARGIN_PCT: float = 15.0


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance (meters) — duplicated locally to avoid importing
    the physics module here (keeps this file pure-Python + shapely only)."""
    earth_r_m = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * earth_r_m * math.asin(math.sqrt(a))


def _remaining_path_intersects(remaining_legs: list[dict[str, Any]], polygon: Polygon) -> bool:
    """Returns True if any waypoint of the remaining plan lies inside the polygon,
    or if any segment crosses it. Lossless 2D check."""
    for leg in remaining_legs:
        geom = leg.get("geometry") or []
        if len(geom) == 0:
            continue
        pts = [(g["lon"], g["lat"]) for g in geom]
        for x, y in pts:
            if polygon.covers(Point(x, y)):
                return True
        if len(pts) >= 2 and LineString(pts).intersects(polygon):
            return True
    return False


def _estimate_rtb_battery_pct(
    current: dict[str, Any],
    home_lon: float,
    home_lat: float,
    cruise_speed_mps: float,
    cruise_power_w: float,
    battery_wh: float,
) -> float:
    """Quick projection of battery consumed to fly direct from current → home.
    Mirrors physics.estimate_leg's cruise model but stays self-contained."""
    if battery_wh <= 0 or cruise_speed_mps <= 0:
        return 0.0
    d_m = _haversine_m(current["lon"], current["lat"], home_lon, home_lat)
    t_s = d_m / cruise_speed_mps
    wh = (cruise_power_w * t_s) / 3600.0
    return (wh / battery_wh) * 100.0


def decide(
    *,
    current: dict[str, Any],
    remaining_legs: list[dict[str, Any]],
    home_lon: float,
    home_lat: float,
    pending_events: Iterable[Event],
    drone_profile: dict[str, Any],
) -> tuple[SupervisorDecision, str]:
    """Pure decision function. Returns (decision, reason).

    `drone_profile` carries cruise_speed_mps, cruise_power_w, battery_wh.
    `current` is a TelemetryFrame dict.
    Events are evaluated in priority order; the first one whose policy yields
    a non-CONTINUE decision wins.
    """
    settings = get_settings()
    cruise_speed = float(drone_profile.get("cruise_speed_mps", 15.0))
    cruise_power = float(drone_profile.get("cruise_power_w", 300.0))
    battery_wh = float(drone_profile.get("battery_wh", 400.0))

    # Always-on background check: enough battery to make it home with margin?
    rtb_cost_pct = _estimate_rtb_battery_pct(
        current, home_lon, home_lat, cruise_speed, cruise_power, battery_wh
    )
    remaining_batt = float(current.get("battery_pct", 100.0))
    projected_at_home = remaining_batt - rtb_cost_pct
    if projected_at_home < _SUPERVISOR_RESERVE_MARGIN_PCT:
        return (
            SupervisorDecision.RTB_NOW,
            f"projected battery at home {projected_at_home:.1f}% < margin "
            f"{_SUPERVISOR_RESERVE_MARGIN_PCT:.0f}% (rtb cost {rtb_cost_pct:.1f}%)",
        )

    # Event-priority order: emergency land > lost link > NFZ > wind.
    order = {
        EventType.MANUAL_LAND: 0,
        EventType.LOST_LINK: 1,
        EventType.NFZ_POPUP: 2,
        EventType.WIND_SPIKE: 3,
    }
    events_sorted = sorted(pending_events, key=lambda e: order.get(e.type, 99))

    for ev in events_sorted:
        if ev.type == EventType.MANUAL_LAND:
            return SupervisorDecision.EMERGENCY_LAND, ev.note or "operator triggered emergency land"

        if ev.type == EventType.LOST_LINK:
            # Standard lost-C2/GPS contingency: hold position, then return home.
            return (
                SupervisorDecision.LOITER_RTB,
                ev.note or "C2/GPS link lost — holding position, then returning to base",
            )

        if ev.type == EventType.NFZ_POPUP:
            try:
                poly = shape(ev.payload.get("polygon") or {})
            except Exception as e:  # noqa: BLE001
                return SupervisorDecision.CONTINUE, f"nfz_popup payload invalid: {e}"
            if not poly.is_valid or poly.is_empty:
                return SupervisorDecision.CONTINUE, "nfz_popup polygon empty"
            if poly.covers(Point(current["lon"], current["lat"])):
                # Drone is already inside it — can't detour out cleanly; bail home.
                return (
                    SupervisorDecision.RTB_NOW,
                    "restricted airspace activated over current position",
                )
            if _remaining_path_intersects(remaining_legs, poly):
                return (
                    SupervisorDecision.REPLAN_FROM_HERE,
                    "restricted airspace ahead — rerouting around it, mission continues",
                )
            return (
                SupervisorDecision.CONTINUE,
                "restricted airspace appeared but clear of remaining route",
            )

        if ev.type == EventType.WIND_SPIKE:
            wind = float(ev.payload.get("wind_mps") or 0.0)
            if wind > settings.weather_max_wind_mps:
                return (
                    SupervisorDecision.RTB_NOW,
                    f"wind gust {wind:.1f} m/s exceeds airframe limit "
                    f"{settings.weather_max_wind_mps:.1f} m/s — returning to base",
                )
            return (
                SupervisorDecision.CONTINUE,
                f"wind gust {wind:.1f} m/s within limit {settings.weather_max_wind_mps:.1f} m/s",
            )

    return SupervisorDecision.CONTINUE, "no actionable events"


__all__ = ["decide"]
