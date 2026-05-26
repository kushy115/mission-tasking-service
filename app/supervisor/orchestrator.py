"""Async orchestrator that wraps `telemetry_stream` with a supervisor loop.

Per active live mission, this owns:
  - the asyncio.Queue of pending operator events (registered in events.py);
  - the current MissionPlan (mutable — swapped on REPLAN / RTB / EMERGENCY_LAND);
  - the resume battery percentage across plan swaps.

For every Nth telemetry tick (configurable via DECIDE_EVERY_N_TICKS) OR every
tick that has a pending event, we drain the queue and invoke the supervisor
subgraph in a worker thread (it does sync DB reads). The decision is applied
to the live plan by terminating the current `telemetry_stream` generator and
starting a fresh one from the new plan + current battery state.

Decisions emit a separate "supervisor" frame so the UI can show a banner +
overlay the replaced plan without misinterpreting it as a telemetry update.
The frame schema stays flat JSON (`kind` discriminator) so the WebSocket
handler stays single-purpose.

See DESIGN_DECISIONS DD-014.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app.schemas.enums import LegType
from app.schemas.plan import MissionLeg, MissionPlan, Waypoint
from app.sim.stream import telemetry_stream
from app.supervisor.builder import build_supervisor
from app.supervisor.events import Event, register_queue, unregister_queue
from app.supervisor.state import SupervisorDecision
from app.validation.kernel import GeoContext
from app.validation.physics import DroneProfile, estimate_leg

log = logging.getLogger(__name__)

DECIDE_EVERY_N_TICKS = 5  # ~1 Hz at the default SIM_DT_S=0.2s
MAX_REPLANS_PER_SESSION = 3  # safety cap; mirrors compile-loop repair cap


def _build_rtb_plan(
    *,
    current_lat: float,
    current_lon: float,
    current_alt: float,
    home_lon: float,
    home_lat: float,
    ceiling_m: float,
    area_id: str,
    profile: DroneProfile,
) -> MissionPlan:
    """Construct a direct-to-home plan (TRANSIT + RETURN_TO_BASE) for RTB/divert."""
    cruise_alt = min(max(current_alt, 40.0), max(20.0, ceiling_m - 5.0))
    cur_wp = Waypoint(lat=current_lat, lon=current_lon, alt_m=cruise_alt)
    home_wp = Waypoint(lat=home_lat, lon=home_lon, alt_m=cruise_alt)
    landing_wp = Waypoint(lat=home_lat, lon=home_lon, alt_m=0.0)

    transit = MissionLeg(
        leg_type=LegType.TRANSIT,
        geometry=[cur_wp, home_wp],
        sensor_mode=None,
        est_duration_s=0.0,
        est_battery_pct=0.0,
    )
    rtb = MissionLeg(
        leg_type=LegType.RETURN_TO_BASE,
        geometry=[home_wp, landing_wp],
        sensor_mode=None,
        est_duration_s=0.0,
        est_battery_pct=0.0,
    )
    for leg in (transit, rtb):
        energy = estimate_leg(leg, profile)
        leg.est_duration_s = energy.duration_s
        leg.est_battery_pct = energy.battery_pct

    return MissionPlan(
        mission_id=f"rtb-{uuid.uuid4().hex[:8]}",
        area_id=area_id,
        status="READY_FOR_APPROVAL",
        legs=[transit, rtb],
        total_duration_s=transit.est_duration_s + rtb.est_duration_s,
        total_battery_pct=transit.est_battery_pct + rtb.est_battery_pct,
        battery_reserve_pct=0.0,
        reasoning_trace="Supervisor RTB: direct-to-home transit + landing.",
    )


def _build_emergency_land_plan(
    *,
    current_lat: float,
    current_lon: float,
    current_alt: float,
    area_id: str,
    profile: DroneProfile,
) -> MissionPlan:
    """Construct a descend-in-place plan: a LOITER at low altitude, then land."""
    hold_alt = 10.0
    descend_alt = 0.0
    top = Waypoint(lat=current_lat, lon=current_lon, alt_m=max(hold_alt, current_alt))
    hold = Waypoint(lat=current_lat, lon=current_lon, alt_m=hold_alt)
    ground = Waypoint(lat=current_lat, lon=current_lon, alt_m=descend_alt)

    descend = MissionLeg(
        leg_type=LegType.TRANSIT,
        geometry=[top, hold, ground],
        sensor_mode=None,
        est_duration_s=0.0,
        est_battery_pct=0.0,
    )
    rtb = MissionLeg(
        leg_type=LegType.RETURN_TO_BASE,
        geometry=[ground, ground],
        sensor_mode=None,
        est_duration_s=0.0,
        est_battery_pct=0.0,
    )
    for leg in (descend, rtb):
        energy = estimate_leg(leg, profile)
        leg.est_duration_s = max(energy.duration_s, 1.0)
        leg.est_battery_pct = energy.battery_pct

    return MissionPlan(
        mission_id=f"eland-{uuid.uuid4().hex[:8]}",
        area_id=area_id,
        status="READY_FOR_APPROVAL",
        legs=[descend, rtb],
        total_duration_s=descend.est_duration_s + rtb.est_duration_s,
        total_battery_pct=descend.est_battery_pct + rtb.est_battery_pct,
        battery_reserve_pct=0.0,
        reasoning_trace="Supervisor EMERGENCY_LAND: descend in place.",
    )


async def _drain_queue(q: asyncio.Queue[Event]) -> list[Event]:
    out: list[Event] = []
    while True:
        try:
            out.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return out


async def live_mission_session(
    *,
    mission_id: str,
    session_id: str,
    plan: MissionPlan,
    profile: DroneProfile,
    geo: GeoContext | None,
    area_id: str,
    drone_profile_id: str,
    starting_battery_pct: float = 100.0,
    speed: float = 10.0,
) -> AsyncIterator[dict[str, Any]]:
    """Yield framed dicts the WebSocket handler forwards verbatim.

    Frame shapes (all flat JSON, `kind` discriminator):
      - `{kind: "telemetry", t, lat, lon, alt_m, battery_pct, leg_idx,
         leg_type, sensor_mode, status, detail}`
      - `{kind: "supervisor", decision, reason, events, plan_swapped: bool,
         new_plan: <MissionPlan dict>|null}`

    The handler streams these directly. The client switches on `kind`.
    """
    q = register_queue(mission_id, session_id)
    supervisor = build_supervisor()
    home_lon = geo.home_point[0] if geo else 0.0
    home_lat = geo.home_point[1] if geo else 0.0
    ceiling_m = float(geo.altitude_ceiling_m) if geo else 120.0

    current_plan = plan
    starting_pct = float(starting_battery_pct)
    replans_used = 0
    tick = 0

    try:
        while True:
            stream = telemetry_stream(
                current_plan,
                profile,
                geo,
                starting_battery_pct=starting_pct,
                speed=speed,
            )
            swap_pending = None  # (decision, new_plan, reason, events)

            async for frame in stream:
                tick += 1
                f_dict = frame.to_dict()
                yield {"kind": "telemetry", **f_dict}
                if frame.status != "flying":
                    return  # terminal frame; session over

                # Decision gating: run supervisor every N ticks OR whenever events queued.
                if tick % DECIDE_EVERY_N_TICKS != 0 and q.empty():
                    continue

                events = await _drain_queue(q)
                # Skip a no-op tick: no events and the supervisor would just say CONTINUE.
                # (We still run periodically to catch the always-on battery margin check.)

                remaining_legs = [
                    leg.model_dump() for leg in current_plan.legs[frame.leg_idx + 1 :]
                ]
                state: dict[str, Any] = {
                    "mission_id": mission_id,
                    "area_id": area_id,
                    "drone_profile_id": drone_profile_id,
                    "current_telemetry": f_dict,
                    "remaining_legs": remaining_legs,
                    "home": (home_lon, home_lat),
                    "altitude_ceiling_m": ceiling_m,
                    "events": [e.to_dict() for e in events],
                }
                try:
                    # The supervisor does sync DB reads; off-thread to keep the
                    # event loop responsive.
                    out = await asyncio.to_thread(supervisor.invoke, state)
                except Exception as e:  # noqa: BLE001 — supervisor is additive.
                    log.exception("supervisor.invoke failed: %s", e)
                    if events:
                        yield {
                            "kind": "supervisor",
                            "decision": "CONTINUE",
                            "reason": f"supervisor error: {type(e).__name__}: {e}",
                            "events": [e.to_dict() for e in events],
                            "plan_swapped": False,
                            "new_plan": None,
                        }
                    continue

                decision = out.get("decision") or SupervisorDecision.CONTINUE.value
                reason = out.get("decision_reason") or ""

                if decision == SupervisorDecision.CONTINUE.value:
                    if events:
                        yield {
                            "kind": "supervisor",
                            "decision": decision,
                            "reason": reason,
                            "events": [e.to_dict() for e in events],
                            "plan_swapped": False,
                            "new_plan": None,
                        }
                    continue

                # Non-CONTINUE: build the replacement plan and prepare swap.
                if replans_used >= MAX_REPLANS_PER_SESSION:
                    yield {
                        "kind": "supervisor",
                        "decision": "CONTINUE",
                        "reason": (
                            f"max replans ({MAX_REPLANS_PER_SESSION}) reached; "
                            "ignoring further supervisor decisions this session"
                        ),
                        "events": [e.to_dict() for e in events],
                        "plan_swapped": False,
                        "new_plan": None,
                    }
                    continue

                if decision == SupervisorDecision.REPLAN_FROM_HERE.value:
                    new_plan_dict = out.get("replan_plan")
                    if new_plan_dict:
                        try:
                            new_plan = MissionPlan.model_validate(new_plan_dict)
                        except Exception as e:  # noqa: BLE001
                            log.exception("supervisor replan plan invalid: %s", e)
                            new_plan = _build_rtb_plan(
                                current_lat=f_dict["lat"],
                                current_lon=f_dict["lon"],
                                current_alt=f_dict["alt_m"],
                                home_lon=home_lon,
                                home_lat=home_lat,
                                ceiling_m=ceiling_m,
                                area_id=area_id,
                                profile=profile,
                            )
                            decision = SupervisorDecision.RTB_NOW.value
                            reason = f"replan plan invalid; falling back to RTB: {e}"
                    else:
                        new_plan = _build_rtb_plan(
                            current_lat=f_dict["lat"],
                            current_lon=f_dict["lon"],
                            current_alt=f_dict["alt_m"],
                            home_lon=home_lon,
                            home_lat=home_lat,
                            ceiling_m=ceiling_m,
                            area_id=area_id,
                            profile=profile,
                        )
                        decision = SupervisorDecision.RTB_NOW.value
                elif decision in (
                    SupervisorDecision.RTB_NOW.value,
                    SupervisorDecision.DIVERT_TO_SAFE_POINT.value,
                ):
                    new_plan = _build_rtb_plan(
                        current_lat=f_dict["lat"],
                        current_lon=f_dict["lon"],
                        current_alt=f_dict["alt_m"],
                        home_lon=home_lon,
                        home_lat=home_lat,
                        ceiling_m=ceiling_m,
                        area_id=area_id,
                        profile=profile,
                    )
                elif decision == SupervisorDecision.EMERGENCY_LAND.value:
                    new_plan = _build_emergency_land_plan(
                        current_lat=f_dict["lat"],
                        current_lon=f_dict["lon"],
                        current_alt=f_dict["alt_m"],
                        area_id=area_id,
                        profile=profile,
                    )
                else:
                    log.warning("unknown supervisor decision: %s", decision)
                    continue

                replans_used += 1
                yield {
                    "kind": "supervisor",
                    "decision": decision,
                    "reason": reason,
                    "events": [e.to_dict() for e in events],
                    "plan_swapped": True,
                    "new_plan": new_plan.model_dump(),
                    "replans_used": replans_used,
                }
                swap_pending = new_plan
                starting_pct = float(f_dict.get("battery_pct") or starting_pct)
                break  # exit inner async-for to swap the stream

            if swap_pending is None:
                # Stream ended naturally (or no swap requested).
                return
            current_plan = swap_pending
    finally:
        unregister_queue(mission_id, session_id)
