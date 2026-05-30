"""Supervisor subgraph nodes.

Three nodes, one decision per invocation. The graph topology is:

    assess → decide → (replan | commit) → END

where `replan` is only entered when `decide` returns REPLAN_FROM_HERE; every
other decision goes straight to `commit`. The "commit" node materializes the
chosen action as a concrete plan mutation that the orchestrator then applies
to the live stream.

State threading: nodes read inputs (current_telemetry, remaining_legs, home,
events), produce outputs (decision, decision_reason, replan_plan). See
SupervisorState in `state.py`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.geo.store import get_engine, load_drone_profile
from app.schemas.enums import LegType
from app.schemas.plan import MissionLeg, MissionPlan, Waypoint
from app.supervisor.events import Event
from app.supervisor.policy import decide as policy_decide
from app.supervisor.state import SupervisorDecision, SupervisorState
from app.validation.physics import estimate_leg

log = logging.getLogger(__name__)


def assess_node(state: SupervisorState) -> dict[str, Any]:
    """Normalize inputs and short-circuit if there is nothing to decide.

    Pure read; no state mutation beyond passing inputs through.
    """
    # Defensive defaults so the rest of the graph doesn't have to guard.
    return {
        "events": list(state.get("events") or []),
        "remaining_legs": list(state.get("remaining_legs") or []),
    }


def decide_node(state: SupervisorState) -> dict[str, Any]:
    """Run the deterministic decision policy."""
    try:
        profile = load_drone_profile(get_engine(), state["drone_profile_id"])
        drone_profile_dict = {
            "cruise_speed_mps": profile.cruise_speed_mps,
            "cruise_power_w": profile.cruise_power_w,
            "battery_wh": profile.battery_wh,
        }
        events = [Event.from_dict(e) for e in (state.get("events") or [])]
        decision, reason = policy_decide(
            current=state["current_telemetry"],
            remaining_legs=state.get("remaining_legs") or [],
            home_lon=state["home"][0],
            home_lat=state["home"][1],
            pending_events=events,
            drone_profile=drone_profile_dict,
        )
        return {"decision": decision.value, "decision_reason": reason}
    except Exception as e:  # noqa: BLE001 — supervisor is additive; fall safe.
        log.exception("supervisor decide_node failed: %s", e)
        return {
            "decision": SupervisorDecision.CONTINUE.value,
            "decision_reason": f"supervisor error: {type(e).__name__}: {e}",
            "error": str(e),
        }


def _direct_home_plan(
    cur: dict[str, Any],
    home_lon: float,
    home_lat: float,
    ceiling: float,
    area_id: str,
    profile: Any,
) -> MissionPlan:
    """Fallback: a direct-to-home TRANSIT + landing plan."""
    cruise_alt = min(max(float(cur.get("alt_m") or 60.0), 40.0), max(20.0, ceiling - 5.0))
    cur_wp = Waypoint(lat=float(cur["lat"]), lon=float(cur["lon"]), alt_m=cruise_alt)
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
        e = estimate_leg(leg, profile)
        leg.est_duration_s, leg.est_battery_pct = e.duration_s, e.battery_pct
    starting = float(cur.get("battery_pct") or 100.0)
    return MissionPlan(
        mission_id=f"replan-{uuid.uuid4().hex[:8]}",
        area_id=area_id,
        status="READY_FOR_APPROVAL",
        legs=[transit, rtb],
        total_duration_s=transit.est_duration_s + rtb.est_duration_s,
        total_battery_pct=transit.est_battery_pct + rtb.est_battery_pct,
        battery_reserve_pct=max(0.0, starting - (transit.est_battery_pct + rtb.est_battery_pct)),
        reasoning_trace="Supervisor replan: direct-to-home from current position.",
    )


def replan_node(state: SupervisorState) -> dict[str, Any]:
    """Reroute the REMAINING mission around a pop-up NFZ so the mission continues.

    This is the real "replan around it" behavior: rather than abandoning the
    mission and flying home, we keep every remaining leg (search pattern, etc.)
    but bend its geometry around the new restricted-airspace polygon using the
    same deterministic visibility-graph router the planner uses. The drone
    visibly skirts the new zone and carries on.

    Falls back to a direct-to-home plan if there's no usable NFZ polygon in the
    events, if rerouting fails, or if the remaining plan is empty.
    """
    cur = state["current_telemetry"]
    home_lon, home_lat = state["home"]
    ceiling = float(state.get("altitude_ceiling_m") or 120.0)
    try:
        profile = load_drone_profile(get_engine(), state["drone_profile_id"])
    except Exception as e:  # noqa: BLE001
        log.exception("replan_node: profile load failed: %s", e)
        return {
            "replan_plan": None,
            "decision": SupervisorDecision.RTB_NOW.value,
            "decision_reason": f"replan failed, falling back to RTB: {e}",
            "error": str(e),
        }

    # Extract the pop-up NFZ polygon (if any) from the triggering events.
    from shapely.geometry import shape as _shape  # noqa: PLC0415

    from app.geo.patterns import METERS_PER_DEG_LAT, _route_through_flyable  # noqa: PLC0415

    nfz_poly = None
    for ed in state.get("events") or []:
        if ed.get("type") == "NFZ_POPUP":
            try:
                cand = _shape((ed.get("payload") or {}).get("polygon") or {})
                if cand.is_valid and not cand.is_empty:
                    nfz_poly = cand
                    break
            except Exception:  # noqa: BLE001
                continue

    remaining = state.get("remaining_legs") or []
    try:
        if nfz_poly is None or not remaining:
            raise ValueError("no NFZ polygon or no remaining legs — using direct home")

        # Buffer the NFZ by ~20m so the reroute keeps a margin.
        import math as _m  # noqa: PLC0415

        margin_deg = (
            20.0
            * (
                1.0 / METERS_PER_DEG_LAT
                + 1.0 / max(METERS_PER_DEG_LAT * _m.cos(_m.radians(home_lat)), 1e-6)
            )
            / 2.0
        )
        nfz_buf = nfz_poly.buffer(margin_deg)
        obstacles = [nfz_buf]

        cruise_alt = min(max(float(cur.get("alt_m") or 60.0), 40.0), max(20.0, ceiling - 5.0))
        legs: list[MissionLeg] = []
        prev_pt = (float(cur["lon"]), float(cur["lat"]))

        for leg_d in remaining:
            ltype = leg_d.get("leg_type") or "TRANSIT"
            sensor = leg_d.get("sensor_mode")
            geom = leg_d.get("geometry") or []
            if not geom:
                continue
            pts = [(g["lon"], g["lat"], g.get("alt_m", cruise_alt)) for g in geom]
            # Prepend current position as the entry into this leg's first point.
            new_wps: list[Waypoint] = []
            cursor = prev_pt
            for lon, lat, alt in pts:
                detour = _route_through_flyable(cursor, (lon, lat), obstacles)
                for dlon, dlat in detour:
                    new_wps.append(Waypoint(lat=dlat, lon=dlon, alt_m=alt))
                new_wps.append(Waypoint(lat=lat, lon=lon, alt_m=alt))
                cursor = (lon, lat)
            prev_pt = cursor
            leg = MissionLeg(
                leg_type=LegType(ltype),
                geometry=new_wps,
                sensor_mode=sensor,
                pattern_name=leg_d.get("pattern_name"),
                est_duration_s=0.0,
                est_battery_pct=0.0,
            )
            energy = estimate_leg(leg, profile)
            leg.est_duration_s, leg.est_battery_pct = energy.duration_s, energy.battery_pct
            legs.append(leg)

        if not legs:
            raise ValueError("rerouted plan empty")
        # Guarantee the plan still ends with an RTB leg.
        if legs[-1].leg_type != LegType.RETURN_TO_BASE:
            home_wp = Waypoint(lat=home_lat, lon=home_lon, alt_m=cruise_alt)
            land_wp = Waypoint(lat=home_lat, lon=home_lon, alt_m=0.0)
            last = legs[-1].geometry[-1]
            detour = _route_through_flyable((last.lon, last.lat), (home_lon, home_lat), obstacles)
            rtb_geo = [Waypoint(lat=last.lat, lon=last.lon, alt_m=cruise_alt)]
            rtb_geo += [Waypoint(lat=dl, lon=dn, alt_m=cruise_alt) for dn, dl in detour]
            rtb_geo += [home_wp, land_wp]
            rtb = MissionLeg(
                leg_type=LegType.RETURN_TO_BASE,
                geometry=rtb_geo,
                sensor_mode=None,
                est_duration_s=0.0,
                est_battery_pct=0.0,
            )
            energy = estimate_leg(rtb, profile)
            rtb.est_duration_s, rtb.est_battery_pct = energy.duration_s, energy.battery_pct
            legs.append(rtb)

        starting = float(cur.get("battery_pct") or 100.0)
        total_dur = sum(leg.est_duration_s for leg in legs)
        total_batt = sum(leg.est_battery_pct for leg in legs)
        plan = MissionPlan(
            mission_id=f"replan-{uuid.uuid4().hex[:8]}",
            area_id=state["area_id"],
            status="READY_FOR_APPROVAL",
            legs=legs,
            total_duration_s=total_dur,
            total_battery_pct=total_batt,
            battery_reserve_pct=max(0.0, starting - total_batt),
            reasoning_trace=(
                "Supervisor replan: remaining mission rerouted around pop-up "
                "restricted airspace; objective preserved."
            ),
        )
        return {"replan_plan": plan.model_dump()}
    except Exception as e:  # noqa: BLE001
        log.info("replan_node: rerouting around NFZ not possible (%s); direct-to-home", e)
        try:
            plan = _direct_home_plan(cur, home_lon, home_lat, ceiling, state["area_id"], profile)
            return {"replan_plan": plan.model_dump()}
        except Exception as e2:  # noqa: BLE001
            log.exception("replan_node fallback failed: %s", e2)
            return {
                "replan_plan": None,
                "decision": SupervisorDecision.RTB_NOW.value,
                "decision_reason": f"replan failed, falling back to RTB: {e2}",
                "error": str(e2),
            }


def commit_node(state: SupervisorState) -> dict[str, Any]:  # noqa: ARG001 — graph terminus, signature required
    """Terminal node: passes the chosen decision through unchanged.

    Exists as an explicit graph terminus so future extensions (logging,
    metrics, persistence) have a single place to land.
    """
    return {}
