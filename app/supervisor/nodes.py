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


def replan_node(state: SupervisorState) -> dict[str, Any]:
    """Build a concrete replacement plan for REPLAN_FROM_HERE.

    Initial implementation: synthesize a direct-to-home plan (single TRANSIT
    leg from current position to home, with the kernel re-validating). This
    is the safe baseline — guaranteed to converge to a flyable plan as long
    as the drone has the battery to make it home.

    A richer "preserve original objective" replan (invoke the compile graph
    with a warm-start command) is a future extension; documented in
    DD-014's trade-off section.
    """
    try:
        cur = state["current_telemetry"]
        home_lon, home_lat = state["home"]
        ceiling = float(state.get("altitude_ceiling_m") or 120.0)

        # Pick a safe cruise altitude: max of (current alt, 40m, ceiling-20m floor).
        cruise_alt = min(max(float(cur.get("alt_m") or 60.0), 40.0), max(20.0, ceiling - 5.0))

        profile = load_drone_profile(get_engine(), state["drone_profile_id"])
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
            energy = estimate_leg(leg, profile)
            leg.est_duration_s = energy.duration_s
            leg.est_battery_pct = energy.battery_pct

        starting = float(cur.get("battery_pct") or 100.0)
        plan = MissionPlan(
            mission_id=f"replan-{uuid.uuid4().hex[:8]}",
            area_id=state["area_id"],
            status="READY_FOR_APPROVAL",  # the orchestrator skips approval for replans
            legs=[transit, rtb],
            total_duration_s=transit.est_duration_s + rtb.est_duration_s,
            total_battery_pct=transit.est_battery_pct + rtb.est_battery_pct,
            battery_reserve_pct=max(
                0.0, starting - (transit.est_battery_pct + rtb.est_battery_pct)
            ),
            reasoning_trace=("Supervisor-generated replan: direct-to-home from current position."),
        )
        return {"replan_plan": plan.model_dump()}
    except Exception as e:  # noqa: BLE001
        log.exception("supervisor replan_node failed: %s", e)
        # Downgrade to RTB_NOW (which the orchestrator handles without a plan).
        return {
            "replan_plan": None,
            "decision": SupervisorDecision.RTB_NOW.value,
            "decision_reason": f"replan failed, falling back to RTB: {e}",
            "error": str(e),
        }


def commit_node(state: SupervisorState) -> dict[str, Any]:  # noqa: ARG001 — graph terminus, signature required
    """Terminal node: passes the chosen decision through unchanged.

    Exists as an explicit graph terminus so future extensions (logging,
    metrics, persistence) have a single place to land.
    """
    return {}
