"""Supervisor state object + decision enum.

The supervisor subgraph runs *per decision*, not as a long-running checkpointed
graph (see DD-014). Each tick that has an event (or that hits a periodic
re-assess threshold) instantiates a fresh `SupervisorState`, runs assess →
decide → (maybe replan) → commit, and returns the outcome.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict


class SupervisorDecision(StrEnum):
    """One of five terminal decisions per supervisor invocation.

    The orchestrator translates each decision into a concrete plan mutation:
      - CONTINUE: no-op, keep streaming the current plan.
      - RTB_NOW: replace the remainder with a single direct-to-home TRANSIT
        leg, then end the mission.
      - DIVERT_TO_SAFE_POINT: same shape as RTB_NOW for now (home is the
        safe point); kept distinct so the UI can show different banner copy.
      - REPLAN_FROM_HERE: reroute the REMAINING legs around an obstacle (e.g. a
        pop-up NFZ) so the mission continues; swap the active plan and resume.
      - LOITER_RTB: hold position briefly (loiter), then fly home — the
        standard lost-link / lost-C2 contingency.
      - EMERGENCY_LAND: replace the remainder with a descend-in-place LOITER
        at low altitude, then end the mission.
    """

    CONTINUE = "CONTINUE"
    RTB_NOW = "RTB_NOW"
    DIVERT_TO_SAFE_POINT = "DIVERT_TO_SAFE_POINT"
    REPLAN_FROM_HERE = "REPLAN_FROM_HERE"
    LOITER_RTB = "LOITER_RTB"
    EMERGENCY_LAND = "EMERGENCY_LAND"


class SupervisorState(TypedDict, total=False):
    """One-shot state object for a single supervisor invocation.

    Inputs are populated by the orchestrator before invoking the subgraph;
    outputs are filled in by `decide_node` and (if applicable) `replan_node`.
    """

    # Inputs.
    mission_id: str
    area_id: str
    drone_profile_id: str
    current_telemetry: dict[str, Any]  # t, lat, lon, alt_m, battery_pct, leg_idx, ...
    remaining_legs: list[dict[str, Any]]  # serialized MissionLeg dicts after current leg
    home: tuple[float, float]  # (lon, lat)
    altitude_ceiling_m: float
    events: list[dict[str, Any]]  # Event.to_dict() list — usually 0–2 per tick

    # Outputs.
    decision: str  # SupervisorDecision value
    decision_reason: str  # human-readable why
    replan_plan: dict[str, Any] | None  # MissionPlan dict if REPLAN_FROM_HERE
    error: str | None
