"""Deterministic fleet sizing for multi-drone missions.

The operator picks a drone profile and a bias slider (fewer drones vs faster
mission); this module decides HOW MANY of that drone to use. The fleet is
homogeneous (all the selected profile) and sized by a closed-form coverage
model on that profile's real endurance + cruise speed — no LLM call, so it's
cheap, instant, and reproducible. Downstream, `multi_drone.assign_slots`
assigns altitude layers + takeoff staggers and the compile graph plans one
drone at a time as before.
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger(__name__)

# Hard ceiling so a runaway estimate can't request 100 drones. Sensible upper
# bound for the kind of areas this tool handles.
MAX_FLEET_SIZE = 8


def _bbox_area_km2(boundary_lonlat: list[list[float]]) -> float:
    """Equirectangular approximation. Cheap, good enough for fleet-sizing."""
    if not boundary_lonlat:
        return 0.0
    lons = [p[0] for p in boundary_lonlat]
    lats = [p[1] for p in boundary_lonlat]
    if not lons or not lats:
        return 0.0
    lon_span = max(lons) - min(lons)
    lat_span = max(lats) - min(lats)
    mean_lat = (max(lats) + min(lats)) / 2.0
    # 111.32 km per degree latitude, 111.32 * cos(lat) per degree longitude
    cos_lat = max(0.01, abs(math.cos(math.radians(mean_lat))))
    km_lon = lon_span * 111.32 * cos_lat
    km_lat = lat_span * 111.32
    return max(0.0, km_lon * km_lat)


def pick_fleet(
    *,
    command: str,
    geo_context: dict[str, Any],
    profiles: list[dict[str, Any]],
    bias: float,
    preferred_profile_id: str | None = None,
    max_fleet: int = MAX_FLEET_SIZE,
) -> dict[str, Any]:
    """Size a HOMOGENEOUS fleet of the operator-selected drone, deterministically.

    Returns `{"drone_profile_ids": [...], "reasoning": str}`.

    Design (deliberately NOT an LLM call):
      - The fleet is N copies of `preferred_profile_id` — the drone the operator
        chose in the UI. Auto mode decides HOW MANY, not WHICH; mixing airframes
        confused operators and broke fleet sizing.
      - Sizing is a closed-form coverage model on THAT profile's real specs
        (endurance, cruise speed). A short-endurance drone therefore gets more
        drones (smaller bands) than a long-endurance one for the same area.
      - No model call: it's cheaper, instant, and reproducible. The deterministic
        kernel still has final say on whether each per-drone band is flyable.
    """
    if not profiles:
        raise ValueError("no drone profiles available")
    bias = max(0.0, min(1.0, float(bias)))
    max_fleet = max(1, min(MAX_FLEET_SIZE, int(max_fleet)))
    area_km2 = _bbox_area_km2(geo_context.get("boundary_lonlat") or [])

    # Resolve the operator's selected profile; fall back to longest-endurance.
    by_id = {p["profile_id"]: p for p in profiles}
    profile = by_id.get(preferred_profile_id or "") or max(
        profiles, key=lambda p: float(p.get("rated_endurance_s", 0) or 0)
    )
    pid = profile["profile_id"]
    cruise = float(profile.get("cruise_speed_mps", 12.0) or 12.0)
    endurance_s = float(profile.get("rated_endurance_s", 1800.0) or 1800.0)

    # Coverage missions scale with area; point/perimeter tasks stay at 1 drone.
    cmd_lc = command.lower()
    coverage_intent = any(
        k in cmd_lc
        for k in (
            "search", "sweep", "cover", "entire", "whole", "all of", "complete",
            "every", "as much", "patrol",
        )
    )
    # An explicit multi-drone ask forces a fleet (>=2) regardless of the coverage
    # heuristic — the operator literally asked for more than one airframe.
    wants_multi = any(
        k in cmd_lc
        for k in ("multiple drone", "multi drone", "several drone", "fleet", "swarm", "many drone")
    )

    if not ((coverage_intent or wants_multi) and area_km2 > 0):
        return {
            "drone_profile_ids": [pid],
            "reasoning": (
                f"1× {pid}: '{command[:40]}' is a point/perimeter task, not a "
                f"full-area sweep — one drone suffices."
            ),
        }

    # Closed-form coverage estimate. Conservative nominal swath (errs toward MORE
    # drones / smaller bands, which is the safe direction). usable = 70% of rated
    # endurance to reserve margin for transit-to-band + the 20% landing reserve.
    nominal_swath_m = 45.0
    sweep_distance_m = (area_km2 * 1_000_000.0) / nominal_swath_m
    sweep_time_s = sweep_distance_m / max(cruise, 1.0)
    usable_s = endurance_s * 0.7
    min_feasible = max(1, math.ceil(sweep_time_s / max(usable_s, 1.0)))

    # Bias multiplier: 0 → just feasible (fewest drones); 1 → ~2× for parallel
    # wall-clock speed-up. Always at least min_feasible so no band exceeds endurance.
    n = math.ceil(min_feasible * (1.0 + bias))
    n = max(min_feasible, min(n, max_fleet))
    # Honor an explicit "multiple drones" request even when the closed-form model
    # says one would suffice — the operator asked for a fleet.
    if wants_multi:
        n = max(n, 2)

    capped_note = ""
    if min_feasible > max_fleet:
        capped_note = (
            f" NOTE: covering {area_km2:.2f} km² within {pid}'s "
            f"{endurance_s / 60:.0f}-min endurance needs ~{min_feasible} drones, "
            f"above the {max_fleet}-drone cap — some bands may still be rejected. "
            f"Consider a longer-endurance drone."
        )

    log.info(
        "auto-fleet: %d× %s (bias=%.2f, area=%.3f km², min_feasible=%d)",
        n, pid, bias, area_km2, min_feasible,
    )
    return {
        "drone_profile_ids": [pid] * n,
        "reasoning": (
            f"{n}× {pid} to cover {area_km2:.2f} km². At {cruise:.0f} m/s with a "
            f"~{endurance_s / 60:.0f}-min endurance, each drone needs a band it can "
            f"sweep within range; {min_feasible} is the minimum feasible and the "
            f"speed bias ({bias:.2f}) raised it to {n}.{capped_note}"
        ),
    }


