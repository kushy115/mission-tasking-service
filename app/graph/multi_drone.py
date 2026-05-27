"""Multi-drone coordination helpers.

See docs/DESIGN_DECISIONS.md §9. Server prescribes altitude layers and takeoff
staggers; the planner LLM handles the spatial decomposition (which third of
the field a drone covers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 30m between adjacent layers — matches VERTICAL_SEP_M in deconfliction.
MIN_VERTICAL_SEP_M = 30.0

# Default layer assignments. Keep the lowest layer above the kernel's
# min AGL (20m) with margin. For N > 3 we extrapolate at 30m intervals.
DEFAULT_LAYERS_M = [40.0, 70.0, 100.0]
LAYER_HALF_BAND = 15.0  # ±15m around the center; sum = 30m

# Per-drone takeoff offset so the group doesn't collide at the home point.
TAKEOFF_STAGGER_S = 30.0


@dataclass(frozen=True)
class DroneSlot:
    """Per-drone scheduling assignment used to brief the planner."""

    index: int
    total: int
    alt_min: float
    alt_max: float
    takeoff_offset_s: float
    sibling_bands: list[tuple[float, float]]
    drone_profile_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "total": self.total,
            "alt_min": self.alt_min,
            "alt_max": self.alt_max,
            "takeoff_offset_s": self.takeoff_offset_s,
            "sibling_bands": [f"{lo:.0f}-{hi:.0f}m" for lo, hi in self.sibling_bands],
            "drone_profile_id": self.drone_profile_id,
        }


def _layer_center(i: int) -> float:
    if i < len(DEFAULT_LAYERS_M):
        return DEFAULT_LAYERS_M[i]
    # Continue the 30m progression
    return DEFAULT_LAYERS_M[-1] + (i - len(DEFAULT_LAYERS_M) + 1) * MIN_VERTICAL_SEP_M


def assign_slots(drone_ids: list[str]) -> list[DroneSlot]:
    """One DroneSlot per drone with non-overlapping altitude bands + staggered
    takeoffs. The planner LLM is told its slot index/band; the safety kernel
    (and deconfliction) enforces the rules.
    """
    n = len(drone_ids)
    centers = [_layer_center(i) for i in range(n)]
    bands = [(c - LAYER_HALF_BAND, c + LAYER_HALF_BAND) for c in centers]
    slots: list[DroneSlot] = []
    for i, did in enumerate(drone_ids):
        siblings = [b for j, b in enumerate(bands) if j != i]
        slots.append(
            DroneSlot(
                index=i,
                total=n,
                alt_min=bands[i][0],
                alt_max=bands[i][1],
                takeoff_offset_s=i * TAKEOFF_STAGGER_S,
                sibling_bands=siblings,
                drone_profile_id=did,
            )
        )
    return slots
