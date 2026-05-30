"""Multi-drone coordination helpers.

See docs/DESIGN_DECISIONS.md §9. Server prescribes altitude layers and takeoff
staggers; the planner LLM handles the spatial decomposition (which third of
the field a drone covers).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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


def _lonlat_distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance for (lon, lat) tuples."""
    lon1, lat1 = a
    lon2, lat2 = b
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    h = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * 6_371_000.0 * math.asin(math.sqrt(h))


def nearest_home_base(
    home_bases: Sequence[Sequence[float]], target_lonlat: tuple[float, float]
) -> tuple[float, float]:
    """Pick the registered base nearest to a drone's assigned band centroid."""
    bases = [(float(b[0]), float(b[1])) for b in home_bases if len(b) >= 2]
    if not bases:
        raise ValueError("at least one home base is required")
    return min(bases, key=lambda base: _lonlat_distance_m(base, target_lonlat))


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


def assign_slots(drone_ids: list[str], ceiling_m: float | None = None) -> list[DroneSlot]:
    """One DroneSlot per drone with non-overlapping altitude bands + staggered
    takeoffs. The planner LLM is told its slot index/band; the safety kernel
    (and deconfliction) enforces the rules.

    When `ceiling_m` is given and the default vertical layers ([40, 70, 100]m)
    would exceed it, the layers are compressed to fit within
    [MIN_AGL_M + margin, ceiling - margin]. This is safe because drones are ALSO
    spatially separated (each sweeps a different latitude band), so they never
    occupy the same airspace even when their altitude bands are close. Without
    this, low-ceiling areas (e.g. a 50m cap) would reject every drone above the
    first layer.
    """
    n = len(drone_ids)
    centers = [_layer_center(i) for i in range(n)]
    top = max(centers) + LAYER_HALF_BAND
    if ceiling_m is not None and top > ceiling_m:
        # Compress all bands to fit under the ceiling. Floor at 25m AGL.
        lo_bound = 25.0 + LAYER_HALF_BAND
        hi_bound = ceiling_m - LAYER_HALF_BAND
        if hi_bound <= lo_bound:
            # Ceiling so low that bands can't separate vertically — stack all
            # drones at one safe altitude; spatial separation keeps them apart.
            mid = max(25.0, min(ceiling_m - 5.0, (ceiling_m + 20.0) / 2.0))
            centers = [mid] * n
        elif n == 1:
            centers = [(lo_bound + hi_bound) / 2.0]
        else:
            step = (hi_bound - lo_bound) / (n - 1)
            centers = [lo_bound + i * step for i in range(n)]
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
