"""Deterministic flight-physics / battery-energy model.

Assumptions (deliberately simple, principled, documented):
- Earth is a sphere of radius 6_371_000 m for haversine distance.
- Horizontal motion happens at the drone profile's `cruise_speed_mps`.
- Vertical motion (climb or descent) happens at `climb_rate_mps`. Descent is
  modeled symmetrically — we conservatively bill descent the same hover-equivalent
  power as climb to avoid implying free energy.
- Power draw has two components:
    * `cruise_power_w`     during transit legs (motors at cruise),
    * `hover_power_w`      during loiter/sensor-task/search-pattern legs,
  with an additive `sensor_power_w` from the active sensor (EO/IR) drawn on
  every leg where the sensor is on.
- Battery capacity is `battery_wh` from the profile; battery percent is
  energy_used / battery_wh * 100.
- This is NOT aerospace grade. It is principled, consistent, and reproducible —
  the safety kernel uses it for budget checks against the rated endurance.

This module has NO LLM calls and NO network access. Pure functions only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas.enums import LegType, SensorMode
from app.schemas.plan import MissionLeg, Waypoint

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class SensorSpec:
    name: str
    mode: SensorMode
    power_w: float
    # Half-angle of the sensor's field of view in degrees. Used to compute
    # ground swath = 2 * alt * tan(half_angle).
    half_angle_deg: float
    # Ground resolution at 100 m AGL, meters/pixel (linear in altitude).
    ground_resolution_at_100m: float


@dataclass(frozen=True)
class DroneProfile:
    """Capability profile loaded from data/drones/*.yaml."""

    profile_id: str
    rated_endurance_s: float
    cruise_speed_mps: float
    climb_rate_mps: float
    battery_wh: float
    cruise_power_w: float
    hover_power_w: float
    sensors: tuple[SensorSpec, ...]

    def sensor(self, mode: SensorMode) -> SensorSpec | None:
        for s in self.sensors:
            if s.mode == mode:
                return s
        return None


@dataclass(frozen=True)
class LegEnergy:
    distance_m: float
    horizontal_time_s: float
    vertical_time_s: float
    duration_s: float
    energy_wh: float
    battery_pct: float


def haversine_m(a: Waypoint, b: Waypoint) -> float:
    """Great-circle distance between two waypoints, ignoring altitude."""
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)
    s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(s))


def leg_distance_m(leg: MissionLeg) -> float:
    pts = leg.geometry
    return sum(haversine_m(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _sensor_draw_w(profile: DroneProfile, mode: SensorMode | None) -> float:
    if mode is None or mode == SensorMode.OFF:
        return 0.0
    spec = profile.sensor(mode)
    return spec.power_w if spec else 0.0


def estimate_leg(
    leg: MissionLeg,
    profile: DroneProfile,
    *,
    wind_mps: float = 0.0,
    wind_coeff: float = 0.0,
) -> LegEnergy:
    """Compute time + energy + battery_pct for a single leg.

    Wind penalty: avg_power_w *= (1 + wind_coeff * wind_mps). Defaults to zero
    so the existing physics behavior is unchanged when no weather is supplied.
    See docs/DESIGN_DECISIONS.md §4.
    """
    pts = leg.geometry
    dist_m = leg_distance_m(leg)
    horizontal_time_s = dist_m / profile.cruise_speed_mps if profile.cruise_speed_mps > 0 else 0.0

    # vertical: sum |delta alt| between consecutive points
    vert_m = sum(abs(pts[i + 1].alt_m - pts[i].alt_m) for i in range(len(pts) - 1))
    vertical_time_s = vert_m / profile.climb_rate_mps if profile.climb_rate_mps > 0 else 0.0

    sensor_w = _sensor_draw_w(profile, leg.sensor_mode)

    if leg.leg_type in (LegType.TRANSIT, LegType.RETURN_TO_BASE):
        # Cruise motion dominates; sensor adder is small.
        duration_s = horizontal_time_s + vertical_time_s
        avg_power_w = profile.cruise_power_w + sensor_w
    elif leg.leg_type == LegType.LOITER:
        # Stationary hover for est_duration_s if author provided; else compute
        # from geometry as time-to-traverse.
        duration_s = max(leg.est_duration_s, horizontal_time_s + vertical_time_s)
        avg_power_w = profile.hover_power_w + sensor_w
    else:
        # SEARCH_PATTERN, SENSOR_TASK: slow path with sensor on
        duration_s = horizontal_time_s + vertical_time_s
        # Search patterns fly at cruise but hold for sensor stability; bill the
        # average of cruise and hover.
        avg_power_w = (profile.cruise_power_w + profile.hover_power_w) / 2.0 + sensor_w

    if wind_mps > 0 and wind_coeff > 0:
        avg_power_w *= 1.0 + wind_coeff * wind_mps

    energy_wh = (avg_power_w * duration_s) / 3600.0
    battery_pct = (energy_wh / profile.battery_wh) * 100.0 if profile.battery_wh > 0 else 0.0
    return LegEnergy(
        distance_m=dist_m,
        horizontal_time_s=horizontal_time_s,
        vertical_time_s=vertical_time_s,
        duration_s=duration_s,
        energy_wh=energy_wh,
        battery_pct=battery_pct,
    )


@dataclass(frozen=True)
class MissionEnergy:
    legs: tuple[LegEnergy, ...]
    total_duration_s: float
    total_energy_wh: float
    total_battery_pct: float


def estimate_mission(
    legs: list[MissionLeg],
    profile: DroneProfile,
    *,
    wind_mps: float = 0.0,
    wind_coeff: float = 0.0,
) -> MissionEnergy:
    estimates = tuple(
        estimate_leg(leg, profile, wind_mps=wind_mps, wind_coeff=wind_coeff) for leg in legs
    )
    total_duration = sum(e.duration_s for e in estimates)
    total_energy = sum(e.energy_wh for e in estimates)
    total_pct = sum(e.battery_pct for e in estimates)
    return MissionEnergy(
        legs=estimates,
        total_duration_s=total_duration,
        total_energy_wh=total_energy,
        total_battery_pct=total_pct,
    )


def sensor_swath_m(sensor: SensorSpec, altitude_m: float) -> float:
    """Ground swath width visible to the sensor at the given AGL altitude."""
    return 2.0 * altitude_m * math.tan(math.radians(sensor.half_angle_deg))


def sensor_ground_resolution_m_per_px(sensor: SensorSpec, altitude_m: float) -> float:
    """Linear in altitude relative to the 100m calibration value."""
    return sensor.ground_resolution_at_100m * (altitude_m / 100.0)
