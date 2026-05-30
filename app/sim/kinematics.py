"""Kinematic flight-trajectory model (DD-015).

A standard 1-D trajectory time-parameterization: the drone is treated as a point
mass moving along the plan's polyline. Unlike the closed-form estimator in
`physics.py` — which assumes the drone teleports to cruise speed and rounds every
corner for free — this integrator respects the airframe's real limits, taken from
the drone profile the operator authored:

  * linear acceleration  (`profile.max_accel_mps2`)  — ramp up / brake down;
  * cornering            (derived from heading change) — slow into sharp turns;
  * climb rate           (`profile.climb_rate_mps`)   — steep legs are climb-bound;
  * a wind triangle       — the along-track wind component speeds up tailwind legs
    and slows + adds power on headwind legs (uses wind *direction*, which the
    scalar penalty in `physics.py` throws away).

It emits a per-tick trajectory trace — the "graph of where the drone actually
goes" — that both the execution verifier (`sim/executor.py`) and the live-sim
stream (`sim/stream.py`) consume. Because the model is INDEPENDENT of the
planner's estimate, the verifier can now legitimately disagree with the plan,
which is the whole point of a verifier (see DD-015 for the rationale).

Pure & deterministic: no LLM, no network, no sleep. Same inputs → same trace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas.enums import LegType, SensorMode
from app.schemas.plan import MissionPlan, Waypoint
from app.validation.physics import DroneProfile, haversine_m

# Minimum airspeed we let the integrator command, so a tick always makes
# progress and the per-segment loop terminates even on degenerate geometry.
_V_FLOOR_MPS = 0.5
# A turn that reverses direction can't be flown at cruise; this is the floor
# fraction of cruise speed we allow through the sharpest corner.
_CORNER_MIN_FRAC = 0.1
# Hard cap on ticks per segment — a safety net against pathological geometry.
_MAX_TICKS_PER_SEG = 200_000


@dataclass(frozen=True)
class KinSample:
    """One tick of the simulated flight."""

    t: float  # simulated elapsed seconds since takeoff
    lat: float
    lon: float
    alt_m: float
    airspeed_mps: float  # commanded airspeed (what the motors fight for)
    ground_speed_mps: float  # airspeed +/- along-track wind
    leg_idx: int
    leg_type: str
    sensor_mode: str | None
    battery_used_pct: float  # cumulative % of pack consumed by this tick


@dataclass(frozen=True)
class KinTrace:
    samples: tuple[KinSample, ...]
    total_duration_s: float
    total_battery_pct: float  # total % consumed across the mission


# --------------------------------------------------------------------------- #
# Motion primitives: the plan is flattened into an ordered list of MOVE / HOVER
# steps, then time-parameterized.
# --------------------------------------------------------------------------- #
@dataclass
class _Move:
    a: Waypoint
    b: Waypoint
    leg_idx: int
    leg_type: LegType
    sensor_mode: SensorMode | None
    heading_deg: float
    horiz_m: float
    dz_m: float


@dataclass
class _Hover:
    wp: Waypoint
    duration_s: float
    leg_idx: int
    leg_type: LegType
    sensor_mode: SensorMode | None


def _heading_deg(a: Waypoint, b: Waypoint) -> float:
    """Track heading a->b in degrees (0=N, 90=E), equirectangular approximation."""
    lat_mid = math.radians((a.lat + b.lat) / 2.0)
    dx = math.radians(b.lon - a.lon) * math.cos(lat_mid)
    dy = math.radians(b.lat - a.lat)
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _turn_angle_deg(h1: float, h2: float) -> float:
    """Absolute heading change between two segments, 0..180."""
    d = abs(h2 - h1) % 360.0
    return 360.0 - d if d > 180.0 else d


def _corner_speed(v_cruise: float, turn_deg: float) -> float:
    """Max pass-through airspeed for a turn of `turn_deg`.

    cos(turn/2): straight (0 deg) -> full cruise; 90 deg -> ~0.71x; 180 deg ->
    near stop. A recognised, simple cornering heuristic — no arc radius needed.
    """
    frac = max(math.cos(math.radians(turn_deg / 2.0)), _CORNER_MIN_FRAC)
    return v_cruise * frac


def _wind_along_mps(track_deg: float, wind_from_deg: float, wind_mps: float) -> float:
    """Along-track wind component. Positive = tailwind, negative = headwind.

    Wind *blows toward* (wind_from + 180). Projecting that onto the track gives
    the helping (+) or hindering (-) component the wind triangle resolves.
    """
    wind_to = (wind_from_deg + 180.0) % 360.0
    return wind_mps * math.cos(math.radians(track_deg - wind_to))


def _regime_power_w(leg_type: LegType, profile: DroneProfile) -> float:
    """Base airframe power by leg regime — mirrors physics.estimate_leg."""
    if leg_type in (LegType.TRANSIT, LegType.RETURN_TO_BASE):
        return profile.cruise_power_w
    if leg_type == LegType.LOITER:
        return profile.hover_power_w
    # SEARCH_PATTERN / SENSOR_TASK: fly slow with the sensor stabilised.
    return (profile.cruise_power_w + profile.hover_power_w) / 2.0


def _sensor_w(profile: DroneProfile, mode: SensorMode | None) -> float:
    if mode is None or mode == SensorMode.OFF:
        return 0.0
    spec = profile.sensor(mode)
    return spec.power_w if spec else 0.0


def _interp(a: Waypoint, b: Waypoint, frac: float) -> tuple[float, float, float]:
    return (
        a.lat + (b.lat - a.lat) * frac,
        a.lon + (b.lon - a.lon) * frac,
        a.alt_m + (b.alt_m - a.alt_m) * frac,
    )


def _build_steps(plan: MissionPlan) -> list[_Move | _Hover]:
    """Flatten the plan's legs into ordered MOVE / HOVER primitives."""
    steps: list[_Move | _Hover] = []
    for li, leg in enumerate(plan.legs):
        geom = leg.geometry
        if len(geom) == 1:
            # A single-point leg is a stationary task (loiter / point sensor).
            steps.append(
                _Hover(
                    wp=geom[0],
                    duration_s=max(leg.est_duration_s, 1.0),
                    leg_idx=li,
                    leg_type=leg.leg_type,
                    sensor_mode=leg.sensor_mode,
                )
            )
            continue
        for a, b in zip(geom[:-1], geom[1:], strict=True):
            horiz = haversine_m(a, b)
            steps.append(
                _Move(
                    a=a,
                    b=b,
                    leg_idx=li,
                    leg_type=leg.leg_type,
                    sensor_mode=leg.sensor_mode,
                    heading_deg=_heading_deg(a, b),
                    horiz_m=horiz,
                    dz_m=b.alt_m - a.alt_m,
                )
            )
    return steps


def simulate_trajectory(
    plan: MissionPlan,
    profile: DroneProfile,
    *,
    wind_mps: float = 0.0,
    wind_from_deg: float = 0.0,
    wind_coeff: float = 0.0,
    dt: float = 0.5,
) -> KinTrace:
    """Time-parameterize the plan into a kinematic trajectory trace.

    `dt` is the integration tick (seconds of simulated time). The verifier uses a
    coarse dt for speed; the live stream uses a fine dt for smooth animation.
    """
    steps = _build_steps(plan)
    moves = [s for s in steps if isinstance(s, _Move)]
    # Index each MOVE within the moves-only list so we can look at neighbours for
    # cornering. HOVER steps reset entry/exit speed to zero (the drone stops).
    move_pos = {id(m): i for i, m in enumerate(moves)}

    v_cruise = max(profile.cruise_speed_mps, _V_FLOOR_MPS)
    accel = max(profile.max_accel_mps2, 0.1)
    climb_rate = max(profile.climb_rate_mps, 0.1)
    batt_wh = profile.battery_wh if profile.battery_wh > 0 else 1.0

    samples: list[KinSample] = []
    t = 0.0
    batt_used = 0.0

    def emit(
        wp_lat: float,
        wp_lon: float,
        wp_alt: float,
        airspeed: float,
        ground_speed: float,
        step: _Move | _Hover,
    ) -> None:
        samples.append(
            KinSample(
                t=t,
                lat=wp_lat,
                lon=wp_lon,
                alt_m=wp_alt,
                airspeed_mps=airspeed,
                ground_speed_mps=ground_speed,
                leg_idx=step.leg_idx,
                leg_type=step.leg_type.value,
                sensor_mode=step.sensor_mode.value if step.sensor_mode else None,
                battery_used_pct=batt_used,
            )
        )

    for step in steps:
        if isinstance(step, _Hover):
            power = _regime_power_w(step.leg_type, profile) + _sensor_w(profile, step.sensor_mode)
            ticks = max(1, int(math.ceil(step.duration_s / dt)))
            for _ in range(ticks):
                t += dt
                batt_used += power * dt / 3600.0 / batt_wh * 100.0
                emit(step.wp.lat, step.wp.lon, step.wp.alt_m, 0.0, 0.0, step)
            continue

        mv = step
        # Vertical-only move (e.g. a pure climb leg): no horizontal distance, so
        # the trapezoid logic doesn't apply — climb at the rated vertical rate.
        if mv.horiz_m < 1.0 and abs(mv.dz_m) > 0.0:
            climb_time = abs(mv.dz_m) / climb_rate
            power = profile.hover_power_w + _sensor_w(profile, mv.sensor_mode)
            ticks = max(1, int(math.ceil(climb_time / dt)))
            for k in range(ticks):
                frac = (k + 1) / ticks
                lat, lon, alt = _interp(mv.a, mv.b, frac)
                t += dt
                batt_used += power * dt / 3600.0 / batt_wh * 100.0
                emit(lat, lon, alt, 0.0, 0.0, mv)
            continue

        if mv.horiz_m <= 0.0:
            continue

        idx = move_pos[id(mv)]
        prev_m = moves[idx - 1] if idx > 0 else None
        next_m = moves[idx + 1] if idx + 1 < len(moves) else None
        # Entry/exit corner speeds: 0 at the very start/end and wherever a HOVER
        # breaks the chain (neighbour absent in the moves list at this position).
        v_in = (
            _corner_speed(v_cruise, _turn_angle_deg(prev_m.heading_deg, mv.heading_deg))
            if prev_m
            else 0.0
        )
        v_out = (
            _corner_speed(v_cruise, _turn_angle_deg(mv.heading_deg, next_m.heading_deg))
            if next_m
            else 0.0
        )
        v_in = min(v_in, v_cruise)
        v_out = min(v_out, v_cruise)

        slope = abs(mv.dz_m) / mv.horiz_m
        # Climb-rate ceiling on horizontal speed: ground_rate * slope <= climb_rate.
        v_climb_cap = climb_rate / slope if slope > 0 else v_cruise
        v_cap = min(v_cruise, v_climb_cap)

        wind_along = _wind_along_mps(mv.heading_deg, wind_from_deg, wind_mps)
        headwind = max(0.0, -wind_along)
        power = (_regime_power_w(mv.leg_type, profile) + _sensor_w(profile, mv.sensor_mode)) * (
            1.0 + wind_coeff * headwind
        )

        v = min(v_in, v_cap) if v_in > 0 else _V_FLOOR_MPS
        s = 0.0
        ticks = 0
        while s < mv.horiz_m - 1e-6 and ticks < _MAX_TICKS_PER_SEG:
            ticks += 1
            d_rem = mv.horiz_m - s
            # Fastest we can be going now and still brake to v_out by segment end.
            v_brake = math.sqrt(max(v_out * v_out + 2.0 * accel * d_rem, 0.0))
            v_target = min(v_cap, v_brake)
            # Accelerate or brake toward the target, capped by the accel limit.
            v = min(v + accel * dt, v_target) if v < v_target else max(v - accel * dt, v_target)
            v = max(v, _V_FLOOR_MPS)

            ground_speed = max(0.2, v + wind_along)
            ds = ground_speed * dt
            if s + ds > mv.horiz_m:
                ds = mv.horiz_m - s
            s += ds
            t += dt
            batt_used += power * dt / 3600.0 / batt_wh * 100.0
            frac = min(1.0, s / mv.horiz_m)
            lat, lon, alt = _interp(mv.a, mv.b, frac)
            emit(lat, lon, alt, v, ground_speed, mv)

    total_duration = samples[-1].t if samples else 0.0
    total_batt = samples[-1].battery_used_pct if samples else 0.0
    return KinTrace(
        samples=tuple(samples),
        total_duration_s=total_duration,
        total_battery_pct=total_batt,
    )
