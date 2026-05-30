"""Plan-execution verifier (DD-015).

Given an approved MissionPlan, re-fly it against an INDEPENDENT, higher-fidelity
kinematic model (`sim/kinematics.py`) — not the planner's own coarse estimator.
Because the verifier no longer checks the planner against itself, it can now
legitimately disagree, which is the entire point of a verifier. It confirms:

  - the kinematic re-fly completes within the drone's rated endurance;
  - the re-flown duration / battery stay within tolerance of the plan's estimate
    (drift here flags geometry the coarse estimator under-counted — sharp turns,
    accel/braking, headwind legs);
  - the path stays inside the geofence and clear of every NFZ at EVERY tick
    (not just at the waypoints — a leg can bulge out between two in-bounds ends);
  - the drone never crosses its point-of-no-return: at every tick it still has
    enough battery to fly home to the nearest base with reserve to spare.

Exposed via POST /v1/missions/{id}:verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Point

from app.config import get_settings
from app.geo.store import get_engine, load_geo_context
from app.schemas.plan import MissionPlan, Waypoint
from app.sim.kinematics import simulate_trajectory
from app.validation.kernel import GeoContext
from app.validation.physics import DroneProfile, haversine_m

# Starting battery assumed at takeoff for the re-fly. A stored plan is verified
# from a full pack; the operator's actual starting charge is a UI compile input.
_START_BATTERY_PCT = 100.0
# Absolute floors below which drift is noise, not signal — keeps short clean
# plans from tripping the relative threshold on a couple of seconds / tenths %.
_DUR_FLOOR_S = 45.0
_BATT_FLOOR_PCT = 5.0


@dataclass
class SimResult:
    flown_ok: bool
    actual_duration_s: float
    actual_battery_pct: float
    deviations: list[str] = field(default_factory=list)


def _home_points(geo: GeoContext) -> list[tuple[float, float]]:
    """Registered home bases as (lon, lat); falls back to the single home_point."""
    bases = list(geo.home_bases) if geo.home_bases else []
    if not bases:
        bases = [geo.home_point]
    return bases


def _return_pct(
    lat: float, lon: float, bases: list[tuple[float, float]], profile: DroneProfile
) -> float:
    """Approx battery % to fly straight home to the nearest base at cruise."""
    here = Waypoint(lat=lat, lon=lon, alt_m=0.0)
    nearest_m = min(
        haversine_m(here, Waypoint(lat=blat, lon=blon, alt_m=0.0)) for blon, blat in bases
    )
    cruise = profile.cruise_speed_mps if profile.cruise_speed_mps > 0 else 1.0
    batt_wh = profile.battery_wh if profile.battery_wh > 0 else 1.0
    return_time_s = nearest_m / cruise
    return profile.cruise_power_w * return_time_s / 3600.0 / batt_wh * 100.0


def simulate_execution(
    plan: MissionPlan,
    profile: DroneProfile,
    tolerance_pct: float = 15.0,
) -> SimResult:
    settings = get_settings()
    deviations: list[str] = []

    # Load the operating area; pull current wind so the kinematic re-fly resolves
    # the same wind triangle the kernel assumed at compile time. Best-effort:
    # any failure falls back to calm air rather than blocking verification.
    geo: GeoContext | None = None
    wind_mps = 0.0
    wind_from_deg = 0.0
    try:
        geo = load_geo_context(get_engine(), plan.area_id)
        try:
            from app.weather import get_weather_for_area

            home_lon, home_lat = geo.home_point
            wx = get_weather_for_area(home_lat, home_lon)
            wind_mps = wx.wind_mps
            wind_from_deg = wx.wind_dir_deg
        except Exception:  # noqa: BLE001 — weather optional; calm-air fallback
            wind_mps = 0.0
    except Exception as e:  # noqa: BLE001
        deviations.append(f"could not load operating area for verification: {e}")

    trace = simulate_trajectory(
        plan,
        profile,
        wind_mps=wind_mps,
        wind_from_deg=wind_from_deg,
        wind_coeff=settings.weather_wind_power_coeff,
        dt=0.5,
    )

    # 1) Endurance.
    if trace.total_duration_s > profile.rated_endurance_s:
        deviations.append(
            f"kinematic re-fly takes {trace.total_duration_s:.0f}s, exceeds rated "
            f"endurance {profile.rated_endurance_s:.0f}s"
        )

    # 2) Drift vs the planner's own estimate (now a real cross-check). The two
    # models legitimately differ a little (the kinematic one climbs while moving
    # forward, the estimator bills climb sequentially), so we only flag drift
    # that is BOTH relatively large AND absolutely meaningful — otherwise a tiny
    # clean plan trips the relative threshold on rounding alone.
    est_dur = plan.total_duration_s or trace.total_duration_s
    est_batt = plan.total_battery_pct or trace.total_battery_pct
    dur_diff = abs(trace.total_duration_s - est_dur)
    batt_diff = abs(trace.total_battery_pct - est_batt)
    if est_dur > 0 and dur_diff > tolerance_pct / 100 * est_dur and dur_diff > _DUR_FLOOR_S:
        deviations.append(
            f"duration drift: kinematic {trace.total_duration_s:.0f}s vs estimated "
            f"{est_dur:.0f}s (>{tolerance_pct:.0f}% — likely cornering / accel / wind)"
        )
    if est_batt > 0 and batt_diff > tolerance_pct / 100 * est_batt and batt_diff > _BATT_FLOOR_PCT:
        deviations.append(
            f"battery drift: kinematic {trace.total_battery_pct:.1f}% vs estimated "
            f"{est_batt:.1f}% (>{tolerance_pct:.0f}%)"
        )

    # 3) Per-tick geofence + NFZ containment, and 4) return-to-home reachability.
    if geo is not None:
        boundary = geo.boundary if geo.boundary.is_valid else geo.boundary.buffer(0)
        nfzs = [(n if n.is_valid else n.buffer(0)) for n in geo.nfz_polygons]
        bases = _home_points(geo)
        reserve_min = settings.battery_reserve_min_pct
        breached_fence = breached_nfz = crossed_pnr = False
        for s in trace.samples:
            pt = Point(s.lon, s.lat)
            if not breached_fence and not boundary.covers(pt):
                deviations.append(f"exits geofence at t={s.t:.0f}s (leg {s.leg_idx}, {s.leg_type})")
                breached_fence = True
            if not breached_nfz:
                for ni, nfz in enumerate(nfzs):
                    if nfz.covers(pt):
                        deviations.append(
                            f"enters no-fly-zone[{ni}] at t={s.t:.0f}s (leg {s.leg_idx})"
                        )
                        breached_nfz = True
                        break
            if not crossed_pnr:
                remaining = _START_BATTERY_PCT - s.battery_used_pct
                if remaining - _return_pct(s.lat, s.lon, bases, profile) < reserve_min:
                    deviations.append(
                        f"crosses point-of-no-return at t={s.t:.0f}s: ~{remaining:.0f}% left, "
                        f"not enough to return home with {reserve_min:.0f}% reserve"
                    )
                    crossed_pnr = True
            if breached_fence and breached_nfz and crossed_pnr:
                break

    return SimResult(
        flown_ok=not deviations,
        actual_duration_s=trace.total_duration_s,
        actual_battery_pct=trace.total_battery_pct,
        deviations=deviations,
    )
