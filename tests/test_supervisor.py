"""Unit tests for the inflight supervisor policy.

The policy module (app/supervisor/policy.py) is pure-Python + shapely. No DB,
no LLM, no async — so tests are fast and deterministic. The orchestrator
(which IS async and DOES touch the DB) is exercised via the end-to-end smoke
test in test_api.py + manual UI verification.
"""

from __future__ import annotations

import pytest

from app.supervisor.events import Event, EventType
from app.supervisor.policy import decide
from app.supervisor.state import SupervisorDecision

# A long-endurance profile (matches data/drones/long_endurance_quad.yaml).
PROFILE = {"cruise_speed_mps": 18.0, "cruise_power_w": 380.0, "battery_wh": 540.0}

# Yard-simple home point (rough): just somewhere we can use as origin.
HOME_LON, HOME_LAT = -95.50, 41.25


def _telemetry(*, lat=41.250, lon=-95.499, alt=60.0, battery=85.0, leg_idx=0):
    return {
        "t": 30.0,
        "lat": lat,
        "lon": lon,
        "alt_m": alt,
        "battery_pct": battery,
        "leg_idx": leg_idx,
        "leg_type": "TRANSIT",
        "sensor_mode": "EO",
        "status": "flying",
    }


def test_no_events_returns_continue():
    decision, _ = decide(
        current=_telemetry(),
        remaining_legs=[],
        home_lon=HOME_LON,
        home_lat=HOME_LAT,
        pending_events=[],
        drone_profile=PROFILE,
    )
    assert decision == SupervisorDecision.CONTINUE


def test_lost_link_triggers_loiter_rtb():
    events = [
        Event(type=EventType.LOST_LINK, note="link dropped"),
    ]
    decision, reason = decide(
        current=_telemetry(),
        remaining_legs=[],
        home_lon=HOME_LON,
        home_lat=HOME_LAT,
        pending_events=events,
        drone_profile=PROFILE,
    )
    assert decision == SupervisorDecision.LOITER_RTB
    assert "link" in reason.lower() or "hold" in reason.lower() or "return" in reason.lower()


def test_manual_land_wins_over_lost_link():
    events = [
        Event(type=EventType.LOST_LINK),
        Event(type=EventType.MANUAL_LAND),
    ]
    decision, _ = decide(
        current=_telemetry(),
        remaining_legs=[],
        home_lon=HOME_LON,
        home_lat=HOME_LAT,
        pending_events=events,
        drone_profile=PROFILE,
    )
    assert decision == SupervisorDecision.EMERGENCY_LAND


def test_wind_spike_over_tolerance_triggers_rtb():
    events = [Event(type=EventType.WIND_SPIKE, payload={"wind_mps": 20.0})]
    decision, _ = decide(
        current=_telemetry(),
        remaining_legs=[],
        home_lon=HOME_LON,
        home_lat=HOME_LAT,
        pending_events=events,
        drone_profile=PROFILE,
    )
    assert decision == SupervisorDecision.RTB_NOW


def test_wind_spike_below_tolerance_is_continue():
    events = [Event(type=EventType.WIND_SPIKE, payload={"wind_mps": 6.0})]
    decision, _ = decide(
        current=_telemetry(),
        remaining_legs=[],
        home_lon=HOME_LON,
        home_lat=HOME_LAT,
        pending_events=events,
        drone_profile=PROFILE,
    )
    assert decision == SupervisorDecision.CONTINUE


def test_nfz_popup_intersecting_remaining_triggers_replan():
    # Remaining leg passes through the popup polygon.
    remaining = [
        {
            "leg_type": "TRANSIT",
            "geometry": [
                {"lat": 41.250, "lon": -95.495, "alt_m": 60.0},
                {"lat": 41.250, "lon": -95.490, "alt_m": 60.0},
            ],
            "est_duration_s": 60.0,
            "est_battery_pct": 2.0,
        }
    ]
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [-95.494, 41.249],
                [-95.491, 41.249],
                [-95.491, 41.251],
                [-95.494, 41.251],
                [-95.494, 41.249],
            ]
        ],
    }
    events = [Event(type=EventType.NFZ_POPUP, payload={"polygon": polygon})]
    decision, _ = decide(
        current=_telemetry(lat=41.250, lon=-95.499),
        remaining_legs=remaining,
        home_lon=HOME_LON,
        home_lat=HOME_LAT,
        pending_events=events,
        drone_profile=PROFILE,
    )
    assert decision == SupervisorDecision.REPLAN_FROM_HERE


def test_nfz_popup_over_current_position_triggers_rtb():
    # Polygon covers current position.
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [-95.5005, 41.2495],
                [-95.4985, 41.2495],
                [-95.4985, 41.2505],
                [-95.5005, 41.2505],
                [-95.5005, 41.2495],
            ]
        ],
    }
    events = [Event(type=EventType.NFZ_POPUP, payload={"polygon": polygon})]
    decision, _ = decide(
        current=_telemetry(lat=41.250, lon=-95.499),
        remaining_legs=[],
        home_lon=HOME_LON,
        home_lat=HOME_LAT,
        pending_events=events,
        drone_profile=PROFILE,
    )
    assert decision == SupervisorDecision.RTB_NOW


def test_nfz_popup_not_intersecting_remains_continue():
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [-95.40, 41.30],
                [-95.39, 41.30],
                [-95.39, 41.31],
                [-95.40, 41.31],
                [-95.40, 41.30],
            ]
        ],
    }
    events = [Event(type=EventType.NFZ_POPUP, payload={"polygon": polygon})]
    decision, _ = decide(
        current=_telemetry(),
        remaining_legs=[],
        home_lon=HOME_LON,
        home_lat=HOME_LAT,
        pending_events=events,
        drone_profile=PROFILE,
    )
    assert decision == SupervisorDecision.CONTINUE


def test_low_battery_background_check_triggers_rtb_without_events():
    # No events at all, but battery is low enough that even the RTB-cost
    # projection puts us below the supervisor's reserve margin.
    decision, reason = decide(
        current=_telemetry(battery=14.0, lat=41.260, lon=-95.490),
        remaining_legs=[],
        home_lon=HOME_LON,
        home_lat=HOME_LAT,
        pending_events=[],
        drone_profile=PROFILE,
    )
    assert decision == SupervisorDecision.RTB_NOW
    assert "projected" in reason or "margin" in reason


def test_event_from_dict_rejects_unknown_type():
    with pytest.raises(ValueError):
        Event.from_dict({"type": "TOTALLY_MADE_UP"})
