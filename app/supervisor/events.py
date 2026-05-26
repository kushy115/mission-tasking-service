"""External event types the supervisor consumes during live execution.

Events arrive via a per-WebSocket asyncio.Queue (one queue per active live
session, keyed by mission_id+session_id). The HTTP injector endpoint pushes
events; the orchestrator drains them on each tick.

Schema is deliberately small and JSON-friendly so the UI can submit events
without server-side typing gymnastics.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field
from typing import Any


class EventType(str, enum.Enum):
    """The events a live mission can experience.

    Keep this list small — every type needs a deterministic policy branch in
    `app/supervisor/policy.py`. Add new types behind a feature flag if they
    are experimental.
    """

    WIND_SPIKE = "WIND_SPIKE"           # sustained wind step-up in m/s
    NFZ_POPUP = "NFZ_POPUP"             # new no-fly polygon appears mid-flight
    BATTERY_FAULT = "BATTERY_FAULT"     # drain rate increases unexpectedly
    SENSOR_FAULT = "SENSOR_FAULT"       # named sensor goes offline (EO/IR)
    GPS_DROPOUT = "GPS_DROPOUT"         # advisory; nav degraded
    MANUAL_RTB = "MANUAL_RTB"           # operator-triggered return-to-base
    MANUAL_LAND = "MANUAL_LAND"         # operator-triggered emergency land


@dataclass
class Event:
    """One in-flight event injected into the active mission.

    Payload is intentionally loose — each EventType reads what it needs.
    See `policy.decide_event` for the contract per type.
    """

    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    note: str = ""  # operator-supplied free text shown in the UI banner

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        type_str = d.get("type") or ""
        try:
            etype = EventType(type_str)
        except ValueError as e:
            raise ValueError(f"unknown event type: {type_str!r}") from e
        return cls(
            type=etype,
            payload=dict(d.get("payload") or {}),
            note=str(d.get("note") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "payload": self.payload, "note": self.note}


# ---- per-process registry of active event queues --------------------------
#
# We key by `(mission_id, session_id)` so two browser tabs watching the same
# mission don't share an injector. The session_id is generated on the client
# side and passed in the WS query string; the injector POST carries it too.
# The registry lives in the FastAPI process — see DD-014 trade-off note.

_QUEUES: dict[tuple[str, str], asyncio.Queue[Event]] = {}


def register_queue(mission_id: str, session_id: str) -> asyncio.Queue[Event]:
    q: asyncio.Queue[Event] = asyncio.Queue(maxsize=64)
    _QUEUES[(mission_id, session_id)] = q
    return q


def unregister_queue(mission_id: str, session_id: str) -> None:
    _QUEUES.pop((mission_id, session_id), None)


def get_queue(mission_id: str, session_id: str) -> asyncio.Queue[Event] | None:
    return _QUEUES.get((mission_id, session_id))


def active_sessions(mission_id: str) -> list[str]:
    return [sid for (mid, sid) in _QUEUES if mid == mission_id]
