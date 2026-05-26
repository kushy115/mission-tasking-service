"""Inflight supervisor — closed-loop safety / replanning during live execution.

See docs/DESIGN_DECISIONS.md DD-014. Augments the live-sim WebSocket stream
with a per-tick decision loop that consumes external events (weather spike,
NFZ popup, sensor/battery fault, manual override) and emits one of:
CONTINUE | REPLAN_FROM_HERE | DIVERT_TO_SAFE_POINT | RTB_NOW | EMERGENCY_LAND.

The supervisor is additive: any exception inside the supervisor path is
logged and absorbed; the underlying telemetry stream keeps running.

This `__init__` is intentionally empty so importing a single submodule (e.g.
`app.supervisor.policy` for unit tests) does NOT cascade-load the orchestrator,
schemas, kernel, and physics modules. Callers import what they need directly.
"""
