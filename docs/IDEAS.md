# Future Ideas

Parking lot for non-trivial features that are not yet implemented. Each entry
should be specific enough that a future you (or another engineer) can pick it
up without re-deriving the motivation.

---

## Idea 1 — Inflight Supervisor (closed-loop replanning during execution)

**Status:** not started. Considered the highest-leverage next feature because
it bridges "tasking" and "autonomy" — the current README says "the service
plans; it never flies", and this is the first piece that lets it *react*.

### What

Turn the live-sim WebSocket from "replay the static plan" into a real-time
safety supervisor that can replan mid-flight.

- A new `supervisor` subgraph (separate LangGraph compile) runs alongside
  `app/sim/stream.py`. Inputs each tick: current telemetry + an *event
  channel*.
- The event channel fires programmable or operator-triggered events: TFR /
  new NFZ drops on the map, wind step-change exceeds threshold, simulated
  sensor fault, battery drains faster than predicted, GPS dropout.
- Supervisor decision is one of:
  `CONTINUE | REPLAN_FROM_HERE | DIVERT_TO_SAFE_POINT | RTB_NOW | EMERGENCY_LAND`.
  The LLM proposes; the deterministic kernel disposes (same validator path).
- `REPLAN_FROM_HERE` invokes a stripped-down compile graph that starts at the
  drone's current `(lat, lon, alt, battery_pct)` and re-plans to mission
  completion or RTB. The new plan goes through the same validator and the
  same critique pass.
- UI: original plan greyed out, replan in cyan, event banner at top,
  supervisor's reasoning trace in the side panel.

### Why this is the standout move for this project

- Bridges tasking and autonomy. The single biggest gap right now is that the
  plan is computed once and frozen at approval; this makes the system
  closed-loop.
- Re-uses everything already built (kernel, physics, checkpointer, executor,
  deconfliction). Not net-new infrastructure — the capstone on it.
- Exercises LangGraph at depth: a long-running graph with an event channel,
  interrupts that resume on telemetry, hot checkpointer so any in-flight
  decision is auditable post-hoc.
- Demo story: operator approves a patrol → 30s in, toggle a "wind spike"
  event → drone replans live on the map → operator sees the new path, the
  new battery budget, and the supervisor's reasoning ("wind exceeds 12 m/s
  sustained, shortened pattern and added direct RTB").

### Rough effort

3–5 focused days. Most of the work is the event channel + the supervisor
decision policy; the replan re-uses the compile graph.

### Sketch of the work

1. `app/supervisor/state.py` — `SupervisorState` (current telemetry, event
   queue, original plan, decision history).
2. `app/supervisor/events.py` — event types + an injector endpoint
   (`POST /v1/missions/{id}/sim:inject`) so the UI and tests can fire events.
3. `app/supervisor/nodes.py` — `assess`, `decide`, `replan`, `commit`.
   `decide` is the LLM node with structured output over the decision enum.
   `replan` calls into the existing compile graph from a "warm start" state.
4. `app/supervisor/builder.py` — assemble the subgraph; reuse the Postgres
   checkpointer so supervisor decisions persist.
5. Wire into `app/sim/stream.py`: per-tick, push telemetry into the supervisor
   subgraph; on a non-`CONTINUE` decision, the sim re-points to the new plan
   and the WebSocket frame includes a `supervisor_event` field.
6. UI: event banner, "inject event" debug menu, original-plan greyout +
   replan overlay.
7. Tests: per-decision unit tests against synthetic telemetry + event pairs.
8. Eval extension: add an `evals/dataset_supervisor.jsonl` with telemetry +
   event scenarios and expected decisions.

### Idea 2 (kept for reference) — Monte Carlo Robustness Certificate

Smaller follow-up: before `READY_FOR_APPROVAL`, run N=100 stochastic rollouts
through the executor with seeded perturbations (±20% wind, gust spikes, ±5%
sensor swath, ±3% battery curve, GPS noise). Aggregate into a per-constraint
success probability and a `P(mission_success)`. Plans below a threshold
become `CONDITIONAL_APPROVAL`. Becomes more meaningful once the supervisor
exists, because rollouts can simulate the supervisor's reactions.
