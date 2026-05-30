# Perception Guide — orientation, safety model & observability

A short, opinionated tour of the **Perception** for someone with
10–30 minutes who wants to read the engineering. For *running* the service see
the top-level [`README.md`](../README.md); for the LangChain/LangGraph internals
see [`LANGCHAIN.md`](./LANGCHAIN.md); for the full decision log see
[`DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md).

> If you read one file, read [`app/validation/kernel.py`](../app/validation/kernel.py).
> That's the safety core — deliberately the most boring file in the repo.

---

## What it is

An operator describes a drone mission in plain English; Perception returns a
schema-validated `MissionPlan` (ordered legs, search patterns, sensor tasks,
geofence-compliant geometry, a battery budget, a reasoning trace) — or a
`NEEDS_CLARIFICATION` / `REJECTED` response with specific reasons. **It plans and
validates; it never flies.** The validated plan is the deliverable; a downstream
autonomy stack would consume it (export formats live in `app/export/`).

Core idea: **the LLM proposes, deterministic code disposes.** The model drafts a
plan (grounding itself with tools); a pure-Python safety kernel — no LLM, no
network — is the single source of truth on whether that plan is safe.

---

## How it's built, in LangChain terms

- **A custom LangGraph `StateGraph`, not just a prebuilt agent.** The compile
  flow is an explicit graph because we need conditional branching into a
  deterministic validator and a bounded repair cycle — shapes a tool-calling
  agent loop doesn't give you. (The `plan` node *contains* a tool-calling agent;
  the overall flow is the graph.)
- **The planner is a real `create_agent` tool-calling agent** (DD-016) on a
  provider-agnostic chat model (`init_chat_model`, so the same code targets
  Anthropic / OpenAI / Google by env). It grounds its plan with tools
  (`check_path_clear`, `estimate_battery`, `get_sensor_coverage`,
  `lookup_search_pattern`, `get_geofence`) and returns a schema-valid plan via
  `ToolStrategy(MissionPlan)`. *(History: the planner was a single `llm.invoke()`
  for a while — DD-001, dropped on Gemini quota — then restored as the agent in
  DD-016 once on Anthropic.)*
- **Planning-effort tiers** (DD-017): a per-compile knob — `fast` (one direct
  LLM call, rate-limit friendly), `balanced` (the agent), `thorough` (agent on
  the strongest model). The tier only changes the model + whether the agent
  loop runs; **every safety calculation is identical across tiers.**
- **The advisor node uses `with_structured_output(OptimizationAdvisory)`** — the
  idiomatic LangChain pattern for forcing the LLM into a Pydantic schema.
- **Branching uses `add_conditional_edges`** (think `RunnableBranch` at the graph
  layer): two routers (`_after_intake`, `_after_validate`) plus the
  `repair → plan` cycle that lets the kernel correct the LLM.
- **The `messages` channel uses LangGraph's `add_messages` reducer** (append, not
  replace) — what makes the multi-turn clarification thread work across attempts.
- **Human-in-the-loop approval is a real `interrupt`.** The graph is compiled
  with `interrupt_after=["finalize"]`; `POST /v1/missions:approve` resumes it.
  The checkpointer is `langgraph.checkpoint.postgres.PostgresSaver`, so the pause
  survives a restart and pods scale horizontally (state is in Postgres, not
  in-process).
- **A second graph: the inflight supervisor** (`app/supervisor/`, DD-014) — a
  separate `StateGraph` (assess → decide → replan|commit) that replans *during*
  the live sim when conditions change, re-running through the same kernel.
- **An independent kinematic verifier** (`app/sim/`, DD-015): re-flies each plan
  tick-by-tick (accel, cornering, climb, a wind triangle) to confirm it's
  actually flyable and to drive the live map — independent of the planner's
  estimate, so the pre-flight check can genuinely disagree.

---

## Reading paths

**10-minute path — "is the agent engineering sound?"** Read these four:

1. [`app/schemas/plan.py`](../app/schemas/plan.py) — the `MissionPlan` Pydantic v2
   contract. What the LLM must produce and what the validator checks; the rest
   falls out of it.
2. [`app/graph/builder.py`](../app/graph/builder.py) — the `StateGraph` wiring.
   Eight nodes (happy path `intake → plan → validate → advisor → finalize`, plus
   `repair`, `clarify`, `reject`), branching via `add_conditional_edges`, one
   `interrupt` after `finalize`, repair loop bounded at 3.
3. [`app/graph/nodes.py`](../app/graph/nodes.py) — the node bodies. `plan_node`
   builds the tool-calling agent (or the fast direct call), then runs the
   deterministic geometry pipeline. `_effort_config` / `_direct_plan` are the
   tier + fallback logic.
4. [`app/validation/kernel.py`](../app/validation/kernel.py) — the deterministic
   safety kernel; eight checks, no LLM, fully unit-tested in
   [`tests/test_validation.py`](../tests/test_validation.py). The repair loop
   feeds its error list back into the next plan attempt verbatim.

**30-minute path — "is the productionization real?"** Add:

5. [`docs/DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md) — the decision log. Start
   with DD-016 (the agent), DD-017 (effort tiers + the fallback), DD-014
   (supervisor), DD-015 (kinematic verifier).
6. [`deploy/helm/mts/`](../deploy/helm/mts/) — the Helm chart; each template has a
   one-line "why". The HPA scales on **inflight requests, not CPU**, because LLM
   calls are I/O-bound and CPU underreports load.
7. [`app/observability/`](../app/observability/) + the auto-provisioned Grafana
   dashboard (`deploy/docker/grafana/`). Three layers: LangSmith (LLM traces),
   OpenTelemetry (service spans HTTP → graph → SQL), Prometheus (ops metrics).
   The metric reference is below.
8. [`app/supervisor/`](../app/supervisor/) — the second LangGraph subgraph; the
   closed-loop replanning story.

---

## The safety model

The kernel (`app/validation/kernel.py`) runs after **every** plan attempt and
returns a structured violation list + a `ConstraintReport`. A plan is only
`READY_FOR_APPROVAL` if every check passes. No LLM, no network.

1. **Geofence containment** — every waypoint inside the boundary polygon
   (Shapely `covers`, boundary loaded from PostGIS).
2. **No-fly-zone avoidance** — no leg *segment* intersects any NFZ polygon (the
   full polyline is tested, not just waypoints — corners between waypoints count).
3. **Altitude window** — every waypoint below the area ceiling and above the
   min-AGL floor (default 20 m, `MTS_MIN_AGL_M`).
4. **Battery reserve** — estimated consumption leaves `battery_reserve_pct ≥ 20%`
   at landing (configurable, `MTS_BATTERY_RESERVE_MIN_PCT`). Physics model uses
   cruise/climb/hover/sensor draw + the weather wind penalty.
5. **Endurance budget** — `total_duration_s` within the drone's rated endurance.
6. **Sensor coverage** — search-pattern track spacing within the sensor's swath
   at the planned altitude (no coverage gaps).
7. **Ends with RTB** — the plan ends with a `RETURN_TO_BASE` leg.
8. **Airspace deconfliction** (DD-008) — runs only when other recently-approved
   missions exist in the same area; rejects plans whose timing + geometry
   conflict with already-approved flights.

The kernel is deliberately conservative — when in doubt it rejects. The repair
loop (cap 3) gives the planner three attempts to fix violations before the
service returns `REJECTED` with the list.

**Adding a constraint:** (1) add the check + a `ConstraintReport` field in
`kernel.py`; (2) add a test in `tests/test_validation.py` (mandatory — the
guarantees only hold with a regression test per constraint); (3) if the planner
needs to know about it, surface it through the geo store and inline it into the
prompt.

---

## Observability — reading the metrics

Open Grafana at <http://localhost:3000> (auto-provisioned from
`deploy/docker/grafana/dashboards/mts.json`). Read the layers in this order when
debugging.

### Compile layer (user-facing)
- **`mts_compile_requests_total{status}`** — `status ∈ {READY_FOR_APPROVAL,
  REJECTED, NEEDS_CLARIFICATION, error}`. Healthy: >70% READY on a clean eval,
  `error` = 0. REJECTED rising → check the violations panel; `error` rising →
  service failing (Postgres down, LLM misconfigured) — `docker compose logs mts`.
- **`mts_compile_duration_seconds`** — request→response wall time. Healthy p95 <
  30s on `fast`/Haiku. Spikes → slow LLM, repair loops, **or rate-limit back-off
  on `balanced`/`thorough`** (see the LLM layer).
- **`mts_repair_loops`** — repair iterations per compile, capped at 3
  (`MTS_REPAIR_LOOP_CAP`). Healthy avg < 1.0. Approaching 3 → planner can't fix
  its plans; a wave of REJECTED is coming.
- **`mts_rejections_total` / `mts_clarifications_total`** — convenience counters
  for alerting.

### LLM layer (cost + provider health)
- **`mts_llm_calls_total{provider, model, outcome}`** — `outcome ∈ {ok, error,
  rate_limited}`. **Calls-per-compile depends on the effort tier:** `fast` ≈ 1
  (+1 per repair); `balanced`/`thorough` ≈ 4–6+ per drone (the agent's tool
  round-trips), multiplied by fleet size. `rate_limited > 0` → you've hit the
  provider's per-minute limit; on a low tier, multi-drone `balanced` bursts will
  throttle and the compile crawls in 429 back-off (a rate limit, not a bug — use
  `fast` or raise the tier).
- **`mts_llm_duration_seconds{provider, model}`** — single-call latency. Healthy
  p50 < 2s / p95 < 5s for Haiku; ~5s p50 for Opus. p50 climbing → prompt grew.

### Safety-kernel layer (what the planner gets wrong)
- **`mts_validation_violations_total{check}`** — one per kernel rejection, by
  check (`inside_geofence`, `avoids_nfz`, `battery_within_budget`,
  `within_endurance`, `sensor_coverage_adequate`, `ends_with_rtb`,
  `weather_acceptable`, `airspace_deconflicted`). A dominant check points at a
  prompt/logic gap — e.g. `avoids_nfz` → make geo context more prominent;
  `ends_with_rtb` → strengthen the RTB rule.

### Plan-shape layer (approved plans only)
- **`mts_plan_legs`** (healthy p50 2–3, p95 ≤ 5), **`mts_plan_battery_pct`**
  (p50 < 30% on the long-endurance drone; p95 > 80% predicts battery
  violations), **`mts_plan_duration_seconds`** (matches workload: patrol 2–5 min,
  full sweep 10–20 min).

### Other
`mts_http_*` are FastAPI middleware metrics (sanity check that requests reach the
service). Ignore `python_*` / `process_*` runtime defaults unless chasing a leak.

**Adding a metric:** define it in `app/observability/metrics.py`, increment from
the code path, add a Grafana panel with a `description`, document it here. Grafana
picks up dashboard JSON within ~10s — no restart.

---

## Likely objections (and the answers)

- **"Why a custom `StateGraph` and not just an agent?"** We need conditional
  branching into a deterministic validator and a bounded self-correction cycle —
  a graph shape, not a tool-calling shape. The agent loop lives *inside* the
  `plan` node; the *validator* decides when planning stops, not the model.
- **"Why was the planner ever a plain `llm.invoke()`?"** It was, between DD-001
  and DD-016 — a cost/quota workaround on Gemini's free tier. On Anthropic the
  tool-calling agent (DD-016) is back, with effort tiers (DD-017) so a
  rate-limited user can still pick the one-call `fast` path.
- **"The flight-physics model isn't aerospace-grade."** Correct, and the module
  docstring says so. It's principled, deterministic, documented; the *validator*,
  not the physics, is the safety surface. The kinematic verifier (DD-015) is a
  higher-fidelity independent cross-check.
- **"You don't fly the drone."** By design — Perception is the tasking + validation
  layer, and the `MissionPlan` is the deliverable.
