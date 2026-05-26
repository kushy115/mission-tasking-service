# Design Decisions

This document records the major design choices made for each non-trivial feature
of the Mission Tasking Service. Read this first before changing anything in the
files referenced from each section.

## How to read this doc

Each section is dated and tagged with the **files it owns**. If you change one
of those files, update the section. If you add a new feature, add a new
section — short, opinionated, links to code, no fluff.

---

## 1. Mission History Browser

**Status:** implemented
**Files:** `app/api/routes.py`, `app/geo/store.py`, `app/static/index.html`

### What
A read-only browser for past mission compilations: list endpoint returns
summaries (id, area, status, timestamp, approved), detail endpoint returns the
full stored `MissionPlan` JSON.

### Why
Missions were already persisted to the `missions` table by `finalize_node` but
there was no read API or UI surface. Operators (and demoers) need to be able to
revisit what they planned earlier without scraping the DB by hand.

### Key choices
- **DB query stays simple**: a single `SELECT mission_id, area_id, status,
  approved, operator_note, created_at FROM missions ORDER BY created_at DESC
  LIMIT N`. No pagination cursors — `LIMIT/OFFSET` is sufficient for the demo
  scale, and the index on `created_at` (the default insertion order) keeps it
  cheap.
- **Two endpoints, not one**: `GET /v1/missions` returns lightweight summaries
  (fast for the history panel), `GET /v1/missions/{id}` returns the heavy
  JSONB `plan`. This avoids streaming megabytes when the user is just
  scrolling.
- **`approved` is reported as a tri-state** (`null`/`true`/`false`) so the UI
  can distinguish "never approved", "approved", "operator-rejected" — all three
  are real states.
- **UI: drawer, not modal.** The history is a side drawer (slides in from the
  right) so the map stays visible. Clicking a mission renders it on the
  current map (changes the area selector if the mission was on a different
  area) and re-uses the same `renderResult` / `drawPlanOnMap` paths.
- **No delete endpoint.** Missions are a durable audit trail — refusing to
  delete is the right default for anything safety-adjacent.

---

## 2. Plan Export (KML / DJI Waypoint)

**Status:** implemented
**Files:** `app/api/routes.py`, `app/export/` (new package)

### What
`GET /v1/missions/{id}/export?format=kml|dji|gpx` returns the mission plan in
a standard waypoint format that real ground-control software can consume.

### Why
Lets you take an MTS-compiled plan and load it into actual flight-planning
software (Mission Planner, QGroundControl, DJI Fly) even though MTS itself has
no hardware integration.

### Key choices
- **Pure-Python generators, no third-party deps.** KML and GPX are plain XML;
  DJI Waypoint v1 is plain CSV/text. We hand-roll them in `app/export/`. Pulling
  in `simplekml` or `gpxpy` would add 200KB of dependencies for what is ~30 lines
  of string formatting.
- **One waypoint per geometry vertex.** We flatten `plan.legs[].geometry[]` into
  a single ordered list. Leg type is encoded in the waypoint name
  (`"L1-TRANSIT-WP3"`) so a human reviewing the file in QGC can see structure.
- **Altitude is AGL, not MSL.** All three formats let altitude be relative;
  KML uses `<altitudeMode>relativeToGround</altitudeMode>`, DJI uses its
  default AGL, GPX puts the value in `<ele>` with no datum tag. We document
  this explicitly in the file headers so an operator doesn't assume MSL and
  fly into terrain.
- **`Content-Disposition: attachment`** so the browser saves the file with a
  sensible name (`mission-<id>.kml`) instead of rendering it as text.

---

## 3. Constraint Violation Overlay

**Status:** implemented
**Files:** `app/static/index.html`, `app/api/routes.py`, `app/api/models.py`

### What
When a mission is REJECTED or has validation errors, the UI shows which
specific waypoints / legs failed which check, color-coded on the map.

### Why
A bullet-list rejection ("leg[2].waypoint[5] outside geofence") is hard to
reason about. Showing the exact waypoint glowing red on the map turns an
abstract violation into a spatial intuition.

### Key choices
- **Reuse the validator's existing violation strings.** The kernel already
  emits strings of the form `"leg[I].waypoint[J] (lat,lon) outside geofence"`
  and `"leg[I] intersects no-fly-zone[K]"`. We parse them in the UI rather
  than re-shaping the API, because:
    1. it keeps the kernel's tests as the single source of truth;
    2. the violation list is already in the response (`plan.rejection_reasons`
       carries violations when the repair budget is blown), and a separate
       structured field would duplicate truth.
- **Violation parser is a single regex in the UI.** If we ever change the
  kernel's string format, the test in `test_validation.py` will catch the
  shape change and we'll update the regex.
- **Color scheme**: red = geofence/NFZ breach (hard safety), orange =
  altitude/coverage (envelope), yellow = battery/endurance (resource).
  Same palette as the existing legend.

---

## 4. Weather Integration (Open-Meteo)

**Status:** implemented
**Files:** `app/weather/__init__.py`, `app/weather/provider.py`,
`app/validation/physics.py`, `app/graph/nodes.py`, `app/validation/kernel.py`

### What
Wind speed/direction, gusts, visibility, precipitation from the free
[Open-Meteo](https://open-meteo.com/) API (no key required), plumbed into:
- the **physics model** as a wind-corrected ground speed / battery penalty;
- the **planner prompt** so the LLM can pick sensor and altitude with weather
  in mind;
- the **safety kernel** as a new "weather acceptable" check.

### Why
The original codebase already had a `weather_provider: synthetic|real`
setting but the `real` path was a TODO. Wind is the single biggest swing
factor on small-drone endurance and visibility is what forces EO→IR fallback.
Wiring this in makes the safety checks meaningful in operational conditions,
not just clean-room sims.

### Key choices
- **Provider abstraction with two implementations**: `SyntheticWeather`
  (seeded RNG, reproducible) and `OpenMeteoWeather` (HTTP). Selected via the
  existing `MTS_WEATHER_PROVIDER` env var.
- **In-process LRU cache, 10-minute TTL.** Open-Meteo is rate-limited
  (~10k req/day per IP); we cache by `(area_id, hour_bucket)` so re-compiles
  of the same mission don't re-fetch.
- **Wind penalty is a single scalar on the physics model**, applied to
  cruise/hover power: `effective_power = nominal_power * (1 + α * |wind_mps|)`
  with α ≈ 0.02 per m/s headwind-equivalent. Principled but deliberately
  simple — the wind direction vs. heading would matter in reality but we don't
  have heading-resolved legs.
- **Hard reject conditions** (in the kernel):
    1. sustained wind > drone's max-wind tolerance (default 12 m/s),
    2. visibility < 1500m AND any leg uses EO sensor,
    3. precipitation > "light" (the drone is not waterproof in this demo).
  Anything below these thresholds is just a battery penalty.
- **Failure mode**: if the weather provider raises, we log and fall through
  to `SyntheticWeather` so a network blip never blocks a compile. Weather is
  an enrichment, not a hard dependency.

---

## 5. Leaflet Draw Area Editor

**Status:** implemented
**Files:** `app/static/index.html`, `app/api/routes.py`, `app/geo/store.py`

### What
Users can sketch a new operating area (boundary + NFZs + home point) directly
on the map and save it via `POST /v1/areas`. The new area immediately becomes
selectable for mission compilation.

### Why
Previously, areas were only seedable via `scripts/seed_db.py` — adding a new
operating area required a server restart and shell access. With the editor,
the system becomes self-service.

### Key choices
- **Leaflet.draw from CDN, no build step.** Pulled in via `<script src>`,
  same style as the base Leaflet load. We do not adopt a bundler.
- **Server-side polygon validation via Shapely.** A user-drawn polygon can be
  self-intersecting or have holes; the endpoint runs `polygon.buffer(0)` to
  repair, then `polygon.is_valid` to reject. NFZs that fall outside the
  boundary are rejected (would be unreachable anyway).
- **Home point auto-snapped to boundary.** The Leaflet editor lets the user
  pick any point as home; we snap to the nearest point on the boundary
  exterior. This matches the kernel's `home_point` semantics
  ("on the boundary").
- **Edit-in-place is upsert, not insert.** `POST /v1/areas` with an existing
  `area_id` overwrites — same as `upsert_area`. The editor surfaces an
  "Editing existing area" warning so this isn't accidental.
- **No auth on the area endpoint.** This is a single-tenant demo; in
  production, area creation would be admin-only (`operator_clearance ==
  "ADMIN"`).

---

## 6. LLM Plan Critique Pass

**Status:** implemented
**Files:** `app/graph/nodes.py`, `app/graph/builder.py`, `app/graph/state.py`,
`app/schemas/plan.py`

### What
After the deterministic kernel passes a plan, a second lightweight LLM call
critiques it for *tactical* quality (orientation vs. wind, target coverage
priority, sensor choice). Returns a `confidence_score` (0–1) and free-text
`critique_notes` that surface in the UI.

### Why
The kernel guarantees *safety*. It does not guarantee *quality* — a plan can
be perfectly legal and still operationally dumb (e.g. flying a lawnmower
oriented across-wind, or searching the wrong half of the area). The critique
pass surfaces those soft issues without giving the LLM veto power.

### Key choices
- **Second LLM call, NOT a multi-turn dialog.** The critique is one extra
  `llm.invoke` after `validate` passes, returning a small JSON. We don't loop
  on it. This costs one extra request per successful compile (free-tier
  friendly) and adds maybe 1–2s of latency.
- **Critique is advisory only.** Low confidence does NOT block approval — it
  shows in the UI as a meter so the operator can make a judgment call. The
  safety kernel remains the only thing that can reject.
- **`confidence_score` is bucketed in the UI**: ≥0.8 green, 0.5–0.8 yellow,
  <0.5 red. Bucketing avoids fake precision (the model is not actually
  calibrated to two decimal places).
- **Graph wiring**: a new `critique` node sits between `validate` (when it
  succeeds) and `finalize`. On critique-LLM failure we set
  `confidence_score=None` and continue — never block on the soft check.
- **Skipped when the plan is REJECTED / NEEDS_CLARIFICATION** — there's
  nothing to critique tactically when the plan is structurally invalid.

---

## 7. Alternative Plan Generation

**Status:** implemented
**Files:** `app/graph/nodes.py`, `app/api/models.py`, `app/api/routes.py`,
`app/static/index.html`

### What
On request (`alternatives: true` in the compile request), the planner returns
2–3 alternative plans (conservative vs. aggressive battery use, different
search orientations). The UI shows them as selectable overlays; operator
picks one before approval.

### Why
There is rarely one "correct" mission. Surfacing alternatives gives the
operator agency and makes the LLM's reasoning visible — they can see WHY one
plan is faster but riskier.

### Key choices
- **One LLM call, multi-plan output.** We extend the planner system prompt
  to optionally emit `{"plans": [<MissionPlan>, ...], "primary_idx": 0}`
  instead of a single plan. ONE call, not N. Same JSON-parsing path, same
  validation kernel run on EACH alternative.
- **Each alternative runs through the safety kernel independently.** Unsafe
  alternatives are dropped silently (operator never sees them) — we never
  surface a violating plan as a "choice".
- **`primary_idx`** is the planner's recommendation; it's pre-selected in
  the UI. The operator can toggle through the others without re-compiling.
- **Repair loop applies to ALL alternatives in aggregate.** If every
  alternative fails the kernel, the repair budget decrements once and the
  planner re-tries the whole set. We do NOT independently repair each
  alternative — that would multiply costs.
- **Defaults to OFF.** `alternatives` is a query option; standard compile is
  still one plan, one validate, one critique. Off by default keeps the demo
  free-tier-cheap.

---

## 8. Airspace Deconfliction

**Status:** implemented
**Files:** `app/validation/deconfliction.py` (new), `app/validation/kernel.py`,
`app/graph/nodes.py`, `app/geo/store.py`, `app/schemas/plan.py`

### What
A 7th safety check: when compiling a new mission, query the DB for missions
in the same area that are APPROVED but not yet completed, and reject if the
new mission's geometry conflicts (same airspace, overlapping time window,
within vertical separation).

### Why
In real operations, multiple drones may share an area. Two missions whose
flight paths cross at the same altitude at the same time is a midair-collision
risk that no per-mission kernel can see.

### Key choices
- **"Active" missions = approved AND created within the last `endurance_s` of
  any drone profile (we use the longest, 45 min).** This is the deconfliction
  window. If `now - mission.created_at > endurance`, the mission has
  necessarily completed and can be ignored. (In production this would be
  swapped for an explicit `completed_at` column.)
- **Conflict = 2D buffer + vertical band overlap.** Two missions conflict if:
    1. their leg geometries' 2D union, buffered by `LATERAL_SEP_M` (default
       50m), intersect; AND
    2. their altitude ranges overlap minus `VERTICAL_SEP_M` (default 30m).
  Both 2D and 3D must overlap — vertical separation alone clears the
  conflict.
- **Self-conflict suppressed.** A repair-loop replan of the same mission_id
  cannot conflict with itself.
- **Reports each conflicting mission by id**, not just a boolean — operator
  needs to know who they're conflicting with to coordinate.
- **PostGIS does the spatial heavy lifting** via `ST_DWithin` and
  `ST_Intersects`. The endurance window is a `WHERE created_at > now() -
  interval`. We do NOT pull all missions into Python.

---

## 9. Multi-Drone Coordination

**Status:** implemented
**Files:** `app/api/models.py`, `app/graph/nodes.py`, `app/graph/state.py`,
`app/graph/builder.py`, `app/api/routes.py`, `app/static/index.html`

### What
Compile a single command into a *group* of plans, one per drone, with
automatic altitude-layer separation and time-staggering so the group is
internally deconflicted.

### Why
Real operations use drone swarms (e.g. three drones sweep a field
simultaneously, each covering a third of it). Doing this as N independent
compiles loses the coordination — they'd each plan the same path. This makes
the planner aware of the whole group.

### Key choices
- **One LLM call per drone, but with shared geo context and an explicit
  layer assignment.** The planner is told: "you are drone 2 of 3, your
  assigned altitude layer is 70m ± 15m, your sibling drones are at 40m and
  100m." The model handles the spatial decomposition (which third of the
  field) but altitudes are server-prescribed.
- **Altitude layers**: 40m, 70m, 100m for up to 3 drones (and a fall-back
  table for more). Spacing is `MIN_VERTICAL_SEP_M = 30`. Documented in
  `app/graph/multi_drone.py`.
- **Time-staggering by 30s per drone for takeoff** prevents collisions at
  the home point (drones can't all be at `(home_lat, home_lon, 30m)` at the
  same second). Encoded as the first leg's `est_duration_s` offset.
- **Deconfliction validator (#8) runs across the group**, treating each
  sibling as a "pre-existing approved mission". This re-uses the same
  primitive — no special-case code path.
- **Failures are per-drone.** If drone 2's plan fails the kernel beyond the
  repair budget, drone 1 and 3's plans still go through; the response
  carries a `plans[]` array with a `status` per slot.
- **Defaults to single-drone.** `CompileRequest.drone_ids` is optional; the
  legacy `drone_state.drone_profile_id` path is preserved for back-compat.

---

## 10. Live Simulation WebSocket Stream

**Status:** implemented
**Files:** `app/sim/executor.py`, `app/sim/stream.py` (new), `app/api/routes.py`,
`app/main.py`, `app/static/index.html`

### What
A WebSocket endpoint `/ws/missions/{id}/sim` that streams synthetic telemetry
(position, altitude, battery, sensor mode, leg index, elapsed time) at 5 Hz
while the planner's mission is "flown" by the executor. The UI animates a
drone marker along the route in real time.

### Why
The biggest demo-quality gap: the existing `verify` endpoint runs a one-shot
sim and returns a summary. There's no visual sense of the drone actually
*moving*. A live stream turns the static map into a flight replay.

### Key choices
- **Server-side discrete-time stepper, NOT client-side interpolation.** The
  server walks the plan at `dt = 0.2s` ticks using the same physics model
  the validator uses, so the telemetry the UI sees is authoritative (battery
  ticks down exactly as the kernel believed it would). The browser's job is
  to draw, not to compute.
- **Time compression default = 10×.** Real missions are 5–45 minutes; we
  send telemetry as if the drone were flying 10× faster (so a 5-min mission
  finishes in 30s of wall clock). The query parameter `?speed=N` lets the
  caller override (e.g. `speed=1` for real-time, `speed=60` for ultra-fast).
- **One ticker per WS connection.** No shared simulator state. Two clients
  watching the same mission get two independent streams (cheap, since the
  state is small). If we ever need multicast, refactor then — not now.
- **Frame schema is flat JSON**, optimized for the UI:
  `{t, lat, lon, alt_m, battery_pct, leg_idx, leg_type, sensor_mode, status}`
  where `status` is `"flying"` / `"done"` / `"aborted"`. The final frame is
  always `status != "flying"` so the UI knows the stream is closed.
- **Auto-abort on geofence breach.** If the simulated position exits the
  boundary, the stream emits one `status="aborted"` frame and closes. (In
  reality this would be a safety-loop hardware command, not a websocket
  close.)
- **Backpressure: drop frames, never block.** If the client's send queue is
  full we skip the frame rather than awaiting. Telemetry is throwaway —
  the UI just re-syncs on the next frame.
- **No persistence.** The stream is ephemeral. The one-shot `verify`
  endpoint is still the canonical "did the sim succeed" record.

---

# Cross-cutting decisions (DD-prefixed)

These decisions span multiple features rather than belonging to one section
above. They were captured before the per-feature sections were written and
are preserved here for continuity.

## DD-001 — `plan` node is a direct LLM call, not a tool-calling agent

**Decision.** The planning step is implemented as a single `llm.invoke()` per
attempt, with the geo context inlined into the system prompt. We do **not**
use `langchain.agents.create_agent` + planning tools, even though `CLAUDE.md`
originally specified an agent.

**Why.** The agent setup made 5+ LLM calls per compile (one per tool decision
plus a final structured-output pass). On Gemini's free tier that exhausted
the 20-RPD quota in 2 compiles. The `response_format=Pydantic` + Gemini
tool-calling combo also silently killed the uvicorn worker on response
handling — no traceback, exit code 0, very hard to debug.

**Trade-off.** We lose the agent's ability to dynamically choose which tool
to call. We don't actually need that here — the geo context, drone profile,
and area boundary are all known at plan time, so we just embed them. The
deterministic safety kernel is still the source of truth on safety.

**Where in code.** `app/graph/nodes.py` — see the big comment block above
`plan_node`.

---

## DD-002 — Provider-agnostic LLM via `init_chat_model` + single `API_KEY` var

**Decision.** The planner routes to whichever LLM provider is configured via
`MTS_LLM_PROVIDER` (`anthropic` | `google_genai` | `openai`). All providers
share the same `API_KEY` env var; the code maps it to whichever kwarg that
provider's chat-model class accepts.

**Why.** During development we had to switch providers mid-project (Gemini's
free tier was too tight for iterative work; Anthropic Haiku is much more
reliable). Hard-coding `ANTHROPIC_API_KEY` everywhere would have been
painful. The unified `API_KEY` makes provider switches a one-config change
with no code edits.

**Trade-off.** Slightly less standard than each provider having its own env
var name. If someone sets `ANTHROPIC_API_KEY` thinking it should work, it
won't — they need `API_KEY`. We document this in `.env.example` and the
notebook Cell 1 comments.

**Where in code.** `app/graph/nodes.py` — `_llm_lazy()`. `app/config.py` —
`Settings.api_key` + `llm_provider` + `llm_model`.

---

## DD-003 — Repair drafts persisted inline as JSONB on the `missions` row

**Decision.** Every repair-loop attempt (draft plan + the violations that
triggered the next attempt) is stored as a JSONB array inside the
`missions.repair_drafts` column, not in a separate `repair_attempts` table.

**Why.** Drafts are 1:N owned by the mission, always loaded together with
the mission, and capped at ~3 entries (the repair-loop cap). A separate
table would require a JOIN on every history lookup for zero benefit.

**Trade-off.** JSONB cells can grow large for complex plans, but practical
size stays under ~50 KB even with 3 drafts. If we ever lift the repair cap
or store hundreds of drafts per mission, revisit this and split the table.

**Where in code.** `app/geo/store.py` — `save_mission`, `load_mission_detail`,
`list_missions`. Schema in the `DDL` constant.

---

## DD-004 — Additive-only DDL with `ADD COLUMN IF NOT EXISTS`

**Decision.** Schema evolution is handled by appending `ALTER TABLE … IF NOT
EXISTS` statements to the `DDL` list. No Alembic, no proper migration framework.

**Why.** This is a demo project. Alembic adds heavyweight infrastructure
(versioned migration files, autogenerate, env config) for a one-developer
codebase. The additive-only constraint (no column renames, no type changes,
no drops) means the same DDL works on fresh and pre-existing databases.

**Trade-off.** Doesn't support destructive migrations or rollbacks. If this
ever becomes a real production service with multiple operators changing the
schema, swap to Alembic.

**Where in code.** `app/geo/store.py` — `DDL` constant + `init_schema()`.

---

## DD-005 — Chat-style clarification thread (UI-local, no new tables)

**Decision.** When `plan_node` returns `NEEDS_CLARIFICATION`, the result panel
becomes a scrollable chat thread. The operator types follow-ups in an input
box; the UI keeps a `conversationHistory` array of `{role, content}` turns and
sends them on every subsequent compile via a new `conversation_history` field
on `CompileRequest`. The planner converts each turn into LangChain
`HumanMessage` / `AIMessage` objects and feeds the full sequence to
`llm.invoke([SystemMessage, ...history, HumanMessage(new_user_msg)])`.

**Why.** The original flow forced operators to re-write the whole command
every time the planner needed clarification — even small refinements lost
context. A chat thread is the natural mental model for "I want to refine a
mission" and the planner already has the necessary infrastructure
(`init_chat_model` + LangChain message types) to consume it. Keeping
conversation state in the UI (not the DB) means no new tables and no
mission_id rewrites: the prior turns are *context*, not first-class records.

**Trade-off.** Conversation does not survive a page reload — once the operator
closes the tab, the thread is gone. Acceptable for a single-operator demo;
revisit when multi-user / audit-trail requirements show up. We also currently
re-send the whole thread on every follow-up; the LangChain `add_messages`
reducer on `state.messages` could carry it inside the graph state instead
(via the LangGraph checkpointer) if we want server-side persistence.

**Where in code.** `app/graph/state.py` (`conversation_history` field),
`app/graph/nodes.py` (`plan_node` builds `[SystemMessage, *history_msgs,
HumanMessage(user_msg)]`), `app/api/models.py` (`CompileRequest`),
`app/api/routes.py` (`_invoke_compile_graph` threading),
`app/static/index.html` (`renderClarificationThread`, `compileWithHistory`).

---

## DD-006 — Operator-authored drone profiles via POST /v1/drones

**Decision.** Drones live in the same `drones` table as the YAML-seeded ones;
operators add new ones via a modal form (`POST /v1/drones`). Seeded profiles
are marked `protected = TRUE` so they always appear and aren't accidentally
overwritten by an operator using the same `profile_id`. (Currently upsert
semantics — same id replaces — but the protected flag means the seeded ones
get re-protected on every `init_schema` run.)

**Why.** Discoverability — the operator UI previously offered no way to add a
drone without rebuilding the container. The form takes the same fields the
YAML schema requires (endurance, cruise/climb, battery, hover/cruise power)
so the safety kernel and physics model work unchanged.

**Trade-off.** Sensor configuration in the modal is a minimal default
(one EO sensor with reasonable specs). Multi-sensor payloads still require
editing JSON directly via the API — an acceptable simplification given the
demo audience.

**Where in code.** `app/api/models.py` (`DroneUpsertRequest`),
`app/api/routes.py` (`upsert_drone_endpoint`), `app/geo/store.py`
(`upsert_drone`, `list_drones` exposes `protected`),
`app/static/index.html` (drone modal + handler).

---

## DD-007 — LLM-driven area research (advisory, not authoritative)

**Decision.** When the operator draws a new boundary, a "✨ Research area"
button calls `POST /v1/areas:research`, which sends the polygon to the
planner LLM with a structured-output prompt asking for: `flight_permitted`,
`ceiling_m`, `suggested_nfzs` (GeoJSON polygons), `notes`. The UI pre-fills
the ceiling field and draws the suggested NFZs as editable polygons. The
operator can delete or modify any of them before saving.

**Why.** Operators don't have FAA charts or terrain DEMs at their fingertips,
and we explicitly forbid proprietary airspace data (per CLAUDE.md). The LLM
has plausible knowledge about famous landmarks, urban areas, airports, etc.
from training data — that's good enough for an advisory hint that beats
"start from a blank polygon". The deterministic safety kernel still
re-validates everything at compile time, so this never substitutes for real
safety checks; it just speeds up area setup.

**Trade-off.** The LLM can hallucinate NFZs that aren't real, or miss real
ones it doesn't recognize. We make it clear in the UI that the output is
ADVISORY and the operator must review. For a real-airspace integration
(OpenAIP, OpenStreetMap, USGS DEM) we'd add it as a separate provider in
parallel; the existing endpoint contract stays the same.

**Where in code.** `app/api/models.py` (`AreaResearchRequest/Response`),
`app/api/routes.py` (`research_area_endpoint` — reuses `_llm_lazy()` from
`graph/nodes.py` and LangChain `SystemMessage`/`HumanMessage` primitives),
`app/static/index.html` (research button handler + draws suggested NFZs as
dashed editable polygons).

---

## DD-008 — LFU cap (10) on operator-authored areas

**Decision.** The `areas` table gains three columns: `access_count`,
`last_accessed_at`, `protected`. Each successful compile bumps the chosen
area's counters. On `POST /v1/areas`, if there are 10 or more *non-protected*
areas, evict the LFU (lowest `access_count`, oldest `last_accessed_at` as
tiebreak). Seeded areas (`yard-simple`, `farmland-complex`) are
`protected = TRUE` and never count toward the cap or get evicted.

**Why.** Operator-authored areas accumulate (every "draw + save") and clutter
the dropdown. A hard cap with LFU eviction keeps the workspace tidy without
asking the operator to manage cleanup. Protecting the seeded areas means
the demo defaults never disappear no matter what the operator does.

**Trade-off.** Eviction is silent — the operator sees the new area appear and
doesn't get a "we removed X" toast. For a real product we'd surface evicted
ids on the response and confirm interactively. The cap of 10 is arbitrary;
plumb through `app/geo/store.py:AREA_LFU_CAP` if it needs tuning.

**Where in code.** `app/geo/store.py` (`AREA_LFU_CAP`,
`evict_lfu_area_if_needed`, `touch_area_access`, additive DDL),
`app/api/routes.py` (eviction call in `upsert_area_endpoint`, access bump
in `compile_mission`), `app/static/index.html` (`area-foot` shows
`N/10 user · M protected`).
