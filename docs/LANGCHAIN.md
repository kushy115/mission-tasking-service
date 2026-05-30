# LangChain & LangGraph in Perception

How LLM orchestration actually works in Perception, chain by
chain — beginner-friendly up top, then the precise current flow. This documents
the **current code**; see the "Evolution" note at the end for how it got here.

One-sentence summary: **Perception turns a natural-language command into a
schema-validated plan using a LangChain tool-calling agent (`create_agent`)
orchestrated inside a hand-built LangGraph `StateGraph`, with a deterministic
safety kernel — not the LLM — as the source of truth.**

---

## TL;DR — where the LLM is (and isn't)

One LangGraph state machine (the *compile graph*) plus a few standalone LLM calls
on HTTP endpoints. Everything safety-critical is deterministic Python with **no
LLM in the path**.

| # | Chain | Entry point | LLM? | Output shaping |
|---|-------|-------------|------|----------------|
| 1 | **Compile graph** (the spine) | `graph/builder.py` → `graph/nodes.py` | only `plan` + `advisor` nodes | — |
| 2 | **Planner** | `nodes.plan_node` | yes — tool-calling agent (or 1 direct call in `fast`) | `ToolStrategy(MissionPlan)` → deterministic geometry pipeline |
| 3 | **Advisor** | `nodes.advisor_node` | yes | `with_structured_output(OptimizationAdvisory)` |
| 4 | **Mission chat** | `routes.chat_about_mission` | yes | free text + `<<<RETRY: …>>>` sentinel |
| 5 | **Area research** | `routes.research_area_endpoint` | yes | manual JSON → `AreaResearchResponse` (advisory) |
| 6 | **Auto-fleet sizing** | `graph/auto_drones.pick_fleet` | **no** — closed-form math | — |
| 7 | **Inflight supervisor** | `supervisor/` (2nd `StateGraph`) | optional | rule-based `decide(...)` |

**Deterministic core (never calls a model):** `validation/kernel.py` (safety),
`validation/physics.py` (energy/time), `sim/kinematics.py` (kinematic re-fly),
`geo/patterns.py` (search/perimeter geometry), `geo/store.py` (PostGIS),
`graph/auto_drones.py` (fleet sizing).

---

## For a LangChain beginner: which framework features are used

If you know the standard "LangChain in N steps" material, here's the exact map:

| LangChain idea | In Perception? | How |
|---|---|---|
| Chat models, `invoke`, system/user messages | ✅ | `init_chat_model`, `invoke([SystemMessage, HumanMessage])` |
| Prompts | ✅ (as strings) | a big `_SYSTEM_PROMPT` constant + hand-built messages (not `PromptTemplate`) |
| Structured output / output parsers | ✅ | agent uses `ToolStrategy(MissionPlan)`; advisor uses `with_structured_output` |
| **Tools & toolkits** | ✅ | the `@tool` suite in `tools/planning_tools.py`, bound to the agent (DD-016) |
| **Agents (`create_agent`)** | ✅ | the `plan` node is a tool-calling agent (DD-016) |
| LCEL pipes (`prompt \| llm \| parser`) | ❌ replaced | by the LangGraph `StateGraph` — we need branching + a loop, which a straight pipe can't express |
| Splitters / retrievers / RAG / vector stores | ❌ not used | grounding is **exact** PostGIS spatial queries, not semantic retrieval |
| Document loaders | ❌ | "documents" are GeoJSON / YAML loaded by our own code |
| **LangGraph `StateGraph`, `add_messages`, checkpointer, `interrupt`** | ✅ core | the whole compile flow |

> Note for a beginner: tutorials often present an LCEL `prompt | llm | parser`
> chain as "the chain". Perception deliberately uses the heavier **LangGraph
> `StateGraph`** instead because it needs conditional branching, a bounded repair
> **loop**, and a human-approval **interrupt** — none of which a linear pipe can
> express. RAG/retrievers are intentionally absent: drone facts ("is this point
> in the geofence?") are spatial and exact, answered by PostGIS, not by
> similarity search.

---

## Shared model wiring + effort tiers

LLM chains get their model from `nodes._llm_for(model_id)` (a cached
`init_chat_model(...)` per model id; `MTS_LLM_PROVIDER` / `MTS_LLM_MODEL` /
`API_KEY`; default Anthropic `claude-haiku-4-5`). `_llm_lazy()` is the default
model used by the advisor + chat.

The **planning-effort tier** (`_effort_config`, DD-017) picks the model + the
plan path per compile:

| Tier | Model (default) | Plan path | Model calls / drone |
|------|------|------|------|
| `fast` | `claude-haiku-4-5` | one direct `llm.invoke` → JSON | **1** |
| `balanced` | `MTS_LLM_MODEL` | tool-calling agent, recursion 30 | ~4–6 |
| `thorough` | `claude-opus-4-8` | tool-calling agent, recursion 60 | more |

The tier changes **only** the model + whether the agent runs; all downstream
calculations are identical.

---

## Chain 1 — The compile graph (`StateGraph`)

**Files:** `graph/state.py` (state), `graph/nodes.py` (nodes), `graph/builder.py`
(assembly). Orchestrates `POST /v1/missions:compile`.

**State** — `CompileState` is a `TypedDict` threaded through every node:
`raw_command`, `area_id`, `operator_clearance`, `drone_state`,
`selected_sensors`, `planning_effort`, `geo_context`, `weather`, `draft_plan`,
`validation_errors`, `constraints_report`, `repair_attempts`, `status`,
`clarification_questions`, `rejection_reasons`, `repair_drafts`, `advisory`,
`alternatives`, `multi_drone_slot`, `assigned_home`, `conversation_history`, and
a `messages` channel (`Annotated[list, add_messages]`).

**Nodes & edges:**
```
START → intake → (plan | clarify | reject)
        plan   → validate
        validate → (advisor | repair | clarify | reject)
        advisor → finalize
        repair  → plan          (loop)
        clarify → finalize → END
        reject  → finalize → END
        finalize → END          (interrupt AFTER finalize = approval gate)
```
- **`intake`** (deterministic): loads `geo_context` (boundary, NFZs, ceiling,
  home bases) from PostGIS; fetches weather once; short-circuits to
  `NEEDS_CLARIFICATION` (hopelessly vague) or `REJECTED` (bad clearance/area).
- **`plan`** (LLM): see Chain 2.
- **`validate`** (deterministic): `kernel.validate_plan` → `(violations,
  ConstraintReport)`; the safety source of truth. Validates alternatives
  independently.
- **`advisor`** (LLM, non-blocking): Chain 3.
- **`repair`/`clarify`/`reject`**: bookkeeping nodes (status/questions/reasons).
- **`finalize`** (deterministic): assembles the final `MissionPlan`, writes a
  **grounded** `reasoning_trace` from real plan facts (and surfaces
  `rejection_reasons` when rejected), attaches the advisory, persists via
  `save_mission`, sets terminal status.

**Conditional routing** (`builder.py`): `_after_intake` → clarify/reject else
plan; `_after_validate` → clarify/reject (planner self-marked), else advisor (no
violations), else repair (violations & `repair_attempts < cap`), else reject
(budget exhausted).

**Repair loop** — cap 3 (`MTS_REPAIR_LOOP_CAP`), a deliberate cost/safety bound;
each pass injects the kernel's violation list back into the prompt.

**Approval gate** — compiled with a checkpointer + `interrupt_after=["finalize"]`
(`MemorySaver` in tests; `PostgresSaver` in prod so the pause survives restarts).
The compile runs through `finalize` and stops; the operator resolves it via
`POST /v1/missions:approve`. Terminal status is always one of
`READY_FOR_APPROVAL | NEEDS_CLARIFICATION | REJECTED`.

**Multi-drone** — `routes.compile_mission` invokes the graph **once per drone**
(`multi_drone.assign_slots` → non-overlapping altitude band + takeoff stagger;
`nearest_home_base` → a base near the band). Results return as `group_plans`; the
mission is "ready" only if **every** drone is `READY_FOR_APPROVAL`.

---

## Chain 2 — The planner (`nodes.plan_node`)

The agent/direct call is chosen by the effort tier. Either way the model only
chooses *intent*; the **geometry is regenerated in code** so it passes the
kernel.

**The tool suite** (`tools/planning_tools.py`, built per-compile by
`build_planning_tools(geo, profile)` — closures over the area + drone so the
model only passes meaningful args): `get_geofence`, `check_path_clear` (Shapely
NFZ/boundary), `estimate_battery` (wraps `physics.estimate_mission`),
`get_sensor_coverage` (swath + spacing), `lookup_search_pattern` (NFZ-aware
geometry from `geo/patterns`).

**Agent path (`balanced`/`thorough`):** `create_agent(model, tools,
system_prompt=_SYSTEM_PROMPT + _TOOL_USE_NOTE,
response_format=ToolStrategy(MissionPlan))`. `ToolStrategy` (a *non-strict* tool
call) is used instead of provider-native structured output because Anthropic's
strict structured output overflows its compiled-grammar limit on the large
`MissionPlan` schema. The node reads `result["structured_response"]`.
**Graceful fallback (DD-017):** if the agent errors or stalls (e.g.
`GraphRecursionError` from looping), the node falls back to one direct call
(`_direct_plan`) so the drone still gets a plan instead of an empty 0-battery
rejection.

**Fast path:** `_direct_plan` — one `llm.invoke([_SYSTEM_PROMPT, …])`, then
`_strip_to_json` → `json.loads` → `MissionPlan.model_validate`. Alternatives mode
returns `{"plans": [...], "primary_idx": N}` (agent: `ToolStrategy(MissionPlanSet)`);
the primary becomes the draft, the rest are stashed for `validate` to filter.

**Prompt** — `_SYSTEM_PROMPT` carries the `MissionPlan` schema, hard safety
constraints, the absolute NFZ-avoidance rule, leg-type rules, **SELECTION RULES**
(selected area/drone are authoritative; clarify only if the command names a
*different* one; constrain sensors to the checkbox set), and status-decision
rules. The human message inlines `COMMAND`, the UI selections, `GEO_CONTEXT`,
weather, the multi-drone slot briefing, and (on repair) the violation list.
`conversation_history` is replayed as `Human`/`AI` messages.

**Deterministic post-processing (the important part)** — runs after the model on
every tier:
1. **Sensor clamp** — force each leg's `sensor_mode` into the operator's selected
   set (default EO).
2. **Search-pattern densification** — replace each `SEARCH_PATTERN` leg's geometry
   with a real lawnmower fitted to the boundary (and the drone's latitude band
   via `band_subregion` in multi-drone), routed around NFZs; pick a coverage
   altitude (high = wider swath) unless the operator asked for detail; collapse
   multiple search legs to a single pass.
3. **Perimeter-patrol fit** — for boundary/perimeter intent, replace the long
   TRANSIT with an inset perimeter routed around buffered NFZs.
4. **Home → work → home routing** — strip the LLM's bookend legs; prepend a
   `TRANSIT` from the nearest base to the first work waypoint and append a
   `RETURN_TO_BASE` to the nearest base to the *last* one (takeoff/landing bases
   chosen independently).
5. **Snap + NFZ-route + re-aggregate** — snap stray waypoints inside the geofence,
   insert NFZ-avoidance detours on non-search legs (`_route_through_flyable`),
   recompute leg/total duration & battery from `physics.estimate_leg`.

The result is written to `state["draft_plan"]` and handed to `validate`.

---

## Chains 3–7 (briefly)

- **Advisor** (`advisor_node`) — runs only after the kernel passes; advisory,
  never blocks. `with_structured_output(OptimizationAdvisory)` → exactly two
  tactical suggestions (each with an `apply_command`) + an optional
  `resource_constrained_fallback`. Failures are caught and dropped.
- **Mission chat** (`routes.chat_about_mission`, `POST /v1/missions/{id}:chat`) —
  stateless Q&A over a compiled plan; client sends full turn history. If the
  operator asks to redo, the model appends a `<<<RETRY: …>>>` sentinel → the UI
  offers a one-click recompile. Doesn't re-run the kernel.
- **Area research** (`routes.research_area_endpoint`, `POST /v1/areas:research`) —
  given a hand-drawn polygon, one LLM call returns **advisory** airspace guidance
  (likely NFZs, ceiling) parsed into `AreaResearchResponse`. The operator edits
  every field; the kernel is still the source of truth at compile.
- **Auto-fleet sizing** (`auto_drones.pick_fleet`) — **no model call.** Closed-form
  coverage math (area ÷ swath ÷ usable endurance) decides how many drones, honored
  by the kernel per band. Reproducible and instant — important for evals.
- **Inflight supervisor** (`supervisor/`, DD-014) — a second `StateGraph`
  (assess → decide → replan|commit) that replans mid-mission on events (e.g. a
  wind spike), re-running the new plan through the same kernel and swapping it on
  the live WebSocket stream.

---

## Evolution (how the planner got here)

`CLAUDE.md` originally specified a `create_agent` tool-calling planner. The code
went through three stages, each recorded in `DESIGN_DECISIONS.md`:

1. **DD-001** — moved to a single `llm.invoke()` with manual JSON parsing,
   because the agent made 5+ calls/compile and the structured-output + tool
   combo was fragile on Gemini's free tier (quota + a worker crash).
2. **DD-016** — restored the `create_agent` tool-calling agent (on Anthropic
   Haiku, where quota isn't the constraint), grounding the plan with the tool
   suite and returning structured output via `ToolStrategy`.
3. **DD-017** — added the fast/balanced/thorough **effort tiers** (so a
   rate-limited user can choose the one-call `fast` path) plus the graceful
   agent→direct **fallback**.

Throughout, the deterministic geometry pipeline + safety kernel did the heavy
lifting; the LLM's job is a sound first draft.
