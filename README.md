# Mission Tasking Service (MTS)

[![CI](https://github.com/kushy115/mission-tasking-service/actions/workflows/ci.yml/badge.svg)](https://github.com/kushy115/mission-tasking-service/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangChain 1.x](https://img.shields.io/badge/langchain-1.x-1c3c3c.svg)](https://docs.langchain.com/)
[![LangGraph 1.x](https://img.shields.io/badge/langgraph-1.x-1c3c3c.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

A standalone Python microservice that turns a **natural-language operator
command** into a **validated, executable, safety-checked drone mission plan** —
or a structured `NEEDS_CLARIFICATION` / `REJECTED` response. The service
plans; it never flies. A `MissionPlan` is the deliverable.

The point is to demonstrate production-grade agent engineering:
schema-constrained LLM output, a bounded self-correction loop, a deterministic
safety kernel, two LangGraph subgraphs (compile + inflight supervisor), full
observability, containerization, and Kubernetes deployment.

> **Reviewing this for the first time?** Start with
> **[`docs/REVIEWER_GUIDE.md`](./docs/REVIEWER_GUIDE.md)** — a 10-minute path
> through the four files that matter, plus the 30-minute extension.

---

## TL;DR

```sh
git clone https://github.com/kushy115/mission-tasking-service
cd mission-tasking-service
cp .env.example .env    # fill API_KEY at minimum
make demo               # docker compose up + seed + compile a sample mission
```

Open the UI at <http://localhost:8000>, Grafana at <http://localhost:3000>
(admin / admin). `make help` lists every developer target.

---

## Architecture

The compile flow is a LangGraph `StateGraph` with six nodes, a bounded repair
loop, and a human-approval interrupt. The full diagram lives in
[`docs/graph.mmd`](./docs/graph.mmd).

```
operator command + area + drone state
        │
        ▼
┌─────────────────── compile graph (LangGraph StateGraph) ───────────────────┐
│                                                                             │
│   intake ──► plan ──► validate ──► critique ──► advisor ──► finalize ──► END│
│     │         ▲          │  │                                    (interrupt)│
│     │         │          │  └──► clarify ──► finalize                       │
│     │         │          └──────► repair ──► (cap=3) ──► reject ──► finalize│
│     └──► clarify / reject                                                   │
│                                                                             │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   MissionPlan (READY_FOR_APPROVAL | NEEDS_CLARIFICATION | REJECTED)
        │
        ▼  (operator approves → live sim begins)
┌────────────── inflight supervisor subgraph (DD-014) ────────────────────────┐
│                                                                             │
│   telemetry tick + event → assess → decide → (CONTINUE | RTB_NOW |          │
│                                                DIVERT | REPLAN | LAND)      │
│   non-CONTINUE → replan via compile graph → kernel validates → swap in-place│
│                                                                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key design choices

- **`plan` is one direct LLM call, not a tool-calling agent.** Originally the
  planner was a `create_agent` + tools + `response_format=MissionPlan` setup,
  which made 5+ LLM calls per compile (one per tool decision plus a final
  structured-output pass). That was slow, expensive (Gemini free-tier was
  exhausted in 2–3 compiles), and the `response_format=Pydantic` + Gemini
  combo had a silent uvicorn-worker crash on response handling. We replaced
  it with a single `llm.invoke()` per attempt, inlining the geo context into
  the prompt and parsing JSON ourselves. ~1 LLM call per repair pass instead
  of 5+. The deterministic kernel is still the safety source of truth. See
  the big comment block above `plan_node` in `app/graph/nodes.py`.
- **The LLM never emits a plan as free text in practice.** The prompt
  requires a strict JSON envelope; we `json.loads()` it and validate against
  the `MissionPlan` Pydantic schema. If parsing or validation fails, the
  repair loop re-prompts.
- **The validator is deterministic Python, not the LLM.**
  `app/validation/kernel.py` is the source of truth on safety;
  `app/validation/physics.py` is the energy / time model. Both have no LLM
  and no network. Tests are mandatory.
- **Repair loop is bounded at 3.** A cap is a deliberate safety/cost ceiling.
  If three repair passes don't produce a safe plan, the service rejects.
- **Human-in-the-loop approval gate.** `finalize` interrupts the graph;
  resumption happens via `/v1/missions:approve`. Postgres-backed
  checkpointing means the interrupt survives restarts.
- **Two graphs, one kernel.** The compile graph plans pre-flight; the
  inflight supervisor replans in-flight. Both gate every decision through
  the same `kernel.validate(...)`.
- **Decoupled from autonomy.** MTS produces `MissionPlan`; a downstream
  consumer would execute it. No hardware. No proprietary inputs.

---

## Endpoints

| Method | Path                                       | Purpose                                               |
| ------ | ------------------------------------------ | ----------------------------------------------------- |
| GET    | `/healthz`                                 | Liveness                                              |
| GET    | `/readyz`                                  | Readiness (checks Postgres)                           |
| GET    | `/metrics`                                 | Prometheus scrape                                     |
| POST   | `/v1/missions:compile`                     | Compile a command into a plan                         |
| POST   | `/v1/missions:approve`                     | Approve or reject a ready plan                        |
| POST   | `/v1/missions/{mission_id}:verify`         | Run the executor over a plan                          |
| GET    | `/v1/missions`                             | List recent missions (history)                        |
| GET    | `/v1/missions/{mission_id}`                | Fetch one mission's full plan                         |
| GET    | `/v1/missions/{mission_id}/export`         | Export as `?format=kml\|gpx\|dji`                     |
| GET    | `/v1/areas`                                | List operating areas                                  |
| POST   | `/v1/areas`                                | Upsert an area drawn in the UI                        |
| DELETE | `/v1/areas/{area_id}`                      | Delete an operating area                              |
| GET    | `/v1/drones`                               | List drone profiles                                   |
| WS     | `/ws/missions/{mission_id}/sim`            | Live telemetry stream (`?speed=N&session_id=…`)       |
| POST   | `/v1/missions/{mission_id}/sim:inject`     | Inject in-flight event (DD-014; needs active session) |

---

## Feature index

Each feature has a numbered section in [`docs/DESIGN_DECISIONS.md`](./docs/DESIGN_DECISIONS.md)
with the *why* and the files it owns.

| #  | Feature                          | What it does                                                                                       |
| -- | -------------------------------- | -------------------------------------------------------------------------------------------------- |
| 1  | Mission history browser          | Side-drawer browser for past compilations; click to replay any plan on the map                     |
| 2  | KML / GPX / DJI export           | Download an approved plan in real GCS formats                                                      |
| 3  | Constraint violation overlay     | Rejected plans color-code waypoints on the map by severity                                         |
| 4  | Weather integration              | Open-Meteo (no key) + synthetic provider; wind penalty in physics; hard reject on threshold breach |
| 5  | Area editor                      | Leaflet.draw boundary + NFZ sketching; server-snapped home point                                   |
| 6  | LLM plan critique                | Second LLM pass after validation passes — advisory tactical confidence score                       |
| 7  | Alternative plans                | Opt-in `alternatives:true` returns 2–3 plans; each runs through the kernel independently           |
| 8  | Airspace deconfliction           | 7th kernel check — rejects plans conflicting with already-approved missions                        |
| 9  | Multi-drone coordination         | `drone_ids: [...]` → group compile with server-assigned altitude layers + takeoff staggers         |
| 10 | Live simulation WebSocket        | Animated drone marker flies the plan in real time; geofence breach auto-aborts                     |
| 14 | Inflight supervisor              | Closed-loop replanning during the live sim — wind / NFZ / fault events trigger LLM decisions       |
| 15 | Optimization advisor + per-mission chat | Advisory pass that suggests plan improvements; per-mission chat thread persisted in Postgres |

---

## Safety model

The validator enforces these constraints on every `READY_FOR_APPROVAL` plan:

1. Every waypoint inside the operating-area geofence.
2. No leg geometry intersects any NFZ.
3. Every altitude below the area ceiling and above the min-AGL floor (20m).
4. `battery_reserve_pct >= 20` at landing.
5. `total_duration_s` within rated endurance.
6. Search-pattern leg spacing within sensor swath at planned altitude.
7. Plan ends with a `RETURN_TO_BASE` leg.

Adding a constraint? Update `app/validation/kernel.py` AND its test in
`tests/test_validation.py`. The kernel is the safety core.

---

## Local development

```sh
# 1. install
make install

# 2. start stack + seed + sample compile
cp .env.example .env    # fill API_KEY at minimum
make demo

# 3. iterate
make test               # pytest -q
make lint               # ruff check + format check
make logs               # tail mts container logs
make compile-sample     # POST a sample mission

# 4. evals
make evals
```

Without Make, the underlying commands are documented in the Makefile and in
[`docs/REVIEWER_GUIDE.md`](./docs/REVIEWER_GUIDE.md). Or open
`quickstart.ipynb` for a step-by-step notebook walkthrough.

Grafana auto-provisions the MTS dashboard on `docker compose up` — just open
<http://localhost:3000>. The Prometheus data source is wired automatically.

---

## Kubernetes (kind)

End-to-end local cluster — Postgres + Redis + MTS + probes + Helm chart. See
[`docs/DESIGN_DECISIONS.md`](./docs/DESIGN_DECISIONS.md) DD-011 for the
rationale behind each step.

```sh
brew install helm kind   # one-time
make kind-up             # creates cluster, loads image, applies postgres+redis, helm installs
kubectl exec deploy/mts -- python -m scripts.seed_db
kubectl port-forward svc/mts 8001:80
# → http://localhost:8001
make kind-down           # tear down
```

### Why these chart choices

- **`replicas: 3`** — the service is stateless (durable state in Postgres +
  Redis) so it scales horizontally without coordination.
- **HPA on inflight requests, not CPU** — LLM calls are slow and I/O-bound.
  CPU underreports load while requests sit in the planner. Inflight-requests
  per pod (via Prometheus Adapter) is the honest signal of queueing pressure.
- **`PodDisruptionBudget` minAvailable=2** — keep a serving quorum during
  node drains and rolling updates.
- **Probes** — `/healthz` for liveness; `/readyz` for readiness (checks
  Postgres connectivity so traffic only hits pods that can serve);
  `startupProbe` to cover slow first-init when the graph compiles.
- **Nightly `CronJob`** — runs `evals/run_evals.py` against the deployed
  service to catch LLM regressions that unit tests cannot.

---

## Observability

**📖 See [`docs/observability.md`](./docs/observability.md) for the metric
interpretation guide** — every metric, healthy ranges, what to investigate
when something moves.

Three layers, all required:

- **LangSmith** — every node traversal, tool call I/O, LLM prompt/completion,
  token counts, and repair-loop counter. Tag runs with `mission_id`,
  `area_id`.
- **OpenTelemetry** — each compile request is a span tree: HTTP handler →
  graph run → each node → tool calls → PostGIS queries. Exports via OTLP.
- **Prometheus + Grafana** — `/metrics` exposes `mts_compile_requests_total`,
  `mts_compile_duration_seconds`, `mts_repair_loops`, `mts_rejections_total`,
  `mts_clarifications_total`, `mts_tokens_total`, `mts_tool_calls_total`.
  A rising repair-loop or rejection rate is the regression signal.

---

## Documentation

- [`AGENTS.md`](./AGENTS.md) — brief for AI coding assistants working in this repo.
- [`CLAUDE.md`](./CLAUDE.md) — the original product brief (longer, human-facing).
- [`docs/REVIEWER_GUIDE.md`](./docs/REVIEWER_GUIDE.md) — 10-min and 30-min reading paths.
- [`docs/DESIGN_DECISIONS.md`](./docs/DESIGN_DECISIONS.md) — running log of non-trivial choices.
- [`docs/observability.md`](./docs/observability.md) — metric interpretation guide.
- [`docs/IDEAS.md`](./docs/IDEAS.md) — parking lot for future features.
- [`docs/graph.mmd`](./docs/graph.mmd) — LangGraph compile diagram.

---

## Repository layout

```
mission-tasking-service/
├── app/
│   ├── api/             # FastAPI routes + request/response models
│   ├── schemas/         # MissionPlan + enums (the LLM-bound contract)
│   ├── graph/           # compile StateGraph (intake → plan → validate → …)
│   ├── supervisor/      # inflight supervisor subgraph (closed-loop replanning)
│   ├── tools/           # @tool suite the planning agent grounds against
│   ├── validation/      # SAFETY CORE — deterministic kernel + physics model
│   ├── geo/             # PostGIS store + search-pattern generators
│   ├── sim/             # plan executor + live telemetry stream
│   ├── export/          # KML / GPX / DJI Waypoint formatters
│   ├── weather/         # Open-Meteo + synthetic providers
│   ├── observability/   # logging, OTel, Prometheus
│   └── static/          # single-page UI (Leaflet + vanilla JS)
├── data/                # self-authored areas (GeoJSON) + drone profiles (YAML)
├── deploy/
│   ├── docker/          # Dockerfile, compose, Grafana provisioning
│   ├── helm/mts/        # Helm chart (Deployment, HPA, PDB, CronJob, …)
│   └── k8s/             # raw manifests (Postgres + Redis for kind)
├── docs/                # design decisions, reviewer guide, observability, ideas
├── evals/               # dataset + LangSmith runner
├── scripts/seed_db.py   # loads data/ into Postgres
├── tests/               # pytest suite (safety-core coverage mandatory)
├── AGENTS.md            # brief for AI coding assistants
├── CLAUDE.md            # original product brief
├── Makefile             # developer entry point (`make help`)
├── pyproject.toml       # uv-managed deps
└── quickstart.ipynb     # self-contained notebook walkthrough
```
