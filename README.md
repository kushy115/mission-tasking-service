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

## Prerequisites

Everything runs inside Docker for the demo path, so the host install is small:

- **Docker Desktop** running (Mac, Linux, or Windows). On Apple Silicon Macs use the standard ARM build — `postgis/postgis:16-3.4` and all other images have arm64 variants.
- **`make`** — preinstalled on macOS once Xcode Command Line Tools are present (`xcode-select --install` if not). Linux: `apt install make`. Windows: use WSL2.
- **~6 GB free RAM** for the six-container stack.
- **One LLM API key.** Default is Anthropic (Claude). Any of these work — pick one and fill it into `.env`:

  | Provider | `MTS_LLM_PROVIDER` | Sample `MTS_LLM_MODEL` | `API_KEY` format |
  | --- | --- | --- | --- |
  | Anthropic *(default)* | `anthropic` | `claude-haiku-4-5` | `sk-ant-...` |
  | OpenAI | `openai` | `gpt-4o-mini` | `sk-...` |
  | Google | `google_genai` | `gemini-2.5-flash` | `AIzaSy...` |

No system Python or `uv` install is needed for the demo — both live inside the container. The host-side `uv`/`mypy`/`pytest` workflow (covered under [Local development](#local-development)) is for editing the code.

---

## TL;DR

```sh
git clone https://github.com/kushy115/mission-tasking-service
cd mission-tasking-service
cp .env.example .env                            # then edit and set API_KEY=...
make demo                                       # build + up + seed + sample compile
```

When `make demo` finishes you'll see:

- **UI** — <http://localhost:8000> (draw an area, compile a mission, watch the live sim)
- **Grafana** — <http://localhost:3000> (admin / admin; the MTS dashboard auto-provisions)
- **Prometheus** — <http://localhost:9090>

Stop with `make down` (preserves the Postgres volume) or `make clean` (wipes it). `make help` lists every target.

---

## Architecture

The compile flow is a LangGraph `StateGraph` with nine nodes (the happy path is
six: `intake → plan → validate → critique → advisor → finalize`, plus the three
escape hatches `repair`, `clarify`, `reject`), a bounded repair loop, and a
human-approval interrupt. The full diagram lives in
[`docs/graph.mmd`](./docs/graph.mmd) and is regenerated from the actual compiled
graph.

```
operator command + area + drone state
        │
        ▼
┌─────────────────── compile graph (LangGraph StateGraph) ───────────────────┐
│                                                                             │
│   intake ──► plan ──► validate ──► critique ──► advisor ──► finalize ──► END│
│     │                    │                                     (interrupt)  │
│     │                    ├──► repair ──► plan        (loop, cap=3)          │
│     │                    ├──► clarify ──► finalize                          │
│     │                    └──► reject ──► finalize    (cap exceeded /        │
│     │                                                 unsafe / NFZ-locked)  │
│     └──► clarify / reject ──► finalize    (ambiguous command / no clearance)│
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
│                                                REPLAN | EMERGENCY_LAND)     │
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

## Tech stack

| Layer | Tool / library | Why it's used |
| --- | --- | --- |
| **Language / runtime** | Python 3.12+ | Type-hinted strict-`mypy` codebase; modern stdlib (`StrEnum`, structural pattern matching) |
| **Dependency / venv** | [`uv`](https://github.com/astral-sh/uv) | Single tool for env, lockfile (`uv.lock`), and `uv run` — no `pip` / `poetry` / `virtualenv` |
| **Web framework** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | Async HTTP + OpenAPI schema for free; WebSocket support for the live sim |
| **Schemas / validation** | Pydantic v2 | One schema per boundary; the `MissionPlan` Pydantic model is what the LLM is constrained to and what the kernel validates |
| **LLM orchestration** | [LangChain 1.x](https://docs.langchain.com/) + [LangGraph 1.x](https://langchain-ai.github.io/langgraph/) | `StateGraph` for the compile flow + inflight supervisor; `interrupt` for the human-approval gate; `langgraph.checkpoint.postgres` for durable interrupt state |
| **LLM providers** | `langchain-anthropic`, `langchain-openai`, `langchain-google-genai` | Provider-agnostic — pick via `MTS_LLM_PROVIDER` env var; same `API_KEY` slot for all (see DD-002) |
| **Database** | PostgreSQL 16 + [PostGIS](https://postgis.net/) 3.4 | Geofence containment, NFZ intersection, and airspace-deconfliction queries all run as SQL spatial joins, not Python loops |
| **Spatial ops (in-proc)** | [Shapely](https://shapely.readthedocs.io/) 2.x | Pure-Python polygon ops in the kernel + search-pattern generators (no DB round-trip in the hot path) |
| **DB access** | SQLAlchemy 2.x + `psycopg` v3 | Type-checked Core API; `Engine` reused across requests; PostGIS via `geoalchemy2` |
| **Cache / idempotency** | Redis 7 | Idempotency keys on compile requests |
| **HTTP client (LLM + weather)** | `httpx` | Async-ready; same library used by LangChain providers |
| **Frontend** | Vanilla JS + [Leaflet](https://leafletjs.com/) + Leaflet.draw | Single-file SPA in `app/static/index.html` — no build step, no bundler, no framework. Renders the map, animates the live sim, draws areas |
| **LLM tracing** | [LangSmith](https://smith.langchain.com/) | Per-node traversal, tool I/O, tokens, repair-loop counter — tagged with `mission_id` / `area_id` |
| **Service tracing** | OpenTelemetry (OTLP) | FastAPI + SQLAlchemy instrumentation; one span tree per compile |
| **Metrics** | `prometheus-client` → Prometheus → Grafana | `/metrics` endpoint scraped on a 15 s interval; Grafana dashboard auto-provisioned in compose |
| **Containers** | Docker (multi-stage) + Docker Compose | One image (`docker-mts`) used unchanged by compose + kind; non-root user; `HEALTHCHECK` on `/healthz` |
| **Orchestration** | Kubernetes via Helm; local dev on [kind](https://kind.sigs.k8s.io/) | Chart in `deploy/helm/mts/`; `values-kind.yaml` overrides for the local path |
| **CI** | GitHub Actions (`.github/workflows/ci.yml`) | `ruff` lint + format check, `mypy` strict, `pytest` — all gating |
| **Testing** | `pytest` + `pytest-asyncio` | 62 tests including the safety-core (`test_validation.py`, `test_physics.py`) |
| **Lint / format / types** | `ruff` + `mypy` (strict, with `pydantic.mypy` plugin) | All three gates green on `main` |
| **Evals** | LangSmith evaluation harness | `evals/dataset.jsonl` + `evals/run_evals.py`; scheduled by the Helm CronJob |

---

## Endpoints

| Method | Path                                       | Purpose                                                            |
| ------ | ------------------------------------------ | ------------------------------------------------------------------ |
| GET    | `/healthz`                                 | Liveness                                                           |
| GET    | `/readyz`                                  | Readiness (checks Postgres)                                        |
| GET    | `/metrics`                                 | Prometheus scrape                                                  |
| POST   | `/v1/missions:compile`                     | Compile a command into a plan                                      |
| POST   | `/v1/missions:approve`                     | Approve or reject a ready plan                                     |
| POST   | `/v1/missions/{mission_id}:verify`         | Run the executor over an approved plan                             |
| POST   | `/v1/missions/{mission_id}:chat`           | Follow-up Q&A about a compiled plan (per-mission chat, DD-015)     |
| GET    | `/v1/missions`                             | List recent missions (history)                                     |
| GET    | `/v1/missions/{mission_id}`                | Fetch one mission's full plan + repair-draft timeline              |
| GET    | `/v1/missions/{mission_id}/export`         | Export as `?format=kml\|gpx\|dji`                                  |
| GET    | `/v1/areas`                                | List operating areas                                               |
| POST   | `/v1/areas`                                | Upsert an area drawn in the UI                                     |
| POST   | `/v1/areas:research`                       | LLM-assisted area research (ceiling, NFZ hints) — DD-007           |
| DELETE | `/v1/areas/{area_id}`                      | Delete an operating area                                           |
| GET    | `/v1/drones`                               | List drone profiles                                                |
| POST   | `/v1/drones`                               | Upsert an operator-authored drone profile — DD-006                 |
| WS     | `/ws/missions/{mission_id}/sim`            | Live telemetry stream (`?speed=N&session_id=…`)                    |
| POST   | `/v1/missions/{mission_id}/sim:inject`     | Inject in-flight event (DD-014; needs active session)              |

---

## Feature index

The numbering in this table tracks `§1`–`§10` in
[`docs/DESIGN_DECISIONS.md`](./docs/DESIGN_DECISIONS.md), then jumps to `#14`
and `#15` because numbers 11–13 are taken by **design decisions** (`DD-011`
local kind cluster, `DD-012` MCP servers in the dev workflow, `DD-013`
`est_battery_pct` semantics) — those are infrastructure / semantic choices
without a user-facing feature surface. DD-006 (drone profile editor) and
DD-007 (LLM area research) are also documented in `DESIGN_DECISIONS.md` and
each backs a real endpoint listed above.

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

This is the host-side workflow (editing the code, running tests / lint / mypy).
The demo path in TL;DR doesn't need any of this.

```sh
# 1. install host-side deps (creates .venv via uv)
make install

# 2. start stack + seed + sample compile
cp .env.example .env    # fill API_KEY=… (see Prerequisites)
make demo

# 3. iterate
make test               # pytest -q (62 tests)
make lint               # ruff check + format check
make typecheck          # mypy strict
make logs               # tail mts container logs
make compile-sample     # POST a sample mission
make down               # stop the stack (preserves Postgres volume)
make clean              # stop + remove volumes (full reset)

# 4. evals — needs LANGSMITH_API_KEY in .env
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
brew install helm kind   # one-time (Linux: use the official installers instead)

# kind-up loads `docker-mts:latest` into the cluster, so the image must exist
# first. `make up` (or `make demo`) builds it once via compose; after that you
# can switch freely between compose and kind.
make up                  # builds docker-mts:latest; safe to `make down` right after
make kind-up             # creates cluster, loads image, applies postgres+redis, helm installs
kubectl exec deploy/mts -- python -m scripts.seed_db
kubectl port-forward svc/mts 8001:80
# → http://localhost:8001
make kind-down           # tear down
```

> **Iterating?** `helm upgrade` on an unchanged image / ConfigMap won't roll
> the pods. After rebuilding the image (`make up`), run `make kind-up` again
> followed by `kubectl rollout restart deploy/mts` to pick up the new code.

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
