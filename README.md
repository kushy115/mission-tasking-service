# Mission Tasking Service (MTS)

A standalone Python microservice that turns a **natural-language operator command**
into a **validated, executable, safety-checked drone mission plan** — or a
structured `NEEDS_CLARIFICATION` / `REJECTED` response. The service plans; it
never flies. A `MissionPlan` is the deliverable.

The point of the project is to demonstrate production-grade agent engineering:
schema-constrained LLM output, a bounded self-correction loop, a deterministic
safety kernel, full observability, containerization, and Kubernetes deployment.

## Architecture

```
operator command + area + drone state
        │
        ▼
┌─────────────────── compile graph (LangGraph StateGraph) ───────────────────┐
│                                                                             │
│   intake ──► plan ──► validate ──► finalize ──► END (interrupt at approval) │
│     │         ▲          │  │                                               │
│     │         │          │  └──► clarify ──► finalize                       │
│     │         │          └──────► repair ──► (cap=3) ──► reject ──► finalize│
│     └──► clarify / reject                                                   │
│                                                                             │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   MissionPlan (READY_FOR_APPROVAL | NEEDS_CLARIFICATION | REJECTED)
```

Key design choices:

- **`plan` node = ONE direct LLM call, NOT a tool-calling agent.** Originally
the planner was a `create_agent` + tools + `response_format=MissionPlan` setup,
which made 5+ LLM calls per compile (one per tool decision plus a final
structured-output pass). That was slow, expensive (free-tier Gemini ran out of
its 20 RPD quota in 2–3 compiles), and the `response_format=Pydantic` + Gemini
combo was fragile — silent uvicorn worker death on response handling. We
replaced it with a single `llm.invoke()` per attempt, inlining the geo context
into the prompt and parsing JSON ourselves. ~1 LLM call per repair pass instead
of 5+. The deterministic kernel is still the safety source of truth. See the
big comment block above `plan_node` in `app/graph/nodes.py`.
- **The LLM never emits a plan as free text in practice.** The prompt requires
a strict JSON envelope; we `json.loads()` it and validate it against the
`MissionPlan` Pydantic schema. If parsing or validation fails, the repair loop
re-prompts.
- **The validator is deterministic Python, not the LLM.** `app/validation/kernel.py`
is the source of truth on safety; `app/validation/physics.py` is the
energy/time model. Both have no LLM and no network. Tests are mandatory.
- **Repair loop is bounded at 3.** A cap is a deliberate safety/cost ceiling.
If three repair passes don't produce a safe plan, the service rejects.
- **Human-in-the-loop approval gate.** `finalize` interrupts the graph;
resumption happens via `/v1/missions:approve`. Postgres-backed checkpointing
means the interrupt survives restarts.
- **Decoupled from autonomy.** MTS produces `MissionPlan`; a downstream
consumer would execute it. No hardware. No proprietary inputs.

See `docs/graph.mmd` for the LangGraph diagram.

## Endpoints


| Method | Path                               | Purpose                        |
| ------ | ---------------------------------- | ------------------------------ |
| GET    | `/healthz`                         | Liveness                       |
| GET    | `/readyz`                          | Readiness (checks Postgres)    |
| GET    | `/metrics`                         | Prometheus scrape              |
| POST   | `/v1/missions:compile`             | Compile a command into a plan  |
| POST   | `/v1/missions:approve`             | Approve or reject a ready plan |
| POST   | `/v1/missions/{mission_id}:verify` | Run the executor over a plan   |


## Local development

```sh
# 1. install
uv sync

# 2. start dependencies + service
cp .env.example .env  # fill API_KEY at minimum
docker compose -f deploy/docker/docker-compose.yml up --build -d

# 3. seed test data (areas + drone profiles)
docker compose -f deploy/docker/docker-compose.yml exec mts \
    python scripts/seed_db.py

# 4. compile a mission
curl -s http://localhost:8000/v1/missions:compile -H 'content-type: application/json' -d '{
  "command": "Patrol the yard perimeter at 60 meters with EO.",
  "area_id": "yard-simple",
  "drone_state": {"drone_profile_id": "long-endurance-quad", "battery_pct": 100}
}' | jq

# 5. tests
uv run pytest

# 6. evals
uv run python evals/run_evals.py
```

Open Grafana at [http://localhost:3000](http://localhost:3000) (admin/admin) and import
`deploy/grafana-dashboard.json`.

## Kubernetes (kind)

```sh
kind create cluster
docker build -t mts:0.1.0 -f deploy/docker/Dockerfile .
kind load docker-image mts:0.1.0

helm install mts deploy/helm/mts/ \
  --set image.tag=0.1.0 \
  --set secrets.API_KEY=$API_KEY
```

### Why these chart choices

- `**replicas: 3**` — the service is stateless (durable state in Postgres + Redis),
so it scales horizontally without coordination.
- `**HorizontalPodAutoscaler` on inflight requests, not CPU** — LLM calls are
slow and I/O-bound. CPU underreports load while requests sit in the planner.
Inflight-requests-per-pod (via Prometheus Adapter) is the honest signal of
queueing pressure.
- `**PodDisruptionBudget` minAvailable=2** — keep a serving quorum during node
drains and rolling updates.
- **Probes** — `/healthz` for liveness; `/readyz` for readiness (checks Postgres
connectivity so traffic only hits pods that can serve); `startupProbe` to
cover slow first-init when the graph compiles.
- **Nightly `CronJob`** — runs `evals/run_evals.py` against the deployed
service to catch LLM regressions that unit tests cannot.

## Observability

**📖 See [docs/observability.md](docs/observability.md) for the metric
interpretation guide** — every metric, healthy ranges, what to investigate when
something moves. The Grafana dashboard is auto-provisioned on
`docker compose up`; just open <http://localhost:3000>.

Three layers, all required:

- **LangSmith** — every node traversal, tool call I/O, LLM prompt/completion,
token counts, and repair-loop counter. Tag runs with `mission_id`, `area_id`.
- **OpenTelemetry** — each compile request is a span tree: HTTP handler →
graph run → each node → tool calls → PostGIS queries. Exports via OTLP.
- **Prometheus + Grafana** — `/metrics` exposes `mts_compile_requests_total`,
`mts_compile_duration_seconds`, `mts_repair_loops`, `mts_rejections_total`,
`mts_clarifications_total`, `mts_tokens_total`, `mts_tool_calls_total`.
A rising repair-loop or rejection rate is the regression signal.

## Safety model

The validator enforces these constraints on every `READY_FOR_APPROVAL` plan:

1. Every waypoint inside the operating-area geofence.
2. No leg geometry intersects any NFZ.
3. Every altitude below the area ceiling and above the min-AGL floor (20m).
4. `battery_reserve_pct >= 20` at landing.
5. `total_duration_s` within rated endurance.
6. Search-pattern leg spacing within sensor swath at planned altitude.
7. Plan ends with a `RETURN_TO_BASE` leg.

Adding a constraint? Update `app/validation/kernel.py` AND its tests in
`tests/test_validation.py`. The kernel is the safety core.

## Repository layout

```
mission-tasking-service/
├── app/                # service code (api, schemas, graph, tools, validation, geo, sim, observability)
├── data/               # self-authored areas (GeoJSON) + drone profiles (YAML)
├── deploy/             # Dockerfile, compose, Helm chart, dashboards
├── docs/graph.mmd      # LangGraph diagram
├── evals/              # dataset + LangSmith runner
├── scripts/seed_db.py  # loads data/ into Postgres
├── tests/              # pytest suite (safety-core coverage mandatory)
├── CLAUDE.md           # original brief
├── pyproject.toml      # uv-managed deps
└── README.md
```

