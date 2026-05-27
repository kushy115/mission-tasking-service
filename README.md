# Mission Tasking Service (MTS)

[![CI](https://github.com/kushy115/mission-tasking-service/actions/workflows/ci.yml/badge.svg)](https://github.com/kushy115/mission-tasking-service/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangChain 1.x](https://img.shields.io/badge/langchain-1.x-1c3c3c.svg)](https://docs.langchain.com/)
[![LangGraph 1.x](https://img.shields.io/badge/langgraph-1.x-1c3c3c.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Turns a plain-English drone command ("patrol the north perimeter and check the
fuel depot for heat anomalies") into a validated mission plan, or tells you
why it can't. MTS plans. It does not fly.

> Reviewing this for the first time? Start at
> [`docs/REVIEWER_GUIDE.md`](./docs/REVIEWER_GUIDE.md).

## Prerequisites

- Docker Desktop, running
- `make` (already on macOS once Xcode CLT is installed; Windows: use WSL2)
- An LLM API key from Anthropic, OpenAI, or Google. Default is Claude.

That's it. Python, `uv`, Postgres all run inside the container.

## Running it

```sh
git clone https://github.com/kushy115/mission-tasking-service
cd mission-tasking-service
cp .env.example .env       # set API_KEY=...
make demo
```

When that finishes you'll have the UI at <http://localhost:8000> and Grafana
at <http://localhost:3000> (admin / admin). `make down` stops everything,
`make clean` also wipes the Postgres volume. `make help` lists the rest.

## Architecture

```
operator command  →  compile graph  →  MissionPlan  →  approval  →  live sim
                                                                       │
                                                                       ▼
                                                            inflight supervisor
```

Two graphs share one safety kernel. The compile graph turns a command into a
plan; the inflight supervisor watches the live execution and can trigger a
replan mid-flight when conditions change (wind spike, pop-up NFZ, sensor
fault). Both paths run every candidate plan through the same deterministic
validator before anything is allowed forward.

The full LangGraph diagram is in [`docs/graph.mmd`](./docs/graph.mmd). Why it
looks the way it does (one LLM call instead of an agent, repair loop capped
at 3, Postgres-backed approval interrupt) is in
[`docs/DESIGN_DECISIONS.md`](./docs/DESIGN_DECISIONS.md). What the safety
kernel actually checks is in [`docs/SAFETY_MODEL.md`](./docs/SAFETY_MODEL.md).

## Tech stack

| Layer | Tool / library | Why it's used |
| --- | --- | --- |
| **Language / runtime** | Python 3.12+ | Type-hinted strict-`mypy` codebase; modern stdlib (`StrEnum`, structural pattern matching) |
| **Dependency / venv** | [`uv`](https://github.com/astral-sh/uv) | Single tool for env, lockfile (`uv.lock`), and `uv run` — no `pip` / `poetry` / `virtualenv` |
| **Web framework** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | Async HTTP + OpenAPI schema for free; WebSocket support for the live sim |
| **Schemas / validation** | Pydantic v2 | One schema per boundary; the `MissionPlan` model is what the LLM is constrained to and what the kernel validates |
| **LLM orchestration** | [LangChain 1.x](https://docs.langchain.com/) + [LangGraph 1.x](https://langchain-ai.github.io/langgraph/) | `StateGraph` for the compile flow + inflight supervisor; `interrupt` for the human-approval gate; Postgres-backed checkpointing |
| **LLM providers** | `langchain-anthropic`, `langchain-openai`, `langchain-google-genai` | Provider-agnostic. Pick via `MTS_LLM_PROVIDER`; one `API_KEY` slot for all |
| **Database** | PostgreSQL 16 + [PostGIS](https://postgis.net/) 3.4 | Geofence containment, NFZ intersection, airspace deconfliction all run as SQL spatial joins |
| **Spatial ops (in-proc)** | [Shapely](https://shapely.readthedocs.io/) 2.x | Polygon ops in the kernel + search-pattern generators, no DB hop in the hot path |
| **DB access** | SQLAlchemy 2.x + `psycopg` v3 | Type-checked Core API; PostGIS via `geoalchemy2` |
| **Cache** | Redis 7 | Idempotency keys on compile requests |
| **HTTP client** | `httpx` | Async; same library LangChain providers use under the hood |
| **Frontend** | Vanilla JS + [Leaflet](https://leafletjs.com/) + Leaflet.draw | Single file at `app/static/index.html`. No bundler, no framework |
| **LLM tracing** | [LangSmith](https://smith.langchain.com/) | Per-node traversal, tool I/O, tokens, repair-loop counter |
| **Service tracing** | OpenTelemetry (OTLP) | FastAPI + SQLAlchemy instrumentation; one span tree per compile |
| **Metrics** | `prometheus-client` → Prometheus → Grafana | `/metrics` scraped on a 15 s interval; Grafana dashboard auto-provisioned |
| **Containers** | Docker (multi-stage) + Compose | One image used unchanged by compose and kind; non-root user; healthcheck on `/healthz` |
| **Orchestration** | Helm chart on [kind](https://kind.sigs.k8s.io/) | `deploy/helm/mts/` + `values-kind.yaml` for the local path |
| **CI** | GitHub Actions | `ruff`, `mypy --strict`, `pytest` — all gating |
| **Testing** | `pytest` + `pytest-asyncio` | 62 tests including the safety core |
| **Evals** | LangSmith evaluation harness | `evals/dataset.jsonl`; scheduled by the Helm CronJob |

## Endpoints

| Method | Path                                       | Purpose                                                            |
| ------ | ------------------------------------------ | ------------------------------------------------------------------ |
| GET    | `/healthz`                                 | Liveness                                                           |
| GET    | `/readyz`                                  | Readiness (checks Postgres)                                        |
| GET    | `/metrics`                                 | Prometheus scrape                                                  |
| POST   | `/v1/missions:compile`                     | Compile a command into a plan                                      |
| POST   | `/v1/missions:approve`                     | Approve or reject a ready plan                                     |
| POST   | `/v1/missions/{mission_id}:verify`         | Run the executor over an approved plan                             |
| POST   | `/v1/missions/{mission_id}:chat`           | Follow-up Q&A about a compiled plan                                |
| GET    | `/v1/missions`                             | List recent missions                                               |
| GET    | `/v1/missions/{mission_id}`                | Fetch one mission's full plan + repair-draft timeline              |
| GET    | `/v1/missions/{mission_id}/export`         | Export as `?format=kml\|gpx\|dji`                                  |
| GET    | `/v1/areas`                                | List operating areas                                               |
| POST   | `/v1/areas`                                | Upsert an area drawn in the UI                                     |
| POST   | `/v1/areas:research`                       | LLM-assisted area research (ceiling, NFZ hints)                    |
| DELETE | `/v1/areas/{area_id}`                      | Delete an operating area                                           |
| GET    | `/v1/drones`                               | List drone profiles                                                |
| POST   | `/v1/drones`                               | Upsert a drone profile                                             |
| WS     | `/ws/missions/{mission_id}/sim`            | Live telemetry stream (`?speed=N&session_id=…`)                    |
| POST   | `/v1/missions/{mission_id}/sim:inject`     | Inject an in-flight event                                          |

## Kubernetes (kind)

`make kind-up` brings up a local cluster with Postgres + Redis + MTS + probes
+ the Helm chart. The chart image is `docker-mts:latest`, so `make up` (or
`make demo`) has to have run once first to build it.

```sh
brew install helm kind          # one-time
make up                         # builds the image
make kind-up                    # installs the chart into a fresh kind cluster
kubectl exec deploy/mts -- python -m scripts.seed_db
kubectl port-forward svc/mts 8001:80
# → http://localhost:8001
make kind-down
```

Iterating on code: rebuild with `make up`, run `make kind-up` again, then
`kubectl rollout restart deploy/mts` to pick up the new image. Helm won't roll
an unchanged spec.

The chart's design (HPA on inflight requests, PDB minAvailable=2, nightly
eval CronJob, etc.) is in DD-011.

## Docs

- [`docs/REVIEWER_GUIDE.md`](./docs/REVIEWER_GUIDE.md) — 10-minute reading path
- [`docs/DESIGN_DECISIONS.md`](./docs/DESIGN_DECISIONS.md) — the why behind each non-trivial choice
- [`docs/SAFETY_MODEL.md`](./docs/SAFETY_MODEL.md) — what the kernel enforces
- [`docs/observability.md`](./docs/observability.md) — what every metric means and what its healthy range looks like
- [`docs/graph.mmd`](./docs/graph.mmd) — full LangGraph diagram
