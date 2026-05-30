# Perception

[![CI](https://github.com/kushy115/mission-tasking-service/actions/workflows/ci.yml/badge.svg)](https://github.com/kushy115/mission-tasking-service/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangChain 1.x](https://img.shields.io/badge/langchain-1.x-1c3c3c.svg)](https://docs.langchain.com/)
[![LangGraph 1.x](https://img.shields.io/badge/langgraph-1.x-1c3c3c.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Turns a plain-English drone command ("patrol the north perimeter and check the
fuel depot for heat anomalies") into a validated mission plan, or tells you
why it can't. Perception plans. It does not fly.

> Reviewing this for the first time? Start at
> [`docs/GUIDE.md`](./docs/GUIDE.md).

## Prerequisites

- A Docker-compatible engine, running. Either:
  - **Docker Desktop** (easiest, GUI), or
  - **Colima** — free, CLI-only, no Docker Desktop required (see [Docker-Desktop-free path](#docker-desktop-free-path-colima) below). Rancher Desktop and Podman also work.
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

### Docker-Desktop-free path (Colima)

If you don't want Docker Desktop (license, RAM footprint, or just preference),
use [Colima](https://github.com/abiosoft/colima). It runs a small Linux VM and
exposes the standard `docker` + `docker compose` CLIs, so the Makefile targets
below work unchanged.

```sh
brew install colima docker docker-compose   # one-time
make colima-up                              # starts the VM (4 CPU / 6 GB)
make demo                                   # same as the Docker Desktop path
# ... when done:
make colima-down                            # stops the VM
```

`make colima-up` is idempotent — re-running it on an already-running VM is a
no-op. The VM persists across reboots until you `colima delete`.

Other engines that work the same way (same `docker`/`docker compose` CLI, no
Makefile changes needed):
- **Rancher Desktop** — GUI; enable the `dockerd (moby)` runtime in settings.
- **Podman** — `brew install podman podman-compose`, then `podman machine init && podman machine start`, then alias `docker=podman` and `docker compose=podman-compose`.
- **OrbStack** — drop-in replacement on macOS.

When that finishes you'll have the UI at <http://localhost:8000> and Grafana
at <http://localhost:3000> (admin / admin). `make down` stops everything,
`make clean` also wipes the Postgres volume. `make help` lists the rest.

### ⚠️ Planning effort & response time — read this before your first compile

The Compose form has a **Planning effort** dropdown that controls how the planner
runs. **This directly affects how long a compile takes**, so pick the right one
for your API key:

| Effort | What it does | Model calls / drone | Use it when |
|--------|--------------|---------------------|-------------|
| **Fast** | one direct LLM call (no agent) | **1** | your API key has **low rate limits** — this is ~as quick as a single chat message |
| **Balanced** (default) | a tool-calling agent that grounds the plan with geofence/battery/coverage tools | ~4–6 | you have **healthy rate limits** |
| **Thorough** | the agent on the strongest model (Opus) | more | you have **high rate limits** and want max quality |

**Important:** Balanced and Thorough make **several model calls per drone**, and a
multi-drone group multiplies that (5 drones ≈ 30 calls in a burst). On a
**low-rate-limit key (e.g. Anthropic tier 1) these tiers can take minutes** — the
service is correct but spends most of that time in API rate-limit back-off
(429s), which you'll see in `docker compose logs mts`. **That is a rate limit,
not a bug.** For snappy responses at Balanced or above you need **high API rate
limits**; for a constrained key, use **Fast** (and/or a light model).

To make Fast the default for everyone, set `MTS_DEFAULT_PLANNING_EFFORT=fast` in
`.env`. Per-tier models are set with `MTS_PLAN_MODEL_FAST` / `_BALANCED` /
`_THOROUGH` (defaults are Anthropic ids — change them to match
`MTS_LLM_PROVIDER`). See [`docs/DESIGN_DECISIONS.md` DD-017](docs/DESIGN_DECISIONS.md).

> **Set your key first.** `cp .env.example .env` then set `API_KEY=` to a key
> matching `MTS_LLM_PROVIDER` (defaults to Anthropic Claude). Without a valid key
> every compile is rejected at the plan step.

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
looks the way it does (a tool-calling planning agent with a fast single-call
tier, repair loop capped at 3, Postgres-backed approval interrupt) is in
[`docs/DESIGN_DECISIONS.md`](./docs/DESIGN_DECISIONS.md). What the safety
kernel actually checks is in [`docs/GUIDE.md`](./docs/GUIDE.md#the-safety-model).

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

`make kind-up` brings up a local cluster with Postgres + Redis + Perception + probes
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

- [`docs/GUIDE.md`](./docs/GUIDE.md) — orientation + reading paths, the safety model, and the metrics reference (for a new reader / reviewer)
- [`docs/LANGCHAIN.md`](./docs/LANGCHAIN.md) — how the LangChain/LangGraph orchestration works, chain by chain
- [`docs/DESIGN_DECISIONS.md`](./docs/DESIGN_DECISIONS.md) — the why behind each non-trivial choice (the decision log)
- [`docs/graph.mmd`](./docs/graph.mmd) — full LangGraph diagram
