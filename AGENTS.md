# AGENTS.md

Quick brief for AI coding assistants (Claude Code, Cursor, Copilot, etc.)
working in this repo. The longer human-facing brief is in [CLAUDE.md](./CLAUDE.md);
the running architectural log is in [docs/DESIGN_DECISIONS.md](./docs/DESIGN_DECISIONS.md).

## Project in one line

A Python microservice that turns a natural-language operator command into a
**validated, safety-checked drone mission plan** using LangChain + LangGraph,
backed by a deterministic Python safety kernel.

## Stack

Python 3.12 · uv · FastAPI · pydantic v2 · LangChain 1.x · LangGraph 1.x
(Postgres-checkpointed) · PostgreSQL 16 + PostGIS · Redis · OpenTelemetry ·
Prometheus + Grafana · LangSmith · Docker · Helm.

## The two rules that override everything

1. **The LLM never decides safety.** `app/validation/kernel.py` (waypoint /
   geofence / NFZ / battery / endurance / sensor coverage / RTB) and
   `app/validation/physics.py` (energy + time model) are the safety source of
   truth. Pure functions, no LLM, no network. **Touch a constraint → update
   its test in `tests/test_validation.py`.**
2. **The repair loop is bounded at 3.** This is a deliberate cost / safety
   ceiling. Do not raise it without an entry in `docs/DESIGN_DECISIONS.md`
   explaining why.

## File map (where to make changes)

| Concern                          | File                                         |
| -------------------------------- | -------------------------------------------- |
| API contract                     | `app/api/routes.py`, `app/api/models.py`     |
| Plan schema (LLM-bound)          | `app/schemas/plan.py`, `app/schemas/enums.py` |
| Compile graph (intake → finalize) | `app/graph/builder.py`, `app/graph/nodes.py` |
| Inflight supervisor subgraph     | `app/supervisor/`                            |
| Tools for the planning agent     | `app/tools/planning_tools.py`                |
| Safety core (deterministic)      | `app/validation/kernel.py`, `app/validation/physics.py` |
| Geo + PostGIS                    | `app/geo/store.py`, `app/geo/patterns.py`    |
| Live simulation stream           | `app/sim/stream.py`, `app/sim/executor.py`   |
| Plan export                      | `app/export/formats.py`                      |
| Weather provider                 | `app/weather/provider.py`                    |
| Observability                    | `app/observability/`                         |
| UI (single page)                 | `app/static/index.html`                      |
| Test data                        | `data/areas/*.geojson`, `data/drones/*.yaml` |

## LangChain / LangGraph house rules

- Use `create_agent` from the `langchain` package, **not** `create_react_agent`
  from `langgraph.prebuilt` (deprecated).
- The `plan` node is one direct `llm.invoke()` per attempt, not a tool-calling
  agent. The agent variant was tried and replaced — see the big comment block
  above `plan_node` in `app/graph/nodes.py`. Don't re-introduce a tool loop
  in `plan` without reading that block.
- Custom graph state lives in `app/graph/state.py` as a `TypedDict`. Use
  `Annotated[list, add_messages]` for message channels.
- Checkpointer is `langgraph.checkpoint.postgres.PostgresSaver`. `MemorySaver`
  is for unit tests only.
- Human-in-the-loop pause uses LangGraph's `interrupt` primitive at the
  `finalize` node; resumption comes from `POST /v1/missions:approve`.

## Workflow

```sh
make install        # uv sync --extra dev
make test           # pytest -q
make lint           # ruff check + format check
make demo           # docker compose up + seed + compile a sample mission
```

## Before you commit

- `make lint && make test` — both must pass.
- If you touched `app/validation/`, the matching test in `tests/` must cover
  the new branch.
- If you added a non-trivial feature or made a design call, add a section to
  `docs/DESIGN_DECISIONS.md`. Date it, name the files it owns, write the
  *why*. This is the audit log future-you and reviewers will read.
- If you added or removed an env var, update `.env.example`.
- If you added a route, update the endpoint table in `README.md`.

## What not to do

- Don't add fallbacks, mocks, or backwards-compatibility shims for scenarios
  that can't happen.
- Don't make the LLM emit a plan as free text and parse it loose — the schema
  is the contract.
- Don't wire MTS to any real flight controller, MAVLink, or hardware. The
  plan is the deliverable.
- Don't bake secrets into images or commit `.env`. Use the `Secret` template
  in the Helm chart.
