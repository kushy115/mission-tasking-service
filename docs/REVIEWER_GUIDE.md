# Reviewer Guide

A short, opinionated path through the codebase for someone who has 10–30
minutes and wants to evaluate the engineering, not the surface area.

If you only have time to read **one file**, read
[`app/validation/kernel.py`](../app/validation/kernel.py) — that is the safety
core, and it is deliberately the most boring file in the repo.

---

## 10-minute path — "is the agent engineering sound?"

Read these four files in order:

1. **[`app/schemas/plan.py`](../app/schemas/plan.py)** — the `MissionPlan`
   contract the LLM is constrained to. Pydantic v2. This is what the LLM
   produces and what the validator checks. If you understand this schema, the
   rest of the system falls out of it.

2. **[`app/graph/builder.py`](../app/graph/builder.py)** — the LangGraph
   `StateGraph`. Six nodes, three conditional edges, one `interrupt`. The
   shape is `intake → plan → validate → (finalize | repair → plan | clarify)`.
   The repair loop is bounded at 3 — see why in `docs/DESIGN_DECISIONS.md`.

3. **[`app/graph/nodes.py`](../app/graph/nodes.py)** — the node
   implementations. The big comment block above `plan_node` documents *why*
   the planner is a single `llm.invoke()` instead of a tool-calling agent
   (cost, latency, and a real provider failure mode). This is the design
   call we'd most expect to be challenged on.

4. **[`app/validation/kernel.py`](../app/validation/kernel.py)** — the
   deterministic safety kernel. Seven checks, no LLM, no network, fully
   unit-tested in [`tests/test_validation.py`](../tests/test_validation.py).
   This is the kernel the LLM is gated by. If the LLM hallucinates, this
   catches it. The repair loop feeds the kernel's errors back into the next
   `plan` attempt.

That's the agent engineering core. The rest of the repo is plumbing, UI,
ops, and one second graph (the inflight supervisor).

---

## 30-minute path — "is the productionization real?"

Add these:

5. **[`docs/DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md)** — the running log of
   non-trivial choices, dated, each tagged with the files it owns. ~14
   features × ~50 lines each. Read DD-01 (the agent-vs-direct-call rewrite)
   and DD-014 (the inflight supervisor).

6. **[`deploy/helm/mts/`](../deploy/helm/mts/)** — Helm chart. Each template
   has a one-line comment explaining *why* that resource is there. Of
   particular interest: the `HorizontalPodAutoscaler` scales on inflight
   requests rather than CPU because LLM calls are I/O-bound and CPU
   underreports load.

7. **[`app/observability/`](../app/observability/)** + the auto-provisioned
   Grafana dashboard in [`deploy/docker/grafana/`](../deploy/docker/grafana/).
   Three layers: LangSmith for LLM-level traces, OTel for service spans,
   Prometheus for ops metrics. `docs/observability.md` explains what each
   metric means and what to investigate when it moves.

8. **[`app/supervisor/`](../app/supervisor/)** — second LangGraph subgraph
   that runs *during* the live sim. Owns the closed-loop replanning story
   ("operator approves a patrol → wind spike event → supervisor decides
   REPLAN_FROM_HERE → new plan re-runs through the same kernel → swapped
   in-place on the WebSocket stream").

---

## What to look for

- **Schema-first LLM I/O.** The LLM is bound to `MissionPlan`; we
  `json.loads()` the response and Pydantic-validate it. If parsing or
  validation fails, the repair loop re-prompts with the error injected.
- **Deterministic safety core.** Pure Python, fully testable, never an LLM
  call. The LLM proposes; the kernel disposes.
- **Bounded self-correction.** Repair loop cap of 3 is a documented
  cost/safety ceiling, not a magic number.
- **Two graphs, one kernel.** The compile graph plans pre-flight; the
  supervisor subgraph replans in-flight. Both end every decision through the
  same `kernel.validate(...)` call.
- **Stateless service, stateful planner.** Postgres holds the
  LangGraph checkpoints so `interrupt` survives restarts and pods scale
  horizontally.
- **Observability isn't an afterthought.** Every node emits a span, every
  tool call is counted, every LLM call's tokens are tracked. A rising
  repair-loop rate is the regression signal we'd alert on first.

---

## What to push back on (we expect it)

- **"Why a custom `StateGraph` and not just `create_agent`?"** — because we
  need conditional branching into the deterministic validator and the repair
  loop, which is a graph shape, not a tool-calling shape. The `plan` node
  itself does not use a tool-calling agent for the reasons in DD-01.
- **"The flight-physics model isn't aerospace-grade."** — correct, and the
  module docstring says so. It is principled, deterministic, and documented;
  the contract is the same regardless of the underlying numbers (which is
  why the validator is the safety surface, not the physics).
- **"You don't fly the drone."** — by design (`CLAUDE.md` §1, hard
  constraints). MTS is the tasking + validation layer; the `MissionPlan` is
  the deliverable. A real autonomy stack would consume it via the export
  formats in `app/export/`.
