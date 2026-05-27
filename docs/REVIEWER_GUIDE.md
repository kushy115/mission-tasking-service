# Reviewer Guide

A short, opinionated tour of the codebase for someone with 10–30 minutes who
wants to read the engineering, not the surface area.

If you only have time for one file, read
[`app/validation/kernel.py`](../app/validation/kernel.py). That's the safety
core, and it's deliberately the most boring file in the repo.

---

## How it's built, in LangChain terms

A LangChain-fluent reviewer will recognize most of the choices below. The
quick map up front:

- **It's a custom LangGraph `StateGraph`, not a prebuilt agent.** No
  `create_agent`, no `create_react_agent`. The graph is explicit because we
  need conditional branching into a deterministic validator and a bounded
  repair cycle — shapes that don't fall out of a tool-calling agent loop.
  See DD-001.
- **The planner is a single `llm.invoke()` against a provider-agnostic chat
  model** (`init_chat_model`, so the same code points at Anthropic, OpenAI,
  or Google by env). It is **not** an LCEL `prompt | llm | parser` chain.
  Messages are composed by hand (`SystemMessage`, `HumanMessage`, optional
  `AIMessage` for the multi-turn clarification thread) and the response is
  parsed manually. The trade-off is in DD-001.
- **The advisor node *does* use `with_structured_output(OptimizationAdvisory)`.**
  That's the idiomatic LangChain pattern for forcing the LLM into a Pydantic
  schema. The planner originally used the same approach with `MissionPlan`
  but ran into a provider-specific failure mode on Gemini — DD-001 has the
  details.
- **The graph's branching uses `add_conditional_edges`.** Think of these as
  `RunnableBranch` lifted into the graph layer: a routing function returns
  the name of the next node, with full state threaded through. The compile
  graph has two such routers (`_after_intake`, `_after_validate`) plus a
  cycle (`repair → plan`).
- **The `messages` channel uses LangGraph's `add_messages` reducer.** New
  messages append rather than replace, which is what makes the multi-turn
  clarification thread work across compile attempts.
- **Human-in-the-loop approval is a real `interrupt`.** The graph is compiled
  with `interrupt_after=["finalize"]`; `/v1/missions:approve` resumes it via
  `graph.invoke(None, config={"thread_id": ...})`. The checkpointer is
  `langgraph.checkpoint.postgres.PostgresSaver`, so the pause survives a
  service restart. Pods can scale horizontally because the checkpoint state
  is in Postgres, not in-process.

If you want to point at "the chain," it's the `StateGraph` itself. Sequential
edges are sequential composition; conditional edges are routing; the
`repair → plan` edge is the cycle that lets the kernel correct the LLM.

---

## 10-minute path — "is the agent engineering sound?"

Read these four files in order:

1. **[`app/schemas/plan.py`](../app/schemas/plan.py)** — the `MissionPlan`
   Pydantic v2 contract. This is what the LLM must produce and what the
   validator checks. Understand the schema and the rest of the system falls
   out of it.

2. **[`app/graph/builder.py`](../app/graph/builder.py)** — the `StateGraph`
   wiring. Nine nodes (six on the happy path: `intake → plan → validate →
   critique → advisor → finalize`, plus `repair`, `clarify`, `reject`).
   Branching via `add_conditional_edges`. One `interrupt` after `finalize`.
   The repair loop is bounded at 3 attempts — see DD-001 for why that number.

3. **[`app/graph/nodes.py`](../app/graph/nodes.py)** — the node bodies. The
   big comment block above `plan_node` documents why the planner is a direct
   `llm.invoke()` instead of `create_agent` (cost, latency, and a real
   provider failure mode on Gemini's `response_format=Pydantic`). This is
   the design call most likely to draw a "why?".

4. **[`app/validation/kernel.py`](../app/validation/kernel.py)** — the
   deterministic safety kernel. Seven checks, no LLM, no network, fully
   unit-tested in [`tests/test_validation.py`](../tests/test_validation.py).
   The LLM proposes; the kernel disposes. The repair loop feeds the kernel's
   error list back into the next `plan` attempt's prompt verbatim.

That's the agent engineering core. Everything else in the repo is plumbing,
UI, ops, and one second graph (the inflight supervisor).

---

## 30-minute path — "is the productionization real?"

Add these:

5. **[`docs/DESIGN_DECISIONS.md`](./DESIGN_DECISIONS.md)** — the running log
   of non-trivial choices, each tagged with the files it owns. Read DD-001
   (the agent-vs-direct-call rewrite) and DD-014 (the inflight supervisor) at
   minimum.

6. **[`deploy/helm/mts/`](../deploy/helm/mts/)** — the Helm chart. Each
   template has a one-line comment explaining why that resource is there.
   Of particular interest: the `HorizontalPodAutoscaler` scales on inflight
   requests, not CPU, because LLM calls are I/O-bound and CPU underreports
   load while requests sit in the planner.

7. **[`app/observability/`](../app/observability/)** plus the
   auto-provisioned Grafana dashboard at
   [`deploy/docker/grafana/`](../deploy/docker/grafana/). Three layers:
   LangSmith for LLM-level traces (every node, every tool call, token
   counts), OpenTelemetry for service spans (HTTP → graph → SQL), Prometheus
   for ops metrics. `docs/observability.md` is the metric-interpretation
   guide.

8. **[`app/supervisor/`](../app/supervisor/)** — the second LangGraph
   subgraph, runs *during* the live sim. Owns the closed-loop replanning
   story: operator approves a patrol → wind spike event arrives → supervisor
   decides `REPLAN_FROM_HERE` → new plan re-runs through the same kernel →
   gets swapped in-place on the WebSocket stream.

---

## What to look for

- **Schema-first LLM I/O.** The LLM is constrained to `MissionPlan`. We
  `json.loads()` the response and Pydantic-validate it. Parse or validation
  failure kicks the repair loop with the kernel's errors injected into the
  next system prompt.
- **Deterministic safety core.** Pure Python, fully testable, never an LLM
  call. The kernel is the single source of truth on whether a plan is safe.
- **Bounded self-correction.** Repair loop cap of 3 is a documented
  cost/safety ceiling, not a magic number (DD-001).
- **Two graphs, one kernel.** The compile graph plans pre-flight; the
  supervisor subgraph replans in-flight. Every decision in both graphs ends
  through the same `kernel.validate(...)`.
- **Stateless service, stateful planner.** Postgres holds the LangGraph
  checkpoints, so `interrupt` survives restarts and pods scale horizontally
  without coordination.
- **Observability isn't an afterthought.** Every node emits a span, every
  tool call is counted, every LLM call's tokens are tracked. A rising
  repair-loop rate is the regression signal we'd alert on first.

---

## Likely objections (and our answers)

The choices below tend to raise eyebrows on a first read. We've thought about
each one, so the reasoning is laid out here instead of making you dig.

- **"Why a custom `StateGraph` and not `create_agent`?"** Because we need
  conditional branching into a deterministic validator and a bounded
  self-correction cycle. That's a graph shape, not a tool-calling shape.
  A `create_agent` loop would run until the LLM decided to stop; we want
  the *validator* to decide when to stop. DD-001 has the trade-off in full.

- **"Why not LCEL chains in the planner?"** No interesting transformation
  to compose. The LCEL `|` syntax adds friction when you want to
  conditionally include the previous attempt's violations in the next
  prompt. A single `llm.invoke()` with a hand-composed message list is
  shorter, easier to debug, and shows up cleanly in LangSmith as one span
  per attempt.

- **"The flight-physics model isn't aerospace-grade."** Correct, and the
  module docstring says so. It is principled, deterministic, and documented.
  The validator — not the physics — is the safety surface. Swapping in a
  more faithful model is a contained change.

- **"You don't fly the drone."** By design. The project's hard constraints
  fix MTS as the tasking + validation layer, and a `MissionPlan` is the
  deliverable. A real autonomy stack would consume it via the export formats
  in `app/export/`.
