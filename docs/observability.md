# Observability — How to read MTS metrics

This guide explains every Prometheus metric MTS emits, what a *healthy* reading
looks like, and what to investigate when something moves. Open the Grafana
dashboard at <http://localhost:3000> (auto-provisioned from
`deploy/docker/grafana/dashboards/mts.json`); each panel has the same
description inline.

The metrics are grouped into four layers — read them in this order when
debugging an issue:

1. **Compile layer** — is the user-facing workflow working?
2. **LLM layer** — is the language model healthy and affordable?
3. **Safety-kernel layer** — what kinds of mistakes is the planner making?
4. **Plan-shape layer** — what kinds of plans is the planner producing?

---

## 1. Compile layer (user-facing)

### `mts_compile_requests_total{status}`
**Counter.** Total compile calls. `status` ∈ `{READY_FOR_APPROVAL,
REJECTED, NEEDS_CLARIFICATION, error}`.

- **Healthy:** the share of `READY_FOR_APPROVAL` should be >70% on a clean
  eval dataset. `error` should be 0.
- **What it means when it moves:**
  - REJECTED rising → planner is producing unsafe plans; check the
    *safety-kernel violations* panel for which checks fire most.
  - NEEDS_CLARIFICATION rising → operators are sending vaguer commands
    OR the planner has become too cautious (try a stronger model).
  - `error` rising → the service itself is failing (Postgres down, LLM
    misconfigured, code bug); check `docker compose logs mts`.

### `mts_compile_duration_seconds`
**Histogram.** Wall-clock time from HTTP request received to HTTP response
sent. Includes the entire graph + every LLM call + validator.

- **Healthy:** p95 < 30s with Claude Haiku / Gemini Flash. p99 < 60s.
- **What it means when it moves:**
  - p95 spikes → upstream LLM is slow (check the *LLM call latency* panel)
    OR the planner is hitting the repair loop (check *repair loops*).
  - p99 only → tail outliers from a single slow LLM call; usually transient.

### `mts_repair_loops`
**Histogram.** Number of repair iterations per compile. Hard-capped at 3 in
`app/config.py` (`MTS_REPAIR_LOOP_CAP`).

- **Healthy:** average < 1.0 (most plans valid on first try); p95 ≤ 1.
- **What it means when it moves:**
  - Avg between 1 and 2 → planner makes a violation half the time; check
    *violations* to see which check is dominant.
  - Avg approaching 3 → the planner is consistently producing unsafe plans
    that it can't fix; you're about to see a wave of REJECTED responses.

### `mts_rejections_total` / `mts_clarifications_total`
**Counters.** Convenience counters for total REJECTED / NEEDS_CLARIFICATION
plans. Duplicated information vs `mts_compile_requests_total` but easier to
alert on.

---

## 2. LLM layer (cost + provider health)

### `mts_llm_calls_total{provider, model, outcome}`
**Counter.** Every call to the LLM API. `outcome` ∈ `{ok, error, rate_limited}`.

- **Healthy:** ratio of LLM calls to compile requests should be ~1 (we make
  one LLM call per compile, plus one per repair pass). `rate_limited` = 0.
- **What it means when it moves:**
  - Big multiplier (e.g. 5× compile rate) → repair loop is hammering the
    LLM; planner is failing repeatedly.
  - `rate_limited` > 0 → you've hit the provider's quota. Free-tier Gemini
    has a 20 RPD cap on flash models. Switch model or pay for the tier.
  - `error` > 0 → upstream LLM is failing; usually transient.

### `mts_llm_duration_seconds{provider, model}`
**Histogram.** Wall-clock latency of a single LLM API call.

- **Healthy:** p50 < 2s, p95 < 5s for Haiku/Flash; p50 ~5s for Sonnet/Pro.
- **What it means when it moves:**
  - p95 climbs gradually → provider is under load.
  - p95 jumps sharply → an upstream incident; check provider status page.
  - p50 climbing → your prompt has grown (added too much context).

---

## 3. Safety-kernel layer (what the planner gets wrong)

### `mts_validation_violations_total{check}`
**Counter.** Each time the deterministic safety kernel rejects a plan for a
specific reason. `check` is one of the six kernel checks:

- `inside_geofence` — plan exits the operating area
- `avoids_nfz` — plan crosses a no-fly-zone
- `battery_within_budget` — battery reserve at landing < 20%
- `within_endurance` — total flight time > rated endurance
- `sensor_coverage_adequate` — search pattern has coverage gaps
- `ends_with_rtb` — plan doesn't end with RETURN_TO_BASE

- **Healthy:** all near zero; small counts for the harder checks
  (`sensor_coverage_adequate`, `battery_within_budget`) are normal.
- **What it means when one dominates:**
  - `sensor_coverage_adequate` → planner is mislabeling perimeter patrols as
    SEARCH_PATTERN. Tweak the prompt's leg-type rules.
  - `avoids_nfz` → planner is ignoring or forgetting about NFZs. Make the
    geo context more prominent in the prompt.
  - `battery_within_budget` → plans are too ambitious for the drone profile.
    Either prompt the planner to be more conservative or upgrade the drone.
  - `inside_geofence` → planner is inventing coordinates rather than using
    the boundary you gave it.
  - `ends_with_rtb` → planner is forgetting the final RETURN_TO_BASE leg.
    Make rule #6 of the system prompt more emphatic.

---

## 4. Plan-shape layer (what kinds of plans?)

These histograms only capture **approved** plans (`READY_FOR_APPROVAL`).
NEEDS_CLARIFICATION and REJECTED plans don't have meaningful shapes.

### `mts_plan_legs`
**Histogram.** Number of legs per approved plan.

- **Healthy:** p50 = 2–3 (most missions are work + RTB); p95 ≤ 5.
- **What it means when it moves:**
  - p95 spikes upward → planner is over-decomposing simple commands.
  - p50 = 1 → planner is producing RTB-only plans; something is broken in
    the prompt or upstream parsing.

### `mts_plan_battery_pct`
**Histogram.** Estimated battery consumption per approved plan.

- **Healthy:** p50 < 30% on the long-endurance drone for the sample
  commands; p95 < 60%.
- **What it means when it moves:**
  - p95 above 80% → planner is regularly cutting it close to the 20%
    reserve floor; expect more `battery_within_budget` violations soon.

### `mts_plan_duration_seconds`
**Histogram.** Estimated flight duration per approved plan.

- **Healthy:** matches the workload — a perimeter patrol should be 2–5 min,
  a full-area lawnmower sweep 10–20 min.
- **What it means when it moves:**
  - Sudden jump → planner has changed its strategy (e.g. lawnmower instead
    of patrol). Cross-check with `mts_plan_legs`.

---

## HTTP layer (sanity check)

`mts_http_requests_total{method,path,status}` and `mts_http_duration_seconds{method,path}`
exist for completeness — they're middleware-level metrics from FastAPI. Useful
to confirm requests are actually reaching the service, separately from how the
graph behaves.

---

## What to ignore

- `python_*`, `process_*` metrics — Python runtime defaults. Not useful unless
  you're chasing a memory leak.

---

## Adding a new metric

1. Define it in `app/observability/metrics.py` (Counter / Histogram).
2. Import + increment it from the relevant code path.
3. Add a panel to `deploy/docker/grafana/dashboards/mts.json` with a
   `description` field explaining what it means.
4. Document it here under the appropriate layer.
5. The Grafana container picks up dashboard JSON changes within ~10s — no
   restart needed.
