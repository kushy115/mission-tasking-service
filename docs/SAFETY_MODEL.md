# Safety model

The kernel (`app/validation/kernel.py`) is the safety core. It runs after every
plan attempt, takes a `MissionPlan` plus the geo context and drone profile, and
returns a structured list of violations plus a `ConstraintReport`. No LLM, no
network — pure Python, fully unit-tested.

A plan is only ever marked `READY_FOR_APPROVAL` if every check below passes.

## The seven constraints

1. **Geofence containment.** Every waypoint of every leg sits inside the
   operating-area boundary polygon. Computed with Shapely `Polygon.covers`
   against the boundary loaded from PostGIS.

2. **No-fly-zone avoidance.** No leg segment intersects any NFZ polygon for the
   area. We test the full polyline, not just the waypoints — clipping an NFZ
   corner between two waypoints would otherwise pass an endpoint-only check.

3. **Altitude window.** Every waypoint altitude is below the area's ceiling and
   above the configured min-AGL floor (default 20 m, `MTS_MIN_AGL_M`). A search
   pattern at 200 m in a 120 m ceiling area gets rejected here.

4. **Battery reserve.** `total_battery_pct` consumption leaves at least
   `battery_reserve_pct >= 20%` at landing. The physics model estimates
   consumption per leg using cruise / climb / hover / sensor draw and the
   weather wind penalty; the 20% floor is configurable via
   `MTS_BATTERY_RESERVE_MIN_PCT`.

5. **Endurance budget.** `total_duration_s` is within the drone profile's rated
   endurance. A plan that fits the battery but blows the timer (e.g. a slow
   loiter pattern with low power draw) is still rejected.

6. **Sensor coverage.** For search-pattern legs, track spacing is within the
   sensor's swath at the planned altitude. The agent is free to pick any
   spacing — the kernel verifies the resulting plan actually covers ground
   without gaps.

7. **Ends with RTB.** The plan ends with a `RETURN_TO_BASE` leg. A mission that
   stops mid-air without committing to a return is unsafe regardless of how
   the legs look in isolation.

There's also an 8th check (DD-008, **airspace deconfliction**) that runs only
when other recently-approved missions exist in the same area, and rejects
plans whose timing and geometry conflict with already-approved flights.

## Adding a constraint

1. Add the check to `app/validation/kernel.py`. The function takes the plan +
   geo context + drone profile, and appends a string to `violations` if the
   constraint fails. Add a corresponding boolean + detail field to
   `ConstraintReport`.
2. Add a test in `tests/test_validation.py`. The safety core is the one place
   tests are mandatory — the safety guarantees only hold if every constraint
   has a regression test.
3. If the constraint depends on something the planner needs to know about
   (e.g. a new keep-out class), surface it through the geo store and inline it
   into the planner prompt.

The kernel is deliberately conservative. When in doubt it rejects. The repair
loop (cap = 3) gives the planner three attempts to fix violations before the
service gives up and returns `REJECTED` with the list.
