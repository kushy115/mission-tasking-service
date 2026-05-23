"""Assemble and compile the compile-flow StateGraph.

Edges:
  intake → plan | clarify | reject
  plan → validate
  validate → finalize | repair | clarify | reject
  repair → plan
  clarify → finalize → END
  reject  → finalize → END

The graph is compiled with a Postgres checkpointer so the human-in-the-loop
interrupt at the approval gate survives process restarts. Unit tests use the
in-memory MemorySaver instead.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph.nodes import (
    clarify_node,
    finalize_node,
    intake_node,
    plan_node,
    reject_node,
    repair_node,
    validate_node,
)
from app.graph.state import CompileState
from app.schemas.enums import MissionStatus

log = logging.getLogger(__name__)


def _after_intake(state: CompileState) -> str:
    status = state.get("status")
    if status == MissionStatus.NEEDS_CLARIFICATION.value:
        return "clarify"
    if status == MissionStatus.REJECTED.value:
        return "reject"
    return "plan"


def _after_validate(state: CompileState) -> str:
    """Branch based on whether the plan needs repair, clarification, or is final.

    A plan that the agent itself marked NEEDS_CLARIFICATION or REJECTED is
    surfaced directly. Otherwise, if there are violations and we have repair
    budget, loop. If repair budget is exhausted, reject.
    """
    settings = get_settings()
    raw = state.get("draft_plan") or {}
    plan_status = raw.get("status")
    if plan_status == MissionStatus.NEEDS_CLARIFICATION.value:
        return "clarify"
    if plan_status == MissionStatus.REJECTED.value:
        return "reject"

    errors = state.get("validation_errors") or []
    if not errors:
        return "finalize"

    if state.get("repair_attempts", 0) >= settings.repair_loop_cap:
        return "reject"
    return "repair"


def build_graph(checkpointer: Any | None = None) -> Any:
    """Build and compile the compile-flow graph.

    Passing `checkpointer=None` here yields an uncompiled graph using
    MemorySaver — useful for unit tests. In-process production should wire in
    `langgraph.checkpoint.postgres.PostgresSaver` so the interrupt survives.
    """
    g: StateGraph = StateGraph(CompileState)

    g.add_node("intake", intake_node)
    g.add_node("plan", plan_node)
    g.add_node("validate", validate_node)
    g.add_node("repair", repair_node)
    g.add_node("clarify", clarify_node)
    g.add_node("reject", reject_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "intake")
    g.add_conditional_edges(
        "intake", _after_intake, {"plan": "plan", "clarify": "clarify", "reject": "reject"}
    )
    g.add_edge("plan", "validate")
    g.add_conditional_edges(
        "validate",
        _after_validate,
        {
            "finalize": "finalize",
            "repair": "repair",
            "clarify": "clarify",
            "reject": "reject",
        },
    )
    g.add_edge("repair", "plan")
    g.add_edge("clarify", "finalize")
    g.add_edge("reject", "finalize")
    g.add_edge("finalize", END)

    cp = checkpointer or MemorySaver()
    # The human-in-the-loop pause is the boundary AFTER finalize: callers resume
    # by either calling the approve endpoint or rejecting. We wire the interrupt
    # via interrupt_after so a fresh invocation runs through finalize, then the
    # caller can read the plan and either continue (approve) or discard.
    return g.compile(checkpointer=cp, interrupt_after=["finalize"])


def mermaid_diagram() -> str:
    """Emit the compiled graph as a Mermaid diagram for docs/graph.mmd."""
    g = build_graph()
    return g.get_graph().draw_mermaid()
