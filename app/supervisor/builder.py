"""Compile the supervisor subgraph.

The supervisor is a one-shot LangGraph: each invocation runs assess → decide →
(replan | commit) → END and returns the final state. No checkpointer — the
graph is stateless across ticks. Persistence (if needed) belongs to the
orchestrator that owns the live mission session, not the supervisor itself.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.supervisor.nodes import assess_node, commit_node, decide_node, replan_node
from app.supervisor.state import SupervisorDecision, SupervisorState


def _after_decide(state: SupervisorState) -> str:
    """REPLAN_FROM_HERE routes to the replan node; everything else commits."""
    if state.get("decision") == SupervisorDecision.REPLAN_FROM_HERE.value:
        return "replan"
    return "commit"


_compiled = None


def build_supervisor() -> Any:
    """Build (and cache) the compiled supervisor graph.

    Cached per-process because compilation is cheap but not free, and the
    graph is invoked many times per second per active mission.
    """
    global _compiled
    if _compiled is not None:
        return _compiled

    g: StateGraph = StateGraph(SupervisorState)
    g.add_node("assess", assess_node)
    g.add_node("decide", decide_node)
    g.add_node("replan", replan_node)
    g.add_node("commit", commit_node)

    g.add_edge(START, "assess")
    g.add_edge("assess", "decide")
    g.add_conditional_edges("decide", _after_decide, {"replan": "replan", "commit": "commit"})
    g.add_edge("replan", "commit")
    g.add_edge("commit", END)

    _compiled = g.compile()
    return _compiled
