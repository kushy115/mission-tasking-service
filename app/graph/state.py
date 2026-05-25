"""Compile graph state. TypedDict so LangGraph's StateGraph can introspect it."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class CompileState(TypedDict, total=False):
    """State threaded through the compile graph.

    Fields are optional (total=False) because they are populated progressively
    by nodes. The graph contract is: `intake` populates the input fields,
    `plan` populates `draft_plan`, `validate` populates `validation_errors` and
    `constraints_report`, `finalize` populates the final status.
    """

    raw_command: str
    area_id: str
    operator_clearance: str
    drone_state: dict[str, Any]
    request_id: str | None
    geo_context: dict[str, Any]  # boundary, nfzs, ceiling, home — serialized
    weather: dict[str, Any] | None  # WeatherObservation dict; see weather/provider.py
    draft_plan: dict[str, Any] | None
    validation_errors: list[str]
    constraints_report: dict[str, Any] | None
    repair_attempts: int
    status: str
    clarification_questions: list[str]
    rejection_reasons: list[str]
    confidence_score: float | None  # set by critique node; see DESIGN_DECISIONS §6
    critique_notes: str
    alternatives: list[dict[str, Any]]  # extra plans; see DESIGN_DECISIONS §7
    primary_idx: int
    messages: Annotated[list, add_messages]
