"""
LangGraph state machine definition.

6 nodes, 2 conditional edges, 1 feedback loop:
  Router → Retriever → Analyst → Validator ←→ Analyst (retry) → Responder → Memory
"""

from typing import Any
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from agents import (
    router_node, retriever_node, analyst_node,
    validator_node, responder_node, memory_node,
)
from data_loader import COLUMN_MAP


class AgentState(TypedDict, total=False):
    """Shared state passed between all nodes in the graph."""

    query: str
    intent: str
    detail_level: str
    entities: dict[str, Any]
    clarification: str | None

    data: list[dict] | None
    data_summary: dict | None
    scope: str

    result: dict | None
    error: str | None
    response: str

    history: list[dict]
    summary: str
    steps: list[dict]

    validation_passed: bool
    analyst_feedback: str | None
    retry_count: int
    last_operations: list[str]


def route_after_router(state: AgentState) -> str:
    """General/unclear without entities skips the data pipeline entirely."""
    ent = state.get("entities", {})
    has_entities = any(ent.get(k) for k in COLUMN_MAP)
    if state.get("intent") == "general" and not has_entities:
        return "responder"
    if state.get("intent") == "unclear" and not has_entities:
        return "responder"
    return "retriever"


def route_after_validator(state: AgentState) -> str:
    """If validation failed, send back to Analyst with feedback. Otherwise continue."""
    if state.get("validation_passed", True):
        return "responder"
    return "analyst"


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("retriever", retriever_node)
    g.add_node("analyst", analyst_node)
    g.add_node("validator", validator_node)
    g.add_node("responder", responder_node)
    g.add_node("memory", memory_node)

    g.set_entry_point("router")
    g.add_conditional_edges("router", route_after_router, {
        "retriever": "retriever",
        "responder": "responder",
    })
    g.add_edge("retriever", "analyst")
    g.add_edge("analyst", "validator")
    g.add_conditional_edges("validator", route_after_validator, {
        "responder": "responder",
        "analyst": "analyst",
    })
    g.add_edge("responder", "memory")
    g.add_edge("memory", END)

    return g.compile()


app = build_graph()


def run_query(query: str, memory_state: dict | None = None) -> dict:
    """Invoke the graph with a query and optional persisted memory from previous turns."""
    state = {"query": query}
    if memory_state:
        state["history"] = memory_state.get("history", [])
        state["summary"] = memory_state.get("summary", "")
        state["last_operations"] = memory_state.get("last_operations", [])
    return app.invoke(state)
