"""Validator - checks if Analyst output matches the query, can trigger one retry."""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from data_loader import COLUMN_MAP
from agents.helpers import llm_invoke, parse_json

logger = logging.getLogger(__name__)

VALIDATOR_PROMPT = """You are a quality checker for a real estate data analysis system.
Check if the Analyst's computed results adequately answer the user's query.

USER QUERY: {query}
INTENT: {intent} | DETAIL LEVEL: {detail_level}
DATA SUMMARY: {data_summary}
ANALYST OPERATIONS: {operations}
ANALYST RESULTS (keys only): {result_keys}

Respond ONLY with valid JSON:
{{
  "pass": true or false,
  "reasoning": "one sentence explaining your decision",
  "feedback": "if pass=false, specific instruction for the Analyst on what operation to add"
}}

RULES:
- pass=true if results contain enough data to answer the query.
- pass=false ONLY if a clearly needed operation is missing. Examples:
  - User asks "split by tenant" but results have no "by_tenant_name" key - feedback: "Add group_by:tenant_name"
  - User asks "compare buildings" but results have no "by_property_name" key - feedback: "Add group_by:property_name"
  - User asks "category breakdown" but results have no "by_ledger_category" key - feedback: "Add group_by:ledger_category"
- Do NOT fail for minor issues. If in doubt, pass=true.
- Keep feedback actionable: name the specific group_by:<column> operation to add."""


def validator_node(state: dict) -> dict:
    """Check Analyst output against user intent. Can send feedback for one retry."""
    steps = list(state.get("steps", []))
    retry = state.get("retry_count", 0)

    if not state.get("result") or retry >= 1:
        if retry >= 1:
            steps.append({"agent": "Validator", "reasoning": "Max retries reached, passing through."})
        return {"validation_passed": True, "steps": steps}

    ent = state.get("entities", {})
    no_specific_entities = not any(ent.get(k) for k in COLUMN_MAP)
    if no_specific_entities and state.get("intent") in ("asset_details", "comparison", "unclear"):
        steps.append({"agent": "Validator", "reasoning": "No valid entities extracted - retry would not help. Passing through."})
        return {"validation_passed": True, "steps": steps}

    result_keys = list(state["result"].keys()) if state.get("result") else []

    try:
        content = llm_invoke([
            SystemMessage(content=VALIDATOR_PROMPT.format(
                query=state["query"],
                intent=state.get("intent", "unknown"),
                detail_level=state.get("detail_level", "summary"),
                data_summary=json.dumps(state.get("data_summary", {}), indent=2),
                operations=state.get("steps", [{}])[-1].get("operations", []),
                result_keys=result_keys,
            )),
            HumanMessage(content="Validate now."),
        ], agent_name="Validator")
        parsed = parse_json(content)
    except Exception as e:
        logger.error(f"Validator failed, passing through: {e}")
        steps.append({"agent": "Validator", "reasoning": f"Validation error ({e}), passing through."})
        return {"validation_passed": True, "steps": steps}

    passed = parsed.get("pass", True)
    reasoning = parsed.get("reasoning", "")
    steps.append({"agent": "Validator", "reasoning": reasoning})

    if passed:
        return {"validation_passed": True, "steps": steps}

    return {
        "validation_passed": False,
        "analyst_feedback": parsed.get("feedback", ""),
        "retry_count": retry + 1,
        "steps": steps,
    }
