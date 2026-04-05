"""Agent 2: Analyst - plans analysis operations, Python executes them."""

import json
import logging
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from data_loader import COLUMN_MAP, VALUE_COLUMN
from agents.helpers import llm_invoke, parse_json, GROUPABLE_COLUMNS

logger = logging.getLogger(__name__)

ANALYST_PROMPT = """You decide what to compute on a filtered real estate dataset. You don't compute it yourself.

USER QUERY: {query}
DETAIL LEVEL: {detail_level}
{feedback_section}{previous_ops_section}
DATA SUMMARY:
{data_summary}

Operations:
- "total": always include this, computes P&L, revenue, expenses
- "group_by:<column>": breakdown by a column. Available: {groupable_columns}

For summary, use only "total". For detailed, add relevant group_by.
For simple follow-ups, reuse previous operations unless the user asks for something different.

Respond with JSON:
{{
  "reasoning": "one sentence",
  "operations": ["total", ...]
}}"""


def _execute_operations(df: pd.DataFrame, operations: list[str]) -> dict:
    """Map each operation string to a deterministic pandas aggregation."""
    type_col = COLUMN_MAP.get("ledger_type", "ledger_type")
    result = {}

    for op in operations:
        try:
            if op == "total":
                result["total_pnl"] = round(df[VALUE_COLUMN].sum(), 2)
                result["total_revenue"] = round(df[df[type_col] == "revenue"][VALUE_COLUMN].sum(), 2)
                result["total_expenses"] = abs(round(df[df[type_col] == "expenses"][VALUE_COLUMN].sum(), 2))
            elif op.startswith("group_by:"):
                col = op.split(":", 1)[1]
                if col in GROUPABLE_COLUMNS and col in df.columns and df[col].notna().any():
                    grouped = {}
                    for val, gdf in df.groupby(col):
                        grouped[val] = {
                            "pnl": round(gdf[VALUE_COLUMN].sum(), 2),
                            "revenue": round(gdf[gdf[type_col] == "revenue"][VALUE_COLUMN].sum(), 2),
                            "expenses": abs(round(gdf[gdf[type_col] == "expenses"][VALUE_COLUMN].sum(), 2)),
                        }
                    result[f"by_{col}"] = grouped
        except Exception as e:
            logger.warning(f"Operation '{op}' failed: {e}")

    return result


def analyst_node(state: dict) -> dict:
    """Decide which aggregations to run, then execute them with pandas."""
    steps = list(state.get("steps", []))

    if state.get("error") or not state.get("data"):
        steps.append({"agent": "Analyst", "reasoning": "Skipped - no data available."})
        return {"result": None, "steps": steps}

    feedback = state.get("analyst_feedback")
    feedback_section = f"VALIDATOR FEEDBACK (fix this):\n{feedback}\n" if feedback else ""

    prev_ops = state.get("last_operations") or []
    previous_ops_section = f"PREVIOUS TURN OPERATIONS: {prev_ops}\n" if prev_ops else ""

    operations = ["total"]
    reasoning = ""

    try:
        content = llm_invoke([
            SystemMessage(content=ANALYST_PROMPT.format(
                query=state["query"],
                detail_level=state.get("detail_level", "summary"),
                data_summary=json.dumps(state.get("data_summary", {}), indent=2),
                feedback_section=feedback_section,
                previous_ops_section=previous_ops_section,
                groupable_columns=", ".join(GROUPABLE_COLUMNS),
            )),
            HumanMessage(content="Decide the analysis plan."),
        ], agent_name="Analyst")
        parsed = parse_json(content)
        operations = parsed.get("operations", ["total"])
        reasoning = parsed.get("reasoning", "")
    except Exception as e:
        logger.error(f"Analyst LLM failed, falling back to total: {e}")
        reasoning = f"LLM failed ({e}), defaulting to total."

    if "total" not in operations:
        operations.insert(0, "total")

    try:
        df = pd.DataFrame(state["data"])
        result = _execute_operations(df, operations)
    except Exception as e:
        logger.error(f"Analyst execution failed: {e}")
        steps.append({"agent": "Analyst", "reasoning": f"Execution error: {e}"})
        return {"result": None, "steps": steps}

    label = "Analyst (retry)" if feedback else "Analyst"
    steps.append({"agent": label, "reasoning": reasoning, "operations": operations})

    return {"result": result, "steps": steps, "last_operations": operations}
