"""Agent 1: Router - classifies intent, extracts entities, resolves references."""

import json
import logging
from datetime import date
from langchain_core.messages import SystemMessage, HumanMessage
from data_loader import get_metadata, COLUMN_MAP
from agents.helpers import llm_invoke, format_history, parse_json

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """You route queries for a real estate asset management system.
Today: {today}

HISTORY:
{history}

AVAILABLE DATA:
{metadata}

Output the COMPLETE entity set for this query. 
Read the conversation naturally. The user's query refers to the most recent context unless they explicitly change topic.
Time: "this year" = {current_year}, "latest" = most recent in data. No time mentioned = leave empty.
ALWAYS use the actual calendar year even if it's not in the available data.
Do NOT substitute with the latest available year, The Retriever will handle the error.
Detail: "total"/"sum"/"overall" = summary. 
"breakdown"/"by category"/"per tenant"/"split by"/"quarterly"/"monthly" = detailed.

Intent: "data" for anything about buildings, tenants, financials, ledgers. 
"general" only when nothing in the dataset is mentioned.
"comparison" when explicitly comparing two or more entities. 
"unclear" when you can't tell.

Respond ONLY with valid JSON:
{{
  "reasoning": "one sentence",
  "intent": "data" or "general" or "comparison" or "unclear",
  "detail_level": "summary" or "detailed",
  "entities": {{
    "buildings": [], "tenants": [], "years": [], "quarters": [], "months": [],
    "ledger_type": "revenue" or "expenses" or null,
    "ledger_groups": [],
    "ledger_categories": [],
    "ledger_codes": [],
    "ledger_descriptions": []
  }}
}}
"""


def router_node(state: dict) -> dict:
    """Classify intent, extract entities from query and conversation history."""
    try:
        today = date.today()
        meta = get_metadata()
        content = llm_invoke([
            SystemMessage(content=ROUTER_PROMPT.format(
                metadata=json.dumps(meta, indent=2),
                history=format_history(state),
                today=today.isoformat(),
                current_year=str(today.year),
            )),
            HumanMessage(content=state["query"]),
        ], agent_name="Router")
        parsed = parse_json(content)
    except Exception as e:
        logger.error(f"Router failed: {e}")
        parsed = {}

    result = {
        "intent": parsed.get("intent", "unclear"),
        "detail_level": parsed.get("detail_level", "summary"),
        "entities": parsed.get("entities", {}),
        "clarification": parsed.get("clarification"),
        "steps": [{"agent": "Router", "reasoning": parsed.get("reasoning", "Router encountered an error.")}],
    }

    ent = result["entities"]
    if result["intent"] == "comparison":
        has_multi = any(len(ent.get(k, [])) > 1 for k in COLUMN_MAP if isinstance(ent.get(k), list))
        if not has_multi:
            result["intent"] = "unclear"
            result["clarification"] = "Which entities would you like to compare? Please name at least two."

    return result
