"""Agent 3: Responder - formats computed results into a natural language answer."""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from agents.helpers import llm_invoke, format_history, parse_json

logger = logging.getLogger(__name__)

RESPONDER_PROMPT = """Present real estate financial data to the user. Be concise, use EUR formatting.

CONVERSATION:
{history}

Scope: {scope}
Summary: {data_summary}
Results: {result}
Error: {error}

Only use numbers from Results. Only mention dimensions that appear in Scope.
For factual questions (counts, lists), read Summary.
If Results is empty, explain what went wrong using Error.
For "price"/"value" queries: this is P&L data, not property valuations.

JSON response:
{{
  "reasoning": "one sentence",
  "answer": "your response"
}}"""


def responder_node(state: dict) -> dict:
    """Format the computed results into a natural language answer."""
    steps = list(state.get("steps", []))

    try:
        content = llm_invoke([
            SystemMessage(content=RESPONDER_PROMPT.format(
                result=json.dumps(state.get("result"), indent=2) if state.get("result") else "None",
                error=state.get("error", "None"),
                history=format_history(state),
                scope=state.get("scope", "N/A"),
                data_summary=json.dumps(state.get("data_summary"), indent=2) if state.get("data_summary") else "None",
            )),
            HumanMessage(content=state["query"]),
        ], agent_name="Responder")
        parsed = parse_json(content)
        response = parsed.get("answer", content)
        reasoning = parsed.get("reasoning", "")
    except Exception as e:
        logger.error(f"Responder failed: {e}")
        response = "Sorry, something went wrong while formatting the response. Please try again."
        reasoning = f"Responder error: {e}"

    steps.append({"agent": "Responder", "reasoning": reasoning})
    return {"response": response, "steps": steps}
