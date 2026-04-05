"""Shared utilities for all agent nodes."""

import json
import os
import time
import logging
from langchain_openai import AzureChatOpenAI
from data_loader import COLUMN_MAP

logger = logging.getLogger(__name__)

SUMMARIZE_AFTER = 6
GROUPABLE_COLUMNS = sorted(COLUMN_MAP.values())
LLM_MAX_RETRIES = 2
LLM_RETRY_DELAY = 1.0

_llm: AzureChatOpenAI | None = None


def get_llm() -> AzureChatOpenAI:
    """Lazy-init the Azure OpenAI client."""
    global _llm
    if _llm is None:
        _llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
            temperature=0,
            timeout=30,
        )
    return _llm


def llm_invoke(messages: list, agent_name: str = "unknown") -> str:
    """Call the LLM with retry logic. Returns raw content string."""
    last_error = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            resp = get_llm().invoke(messages)
            return resp.content
        except Exception as e:
            last_error = e
            logger.warning(f"{agent_name} LLM call failed (attempt {attempt + 1}): {e}")
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"{agent_name} LLM call failed after {LLM_MAX_RETRIES + 1} attempts: {last_error}")


def format_history(state: dict) -> str:
    """Build a text block of conversation history with entity context."""
    parts = []
    if state.get("summary"):
        parts.append(f"[Summary of older conversation]: {state['summary']}")
    for turn in (state.get("history") or []):
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user" and turn.get("entities"):
            ent = turn["entities"]
            scope = []
            for k, v in ent.items():
                if isinstance(v, list) and v:
                    scope.append(f"{k}={v}")
                elif isinstance(v, str) and v:
                    scope.append(f"{k}={v}")
            scope_str = ", ".join(scope) if scope else "no filters (all data)"
            parts.append(f"[user]: {content}\n  [scope: {scope_str}]")
        else:
            parts.append(f"[{role}]: {content}")
    return "\n".join(parts) if parts else "No prior conversation."


def parse_json(text: str) -> dict:
    """Parse JSON from LLM output. Handles markdown fences and malformed output."""
    cleaned = text.strip()
    for prefix in ["```json", "```"]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM JSON: {cleaned[:200]}")
        return {}
