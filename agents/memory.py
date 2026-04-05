"""Memory node - stores conversation history and compresses old turns."""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from agents.helpers import llm_invoke, SUMMARIZE_AFTER

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """Summarize this conversation into a concise paragraph.
Preserve: building names, tenant names, timeframes, financial figures, user interests. Drop filler.

{conversation}

Respond with ONLY the summary paragraph."""


def memory_node(state: dict) -> dict:
    """Append turn to history and compress old turns into a summary."""
    history = list(state.get("history") or [])
    summary = state.get("summary", "")

    history.append({
        "role": "user",
        "content": state.get("query", ""),
        "entities": state.get("entities"),
    })
    history.append({
        "role": "assistant",
        "content": state.get("response", ""),
    })

    if len(history) >= SUMMARIZE_AFTER * 2:
        old = history[:-4]
        recent = history[-4:]
        convo = "\n".join(f"[{t.get('role','?')}]: {t.get('content','')}" for t in old)
        if summary:
            convo = f"[Previous summary]: {summary}\n{convo}"
        try:
            summary = llm_invoke([
                SystemMessage(content=SUMMARIZE_PROMPT.format(conversation=convo)),
                HumanMessage(content="Summarize now."),
            ], agent_name="Memory").strip()
            history = recent
        except Exception as e:
            logger.warning(f"Summarization failed, keeping full history: {e}")

    return {
        "history": history,
        "summary": summary,
        "steps": state.get("steps", []),
    }
