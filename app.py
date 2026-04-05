"""Streamlit chat UI with reasoning chain display."""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RE Asset Manager", page_icon="🏢")
st.title("🏢 Real Estate Asset Manager")
st.caption("Multi-agent system powered by LangGraph + Azure OpenAI (GPT-4.1 mini)")

from graph import run_query

EMPTY_MEMORY = {"history": [], "summary": "", "last_operations": []}
AGENT_LABELS = {
    "Router": "Agent 1 - Router (LLM)",
    "Retriever": "Retriever (Python)",
    "Analyst": "Agent 2 - Analyst (LLM)",
    "Analyst (retry)": "Agent 2 - Analyst (LLM, retry)",
    "Validator": "Validator (LLM)",
    "Responder": "Agent 3 - Responder (LLM)",
}


def render_steps(steps):
    """Show the reasoning chain as labeled steps inside an expander."""
    if not steps:
        return
    with st.expander("Reasoning chain"):
        for step in steps:
            agent = step.get("agent", "Unknown")
            label = AGENT_LABELS.get(agent, agent)
            reasoning = step.get("reasoning", "")
            ops = step.get("operations")
            st.markdown(f"**{label}**")
            st.markdown(f"{reasoning}")
            if ops:
                st.caption(f"Operations: {', '.join(ops)}")
            st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory_state" not in st.session_state:
    st.session_state.memory_state = dict(EMPTY_MEMORY)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            render_steps(msg["steps"])

query = st.chat_input("Ask about your properties...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Agents working..."):
            try:
                result = run_query(query, st.session_state.memory_state)
                response = result.get("response", "No response generated.")
                steps = result.get("steps", [])
                st.session_state.memory_state = {
                    "history": result.get("history", []),
                    "summary": result.get("summary", ""),
                    "last_operations": result.get("last_operations", []),
                }
            except Exception as e:
                response = f"Error: {str(e)}"
                steps = []

        st.markdown(response)
        render_steps(steps)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "steps": steps,
    })
    st.rerun()
