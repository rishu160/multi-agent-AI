import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_SYSTEM = """You are a routing supervisor. Classify the user query into exactly one category:

- research  -> needs current facts, news, or web lookup
- code      -> needs calculation, algorithm, data analysis, or runnable code
- writer    -> needs creative writing, summarisation, explanation, or structured prose

Respond with ONLY one word: research, code, or writer.
If there is critic feedback, take it into account when re-routing."""

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            max_tokens=10,
        )
    return _llm


def supervisor_node(state: AgentState) -> dict:
    feedback_note = (
        f"\n\nCritic feedback from previous attempt: {state.get('critic_feedback', '')}"
        if state.get("critic_feedback")
        else ""
    )
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=state["query"] + feedback_note),
    ]
    response = _get_llm().invoke(messages)
    route = response.content.strip().lower()
    if route not in {"research", "code", "writer"}:
        route = "writer"
    return {"route": route, "critic_feedback": ""}
