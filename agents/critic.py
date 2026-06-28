import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_SYSTEM = """You are a quality critic. Review the agent's response against the original query.

Reply in EXACTLY this format (two lines, nothing else):
VERDICT: PASS
FEEDBACK: <one sentence - empty if passing>

Criteria for FAIL:
- The response does not answer the query
- The response contains obvious factual errors
- The response is incomplete or cut off
- Code output shows an unhandled exception

Be lenient on style; only fail on substance."""

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            max_tokens=128,
        )
    return _llm


def critic_node(state: AgentState) -> dict:
    outputs = state.get("agent_outputs", {})
    latest_output = next(reversed(outputs.values())) if outputs else ""

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=f"Query: {state['query']}\n\nAgent response:\n{latest_output}"
        ),
    ]
    response = _get_llm().invoke(messages)
    text = response.content.strip()

    verdict = "PASS"
    feedback = ""
    for line in text.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
        elif line.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    if verdict == "PASS":
        return {"final_answer": latest_output, "critic_feedback": ""}
    else:
        return {
            "critic_feedback": feedback,
            "retry_count": state.get("retry_count", 0) + 1,
        }
