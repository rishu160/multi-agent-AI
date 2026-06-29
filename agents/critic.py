import os
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_SYSTEM = """You are a quality critic. Review the agent's response against the original query.

Reply in EXACTLY this format (three lines, nothing else):
VERDICT: PASS
CONFIDENCE: <integer 0-100>
FEEDBACK: <one sentence - empty if passing>

Criteria for FAIL:
- The response does not answer the query
- The response contains obvious factual errors
- The response is incomplete or cut off
- Code output shows an unhandled exception

Be lenient on style; only fail on substance.
CONFIDENCE should reflect how well the response answers the query (100 = perfect)."""

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            max_tokens=150,
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
    t0 = time.time()
    response = _get_llm().invoke(messages)
    elapsed = round(time.time() - t0, 2)
    text = response.content.strip()

    verdict = "PASS"
    feedback = ""
    confidence = 85

    for line in text.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
        elif line.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = int(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = 85

    trace = state.get("agent_trace", [])
    trace.append({"agent": "critic", "verdict": verdict, "confidence": confidence, "latency_s": elapsed})

    if verdict == "PASS":
        return {
            "final_answer": latest_output,
            "critic_feedback": "",
            "confidence_score": confidence,
            "agent_trace": trace,
            "token_estimate": state.get("token_estimate", 0) + 150,
        }
    else:
        return {
            "critic_feedback": feedback,
            "retry_count": state.get("retry_count", 0) + 1,
            "confidence_score": confidence,
            "agent_trace": trace,
            "token_estimate": state.get("token_estimate", 0) + 150,
        }
