import os
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_SYSTEM = """You are a writer specialist. Compose clear, well-structured responses.
If prior agent outputs are provided, synthesise them into a coherent answer.
Use markdown formatting where it improves readability."""

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            max_tokens=2048,
        )
    return _llm


def writer_node(state: AgentState) -> dict:
    prior = state.get("agent_outputs", {})
    context = (
        "\n\n".join(f"**{k.title()} Agent output:**\n{v}" for k, v in prior.items())
        if prior
        else ""
    )
    user_content = f"{state['query']}\n\n{context}".strip()
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=user_content)]
    t0 = time.time()
    response = _get_llm().invoke(messages)
    elapsed = round(time.time() - t0, 2)
    output = response.content

    trace = state.get("agent_trace", [])
    trace.append({"agent": "writer", "latency_s": elapsed})

    return {
        "agent_outputs": {**prior, "writer": output},
        "messages": [HumanMessage(content=f"[Writer Agent]: {output}")],
        "agent_trace": trace,
        "token_estimate": state.get("token_estimate", 0) + 2048,
    }
