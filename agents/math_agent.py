import os
import time
import math
import statistics
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

_SAFE_GLOBALS = {
    "__builtins__": {},
    "math": math,
    "statistics": statistics,
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "range": range, "list": list,
    "int": int, "float": float, "pow": pow, "divmod": divmod,
}

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            max_tokens=512,
        )
    return _llm


def _try_direct_eval(query: str):
    """Try to eval a simple math expression directly."""
    import re
    # Extract a math expression from the query
    expr = re.sub(r"[^0-9+\-*/().,%^ ]", "", query).strip()
    expr = expr.replace("^", "**").replace("%", "/100")
    if expr:
        try:
            result = eval(expr, _SAFE_GLOBALS)  # noqa: S307
            return result
        except Exception:
            pass
    return None


def math_node(state: AgentState) -> dict:
    t0 = time.time()

    # Try fast direct eval first
    direct = _try_direct_eval(state["query"])
    if direct is not None:
        output = f"**Result:** `{direct}`\n\n*Computed directly from expression.*"
    else:
        # Fall back to LLM with a math-focused prompt
        messages = [
            SystemMessage(content=(
                "You are a precise math solver. Solve the problem step-by-step. "
                "Show your working clearly. Give the final numeric answer on the last line "
                "prefixed with 'Answer:'. Use LaTeX-style notation where helpful."
            )),
            HumanMessage(content=state["query"]),
        ]
        response = _get_llm().invoke(messages)
        output = response.content

    elapsed = round(time.time() - t0, 2)
    trace = state.get("agent_trace", [])
    trace.append({"agent": "math", "latency_s": elapsed})

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "math": output},
        "messages": [HumanMessage(content=f"[Math Agent]: {output}")],
        "agent_trace": trace,
        "token_estimate": state.get("token_estimate", 0) + 150,
    }
