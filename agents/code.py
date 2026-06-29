import os
import re
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from tools.code_exec import python_repl

_SYSTEM = """You are a code specialist. When given a query, write Python code to solve it.

IMPORTANT: Always wrap your code in ```python ... ``` blocks.
Always use print() to show the final result.
Keep the code simple and focused."""

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


def _extract_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def code_node(state: AgentState) -> dict:
    llm = _get_llm()
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=state["query"])]
    t0 = time.time()
    response = llm.invoke(messages)

    explanation = response.content
    code = _extract_code(explanation)

    exec_result = ""
    if code:
        exec_result = python_repl.invoke({"code": code})
        output = f"{explanation}\n\n**Execution Output:**\n```\n{exec_result}\n```"
    else:
        output = explanation

    elapsed = round(time.time() - t0, 2)
    trace = state.get("agent_trace", [])
    trace.append({"agent": "code", "executed": bool(code), "latency_s": elapsed})

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "code": output},
        "messages": [HumanMessage(content=f"[Code Agent]: {output}")],
        "agent_trace": trace,
        "token_estimate": state.get("token_estimate", 0) + 2048,
    }
