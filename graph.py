import os
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from agents.supervisor import supervisor_node
from agents.research import research_node
from agents.code import code_node
from agents.writer import writer_node
from agents.critic import critic_node
from agents.math_agent import math_node

MAX_RETRIES = 2


def route_after_supervisor(state: AgentState) -> str:
    return state["route"]


def route_after_critic(state: AgentState) -> str:
    if state.get("final_answer"):
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "end"
    return "retry"


def _promote_best_effort(state: AgentState) -> dict:
    outputs = state.get("agent_outputs", {})
    best = next(reversed(outputs.values())) if outputs else "I was unable to produce a satisfactory answer."
    return {"final_answer": best}


def build_graph(db_path: str | None = None):
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research", research_node)
    builder.add_node("code", code_node)
    builder.add_node("math", math_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("promote_best_effort", _promote_best_effort)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"research": "research", "code": "code", "math": "math", "writer": "writer"},
    )

    for specialist in ("research", "code", "math", "writer"):
        builder.add_edge(specialist, "critic")

    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"end": "promote_best_effort", "retry": "supervisor"},
    )

    builder.add_edge("promote_best_effort", END)

    path = db_path or os.environ.get("SQLITE_DB_PATH", "checkpoints.db")
    try:
        conn = sqlite3.connect(path, check_same_thread=False)
        memory = SqliteSaver(conn)
    except Exception:
        memory = MemorySaver()
    return builder.compile(checkpointer=memory)


graph = build_graph()
