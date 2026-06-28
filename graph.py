import os
import sqlite3
from contextlib import closing
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from agents.supervisor import supervisor_node
from agents.research import research_node
from agents.code import code_node
from agents.writer import writer_node
from agents.critic import critic_node

MAX_RETRIES = 2


# ── Conditional edge: Supervisor → specialist ─────────────────────────────────
def route_after_supervisor(state: AgentState) -> str:
    return state["route"]  # "research" | "code" | "writer"


# ── Conditional edge: Critic → END or retry ───────────────────────────────────
def route_after_critic(state: AgentState) -> str:
    if state.get("final_answer"):
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        # Best-effort: promote whatever we have
        return "end"
    return "retry"


def _promote_best_effort(state: AgentState) -> dict:
    """Fallback node: when retries are exhausted, use the last agent output."""
    outputs = state.get("agent_outputs", {})
    best = next(reversed(outputs.values())) if outputs else "I was unable to produce a satisfactory answer."
    return {"final_answer": best}


def build_graph(db_path: str | None = None) -> "CompiledGraph":  # type: ignore[name-defined]
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research", research_node)
    builder.add_node("code", code_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("promote_best_effort", _promote_best_effort)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("supervisor")

    # ── Supervisor → one of the three specialists ─────────────────────────────
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"research": "research", "code": "code", "writer": "writer"},
    )

    # ── Every specialist feeds into the Critic ────────────────────────────────
    for specialist in ("research", "code", "writer"):
        builder.add_edge(specialist, "critic")

    # ── Critic decides: done, retry, or best-effort ───────────────────────────
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end": "promote_best_effort",   # PASS path also normalises via this node
            "retry": "supervisor",          # loop back with feedback
        },
    )

    # ── promote_best_effort is always terminal ────────────────────────────────
    builder.add_edge("promote_best_effort", END)

    # ── Checkpointer (SQLite for persistence) ────────────────────────────────
    # SqliteSaver.from_conn_string is a context manager in langgraph-checkpoint-sqlite 3.x
    # so we open the connection directly and pass it to the constructor.
    path = db_path or os.environ.get("SQLITE_DB_PATH", "checkpoints.db")
    try:
        conn = sqlite3.connect(path, check_same_thread=False)
        memory = SqliteSaver(conn)
    except Exception:
        memory = MemorySaver()  # in-memory fallback
    return builder.compile(checkpointer=memory)


# Singleton used by app.py and streamlit_app.py
graph = build_graph()
