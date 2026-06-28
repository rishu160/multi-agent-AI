# Multi-Agent AI Assistant

A production-style multi-agent system built with **LangGraph**, **Claude Sonnet**, and **FastAPI**.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│  Supervisor │  ← classifies query → route: research | code | writer
└──────┬──────┘
       │
   ┌───┴────────────┐
   ▼                ▼                ▼
Research Agent   Code Agent      Writer Agent
(Tavily search)  (Python REPL)   (prose synthesis)
   │                │                │
   └────────────────┴────────────────┘
                    │
                    ▼
            ┌──────────────┐
            │ Critic Agent │  ← PASS → final answer
            └──────┬───────┘    FAIL + retry_count < 2 → back to Supervisor
                   │            FAIL + retries exhausted → best-effort answer
                   ▼
             Final Answer
```

### Agent roles

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| Supervisor | Route query to correct specialist | None (structured LLM output) |
| Research | Find current facts and information | Tavily web search |
| Code | Write and execute Python code | Python REPL |
| Writer | Compose polished prose / synthesise | None |
| Critic | Quality-gate every specialist output | None |

### Why a Critic loop?

LLM outputs are non-deterministic. A single agent can confidently produce wrong or incomplete answers. The Critic adds a cheap second-opinion pass (~128 output tokens) that catches:
- Misrouted queries answered incorrectly
- Code that raised an exception
- Truncated or off-topic responses

If it rejects, the Supervisor re-routes with the feedback attached, so the retry is informed rather than blind. Max 2 retries prevents infinite loops, and the best-effort fallback guarantees the user always gets *something*.

---

## Project Structure

```
├── agents/
│   ├── supervisor.py   # classifies query, sets route
│   ├── research.py     # Tavily-powered research
│   ├── code.py         # Python REPL code execution
│   ├── writer.py       # prose composition
│   └── critic.py       # quality gate
├── tools/
│   ├── search.py       # Tavily wrapper
│   └── code_exec.py    # sandboxed Python exec tool
├── eval/
│   └── eval.py         # 20-query benchmark
├── graph.py            # LangGraph StateGraph
├── app.py              # FastAPI /chat endpoint
├── streamlit_app.py    # Chat UI
├── state.py            # AgentState TypedDict
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in your API keys:
# ANTHROPIC_API_KEY — from console.anthropic.com
# TAVILY_API_KEY    — from app.tavily.com
# LANGCHAIN_API_KEY — from smith.langchain.com (optional, for tracing)
```

### 3. Run the backend
```bash
uvicorn app:app --reload
```

### 4. Run the frontend (separate terminal)
```bash
streamlit run streamlit_app.py
```

### 5. Run the evaluation
```bash
# Make sure the backend is running first
python eval/eval.py
```

---

## Shared State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation history
    query: str           # current user query
    route: str           # supervisor's routing decision
    agent_outputs: dict  # keyed outputs from each specialist
    retry_count: int     # Critic rejection counter
    final_answer: str    # promoted when Critic passes
    critic_feedback: str # forwarded to Supervisor on retry
```

`add_messages` is a LangGraph reducer that merges message lists rather than replacing them, which is how multi-turn memory works without extra code.

---

## LangSmith Tracing

Set the `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` env vars — no code changes needed. Every graph invocation will appear in your LangSmith dashboard with full node-level traces.

---

## Design Decisions

### Why LangGraph over plain LangChain chains?
LangGraph's `StateGraph` makes conditional branching (the Critic retry loop) first-class. With chains you'd need manual recursion or callbacks; with LangGraph it's a single `add_conditional_edges` call.

### Why SQLite checkpointer?
It's zero-infrastructure for development and gives you free multi-turn memory. Swap to `PostgresSaver` for production with one line change.

### Why Claude Sonnet for all agents?
Consistency: all agents share the same capability baseline so the Critic's quality bar is calibrated. The Supervisor uses `max_tokens=10` to keep routing cheap.

### Why is the Python REPL sandboxed manually rather than using `langchain_experimental.PythonREPLTool`?
`langchain_experimental` sandboxing is minimal. Our wrapper explicitly blocks `open`, `exec`, `eval`, and `__import__` to reduce risk while still supporting pure computation.
