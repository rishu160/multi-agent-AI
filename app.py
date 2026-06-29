from uuid import uuid4
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import graph

app = FastAPI(title="Multi-Agent AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store for stats
_session_store: dict[str, list] = defaultdict(list)


class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    route: str
    retry_count: int
    agent_outputs: dict
    agent_trace: list
    confidence_score: int
    token_estimate: int


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "query": req.query,
        "messages": [],
        "route": "",
        "agent_outputs": {},
        "retry_count": 0,
        "final_answer": "",
        "critic_feedback": "",
        "agent_trace": [],
        "confidence_score": 0,
        "token_estimate": 0,
    }

    try:
        result = graph.invoke(initial_state, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    response = ChatResponse(
        thread_id=thread_id,
        answer=result.get("final_answer", "No answer generated."),
        route=result.get("route", ""),
        retry_count=result.get("retry_count", 0),
        agent_outputs=result.get("agent_outputs", {}),
        agent_trace=result.get("agent_trace", []),
        confidence_score=result.get("confidence_score", 0),
        token_estimate=result.get("token_estimate", 0),
    )

    # Store for history/stats
    _session_store[thread_id].append({
        "query": req.query,
        "route": response.route,
        "confidence_score": response.confidence_score,
        "retry_count": response.retry_count,
    })

    return response


@app.get("/history/{thread_id}")
async def history(thread_id: str):
    return {"thread_id": thread_id, "history": _session_store.get(thread_id, [])}


@app.get("/stats")
async def stats():
    all_entries = [e for entries in _session_store.values() for e in entries]
    if not all_entries:
        return {"total_queries": 0}
    route_counts = defaultdict(int)
    for e in all_entries:
        route_counts[e["route"]] += 1
    avg_confidence = sum(e["confidence_score"] for e in all_entries) / len(all_entries)
    return {
        "total_queries": len(all_entries),
        "route_distribution": dict(route_counts),
        "avg_confidence": round(avg_confidence, 1),
        "total_sessions": len(_session_store),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
