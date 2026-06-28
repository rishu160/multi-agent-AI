import uuid
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph import graph

app = FastAPI(title="Multi-Agent AI Assistant")


class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None  # omit to start a new conversation


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    route: str
    retry_count: int
    agent_outputs: dict


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "query": req.query,
        "messages": [],
        "route": "",
        "agent_outputs": {},
        "retry_count": 0,
        "final_answer": "",
        "critic_feedback": "",
    }

    try:
        result = graph.invoke(initial_state, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        thread_id=thread_id,
        answer=result.get("final_answer", "No answer generated."),
        route=result.get("route", ""),
        retry_count=result.get("retry_count", 0),
        agent_outputs=result.get("agent_outputs", {}),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
