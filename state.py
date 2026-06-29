from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    route: str
    agent_outputs: dict[str, Any]
    retry_count: int
    final_answer: str
    critic_feedback: str
    # NEW: trace of which agents ran and when
    agent_trace: list[dict]
    # NEW: critic confidence score 0-100
    confidence_score: int
    # NEW: total tokens used estimate
    token_estimate: int
