from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Full conversation history; add_messages merges lists rather than replacing
    messages: Annotated[list, add_messages]
    # The current user query being processed
    query: str
    # Which specialist the supervisor chose: "research" | "code" | "writer"
    route: str
    # Keyed outputs from each specialist so the Critic can inspect them
    agent_outputs: dict[str, Any]
    # How many times the Critic has rejected and we've retried
    retry_count: int
    # The final answer sent back to the user
    final_answer: str
    # Critic feedback forwarded to the Supervisor on retry
    critic_feedback: str
