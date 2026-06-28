import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from state import AgentState
from tools.search import get_search_tool

_SYSTEM = """You are a research specialist. Use the search tool to find accurate,
up-to-date information. Always cite your sources. Be concise and factual."""

_search = None
_llm = None


def _get_llm():
    global _search, _llm
    if _llm is None:
        _search = get_search_tool()
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            max_tokens=1024,
        ).bind_tools([_search])
    return _llm, _search


def research_node(state: AgentState) -> dict:
    llm, search = _get_llm()
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=state["query"])]

    try:
        response = llm.invoke(messages)

        if response.tool_calls:
            tool_results = []
            for tc in response.tool_calls:
                result = search.invoke(tc["args"])
                tool_results.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
            messages = messages + [response] + tool_results
            response = llm.invoke(messages)

        output = response.content

    except Exception as e:
        # Fallback: answer without search if tool call fails
        fallback_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            max_tokens=1024,
        )
        fallback_msg = [
            SystemMessage(content="You are a helpful research assistant. Answer based on your knowledge."),
            HumanMessage(content=state["query"]),
        ]
        response = fallback_llm.invoke(fallback_msg)
        output = response.content

    return {
        "agent_outputs": {**state.get("agent_outputs", {}), "research": output},
        "messages": [HumanMessage(content=f"[Research Agent]: {output}")],
    }
