import os
from langchain_tavily import TavilySearch

def get_search_tool(max_results: int = 4) -> TavilySearch:
    return TavilySearch(
        max_results=max_results,
        api_key=os.environ.get("TAVILY_API_KEY", ""),
    )
