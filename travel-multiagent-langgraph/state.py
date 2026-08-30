from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class TravelState(TypedDict):
    """Shared state passed between all agents in the graph."""
    messages: Annotated[list, add_messages]
    flight_info: str
    hotel_info: str
    budget_summary: str