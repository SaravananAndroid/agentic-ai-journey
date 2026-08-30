from langgraph.graph import StateGraph, END, START
from state import TravelState
from nodes import supervisor_node, flight_node, hotel_node, budget_node

graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("flight_agent", flight_node)
graph.add_node("hotel_agent", hotel_node)
graph.add_node("budget_agent", budget_node)

graph.add_edge(START, "supervisor")
graph.add_edge("supervisor", "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "budget_agent")
graph.add_edge("budget_agent", END)

app = graph.compile()