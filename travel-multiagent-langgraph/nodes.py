import os
import time
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from state import TravelState
from tools import search_flights, search_hotels, calculator

# --- Two models: fast/cheap-quota for simple calls, strong for reliable tool-use ---
# llm_fast = ChatGroq(
#     model="llama-3.1-8b-instant",
#     api_key=os.getenv("GROQ_API_KEY"),
#     temperature=0,
#     max_retries=1,
#     timeout=30
# )

# llm_strong = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     api_key=os.getenv("GROQ_API_KEY"),
#     temperature=0,
#     max_retries=1,
#     timeout=30
# )

llm_fast = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_retries=1,
    timeout=30
)

llm_strong = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_retries=1,
    timeout=30
)

flight_agent = create_react_agent(llm_strong, tools=[search_flights])  # upgraded after hallucination
hotel_agent = create_react_agent(llm_strong, tools=[search_hotels])
budget_agent = create_react_agent(llm_strong, tools=[calculator])


def get_original_request(state: TravelState) -> str:
    """Extracts the user's original request text, regardless of message format."""
    first_msg = state["messages"][0]
    return first_msg.content if hasattr(first_msg, "content") else first_msg["content"]


def supervisor_node(state: TravelState):
    return {}


def flight_node(state: TravelState):
    print("--> Entering flight_node")
    original_request = get_original_request(state)
    scoped_prompt = (
        f"You MUST call the search_flights tool to answer this - do not simulate, "
        f"guess, or describe what the output might be. Only report the ACTUAL "
        f"result returned by the tool. Extract origin/destination IATA codes and "
        f"call search_flights for this request: {original_request}"
    )
    result = flight_agent.invoke({"messages": [{"role": "user", "content": scoped_prompt}]})

    for msg in result["messages"]:
        print(f"    [{type(msg).__name__}]: {str(msg.content)[:150]}")

    return {"flight_info": result["messages"][-1].content, "messages": [result["messages"][-1]]}


def hotel_node(state: TravelState):
    print("--> Entering hotel_node")
    original_request = get_original_request(state)
    scoped_prompt = (
        f"You have exactly ONE tool available: search_hotels. Do not use any other tool. "
        f"You MUST call it - do not simulate or guess results. "
        f"Use search_hotels to find hotel information for this request: {original_request}"
    )
    result = hotel_agent.invoke({"messages": [{"role": "user", "content": scoped_prompt}]})

    for msg in result["messages"]:
        print(f"    [{type(msg).__name__}]: {str(msg.content)[:150]}")

    return {"hotel_info": result["messages"][-1].content, "messages": [result["messages"][-1]]}


def budget_node(state: TravelState):
    print("--> Entering budget_node")
    time.sleep(3) 
    context = (
        f"Flight info: {state['flight_info']}\n"
        f"Hotel info: {state['hotel_info']}\n"
        f"Use your calculator tool to estimate a rough total budget."
    )
    result = budget_agent.invoke({"messages": [{"role": "user", "content": context}]})
    return {"budget_summary": result["messages"][-1].content, "messages": [result["messages"][-1]]}