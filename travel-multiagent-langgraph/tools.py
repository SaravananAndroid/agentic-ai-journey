import os
import requests
from langchain_core.tools import tool

# --- Flight search (AviationStack - real schedule/status data) ---
@tool
def search_flights(dep_iata: str, arr_iata: str) -> str:
    """Searches REAL, currently scheduled flights between two airports.
    dep_iata/arr_iata must be 3-letter IATA codes (e.g. 'BOM', 'GOI')."""
    url = "http://api.aviationstack.com/v1/flights"
    params = {
        "access_key": os.getenv("AVIATIONSTACK_API_KEY"),
        "dep_iata": dep_iata,
        "arr_iata": arr_iata,
        "limit": 5
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        if "error" in data:
            return f"Flight search error: {data['error']}"

        flights = data.get("data", [])
        if not flights:
            return f"No scheduled flights found for {dep_iata} -> {arr_iata}."

        results = []
        for f in flights:
            airline = f["airline"]["name"]
            flight_num = f["flight"]["iata"]
            status = f["flight_status"]
            results.append(f"{airline} {flight_num} ({status})")
        return f"Flights {dep_iata}->{arr_iata}: " + ", ".join(results) + \
               " | Note: live pricing unavailable on free tier."
    except requests.exceptions.Timeout:
        return f"Flight search timed out for {dep_iata} -> {arr_iata}."
    except Exception as e:
        return f"Flight search failed: {e}"


# --- Hotel search (StayAPI - key goes in HEADER, not query param) ---
@tool
def search_hotels(hotel_name: str, location: str) -> str:
    """Searches for real hotel data by hotel name and location using StayAPI."""
    url = "https://api.stayapi.com/v1/meta/search"
    headers = {"x-api-key": os.getenv("STAYAPI_KEY")}
    params = {
        "hotel_name": hotel_name,
        "location": location
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        data = response.json()
        return f"Hotel search results for '{hotel_name}' in {location}: {data}"
    except requests.exceptions.Timeout:
        return f"Hotel search timed out for {location}."
    except Exception as e:
        return f"Hotel search failed: {e}"


# --- Budget calculator ---
@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression for budget calculations."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"