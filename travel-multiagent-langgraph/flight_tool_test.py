import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
BASE_URL = "http://api.aviationstack.com/v1/flights"

def search_flights(dep_iata, arr_iata):
    """Searches REAL, currently scheduled flights between two airports
    using AviationStack's free tier. Returns actual flight numbers,
    airlines, and status - but NOT live pricing (that data isn't
    available for free anywhere right now, so we estimate it separately)."""
    params = {
        "access_key": API_KEY,
        "dep_iata": dep_iata,
        "arr_iata": arr_iata,
        "limit": 5
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "error" in data:
        print(f"API Error: {data['error']}")
        return

    flights = data.get("data", [])
    if not flights:
        print(f"No scheduled flights found for {dep_iata} -> {arr_iata} right now.")
        return

    print(f"Found {len(flights)} real scheduled flight(s):\n")
    for f in flights:
        airline = f["airline"]["name"]
        flight_num = f["flight"]["iata"]
        status = f["flight_status"]
        dep_time = f["departure"]["scheduled"]
        print(f"  {airline} {flight_num} | Status: {status} | Departs: {dep_time}")

def estimate_price(dep_iata, arr_iata):
    """Honest note: this is a rough distance-independent placeholder,
    NOT real pricing - free live fare data isn't available anywhere
    currently. Flag this clearly to the user in any real output."""
    return "Price data unavailable on free tier - estimate: ₹4,500-9,000 (domestic route average)"

# --- Test it ---
search_flights("BOM", "GOI")
print()
print(estimate_price("BOM", "GOI"))