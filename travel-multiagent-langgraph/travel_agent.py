from dotenv import load_dotenv
load_dotenv()

from graph import app

if __name__ == "__main__":
    result = app.invoke({
        "messages": [{"role": "user", "content":
            "Find flights from BOM to GOI, and hotels in Goa, checkin 2026-09-15, checkout 2026-09-18."}],
        "flight_info": "",
        "hotel_info": "",
        "budget_summary": ""
    })

    print("\n" + "=" * 60)
    print("FLIGHT INFO:", result["flight_info"])
    print("=" * 60)
    print("HOTEL INFO:", result["hotel_info"])
    print("=" * 60)
    print("BUDGET SUMMARY:", result["budget_summary"])