import os
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Real tools (actual Python functions, not the LLM guessing) ---
def calculator(expression):
    """Safely evaluates a basic math expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

def get_weather(city):
    """Fake weather tool simulating a real API - swap this for a real
    weather API call later and nothing else about this agent changes."""
    fake_weather_db = {
        "chennai": "32°C, humid, no rain expected",
        "london": "14°C, overcast, light rain",
        "tokyo": "22°C, clear skies",
    }
    return fake_weather_db.get(city.lower(), f"No weather data for {city}")

TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather,
}

SYSTEM_PROMPT = """You are an assistant that can use tools to answer questions accurately.

Available tools:
- calculator(expression): evaluates a math expression, e.g. calculator("340 * 0.15")
- get_weather(city): returns current weather for a city, e.g. get_weather("Chennai")

You MUST respond in EXACTLY this format, one step at a time:

Thought: <your reasoning about what to do next>
Action: <tool name, either calculator or get_weather>
Action Input: <the input to pass to that tool>

When you have enough information to answer the original question, respond with:

Thought: <your final reasoning>
Final Answer: <the complete answer to the user's question>

Only output ONE Thought/Action step at a time. Do not answer multiple steps at once.
"""

def run_agent(question, max_steps=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0  # deterministic - we want reliable tool-calling, not creativity
        )
        reply = response.choices[0].message.content
        print(f"--- Step {step + 1} ---")
        print(reply)
        print()

        # Check if the model gave its final answer
        if "Final Answer:" in reply:
            return reply.split("Final Answer:")[1].strip()

        # Otherwise, parse out the Action and Action Input
        action_match = re.search(r"Action:\s*(\w+)", reply)
        input_match = re.search(r"Action Input:\s*(.+)", reply)

        if not action_match or not input_match:
            return "Agent got confused and didn't follow the format."

        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip()

        # Actually RUN the tool in real Python code
        if action in TOOLS:
            observation = TOOLS[action](action_input)
        else:
            observation = f"Unknown tool: {action}"

        print(f"[Real tool executed] {action}({action_input}) -> {observation}\n")

        # Feed the real result back into the conversation, and let the
        # model reason again with this new information
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "Max steps reached without a final answer."


# --- Run it ---
question = "What's 15% of a $340 hotel bill, and is it currently raining in Chennai?"
final_answer = run_agent(question)
print("=" * 60)
print("FINAL ANSWER:", final_answer)