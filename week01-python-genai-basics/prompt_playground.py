import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(prompt, temperature=0.7, max_tokens=100, top_p=1.0):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return response.choices[0].message.content


prompt = "explain what an AI agent is."

# Run the SAME prompt twice at each temperature to observe determinism vs. variability
for temp in [0.0, 0.7, 1.5]:
    print("=" * 60)
    print(f"TEMPERATURE = {temp}  (running twice)")
    print("=" * 60)
    print("Run 1:", ask(prompt, temperature=temp))
    print("Run 2:", ask(prompt, temperature=temp))
    print()

# --- Bonus: demonstrate max_tokens cutting a response short ---
print("=" * 60)
print("MAX_TOKENS = 15 (deliberately too short)")
print("=" * 60)
print(ask("Explain how photosynthesis works.", max_tokens=15))