import os
from dotenv import load_dotenv
from groq import Groq
from google import genai

load_dotenv()

prompt = "In one sentence, explain what an AI agent is."

# --- Groq (Llama 3.1) ---
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
groq_response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}]
)

# --- Gemini (new google-genai SDK) ---
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
gemini_response = gemini_client.models.generate_content(
    model="gemini-flash-latest",
    contents=prompt
)

# --- Print both side by side ---
print("=" * 50)
print("PROMPT:", prompt)
print("=" * 50)
print("\nGROQ (Llama 3.1):\n", groq_response.choices[0].message.content)
print("\n" + "=" * 50)
print("\nGEMINI:\n", gemini_response.text)