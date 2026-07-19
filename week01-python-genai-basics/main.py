import os
from dotenv import load_dotenv
from groq import Groq

# Load the .env file so we can read the API key
load_dotenv()

# Create a client using your Groq API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Send a prompt to the free LLM
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",python main.py
    messages=[
        {"role": "user", "content": "In one sentence, explain what an AI agent is."}
    ]
)

# Print the model's reply
print(response.choices[0].message.content)