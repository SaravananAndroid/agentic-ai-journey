import os
from dotenv import load_dotenv
from groq import Groq
from google import genai
import ollama

load_dotenv()

class LLMClient:
    """
    A single, unified interface for calling Groq, Gemini, or Ollama.
    Every future script in this course will import THIS instead of
    writing provider-specific code directly.
    """

    def __init__(self, provider="groq"):
        self.provider = provider

        if provider == "groq":
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        elif provider == "gemini":
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        elif provider == "ollama":
            self.client = None  # ollama package doesn't need a client object
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def ask(self, prompt, system=None, temperature=0.7, max_tokens=500):
        """One method. Same signature. Works no matter which provider is active."""

        if self.provider == "groq":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif self.provider == "gemini":
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = self.client.models.generate_content(
                model="gemini-flash-latest",
                contents=full_prompt
            )
            return response.text

        elif self.provider == "ollama":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = ollama.chat(model="llama3.1", messages=messages)
            return response["message"]["content"]


# --- Demo: prove the SAME code works across all 3 providers ---
if __name__ == "__main__":
    prompt = "In one sentence, what is an AI agent?"

    for provider in ["groq", "gemini", "ollama"]:
        print("=" * 60)
        print(f"PROVIDER: {provider}")
        print("=" * 60)
        try:
            llm = LLMClient(provider=provider)
            print(llm.ask(prompt))
        except Exception as e:
            print(f"Error with {provider}: {e}")
        print()