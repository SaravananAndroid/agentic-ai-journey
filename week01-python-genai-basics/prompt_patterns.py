import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(prompt, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return response.choices[0].message.content


task = "A store had 23 apples. They sold 8, then received a delivery of 15 more. How many apples do they have now?"

# 1. Zero-shot
print("=" * 60)
print("1. ZERO-SHOT")
print("=" * 60)
print(ask(task))

# 2. Few-shot
few_shot_prompt = """Here are some examples of solving word problems:

Q: A bakery had 10 cakes. They sold 3. How many are left?
A: 10 - 3 = 7 cakes.

Q: A farmer had 50 eggs. He collected 20 more. How many total?
A: 50 + 20 = 70 eggs.

Now solve this one the same way:
Q: A store had 23 apples. They sold 8, then received a delivery of 15 more. How many apples do they have now?
A:"""
print("\n" + "=" * 60)
print("2. FEW-SHOT")
print("=" * 60)
print(ask(few_shot_prompt))

# 3. Chain-of-Thought
cot_prompt = task + "\n\nLet's think step by step."
print("\n" + "=" * 60)
print("3. CHAIN-OF-THOUGHT")
print("=" * 60)
print(ask(cot_prompt))

# 4. Role/System prompting
print("\n" + "=" * 60)
print("4. ROLE-BASED (System Prompt)")
print("=" * 60)
print(ask(task, system="You are a patient elementary school math teacher. Explain simply, like you're talking to a 9-year-old."))