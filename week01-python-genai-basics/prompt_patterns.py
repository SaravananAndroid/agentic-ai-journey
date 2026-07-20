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
# Zero-shot — you give only the instruction.
#The model must map your task purely from patterns it learned during training. 
#Works well for common tasks it has seen thousands of times (basic Q&A, summarization), unreliable for niche or precisely-formatted tasks.

# 1. Zero-shot
print("=" * 60)
print("1. ZERO-SHOT")
print("=" * 60)
print(ask(task))

# Few-shot — by placing 2–3 example input→ou tput pairs in the prompt, you're not "teaching" the model in the training sense — you're biasing token prediction toward continuing that specific pattern. This is called in-context learning: the model never updates its weights, it just conditions on the examples present in this one prompt.
# Chain-of-Thought (CoT) — LLMs generate tokens one at a time with no ability to "revise" earlier tokens. If you ask for a direct answer to a multi-step problem, the model has to get it right in one shot with no scratch space. Asking it to "think step by step" forces it to generate intermediate reasoning tokens first — and since each new token is conditioned on all previous tokens (including its own reasoning so far), those intermediate steps act as a working memory that measurably improves accuracy on math/logic tasks.
# Role/System prompting — the system message sits at the start of the context and every subsequent token generation is conditioned on it. It doesn't grant new capabilities — it re-weights tone, vocabulary, and framing toward what a person with that persona would plausibly say.
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

# Chain-of-Thought (CoT) — LLMs generate tokens one at a time with no ability to "revise" earlier tokens. If you ask for a direct answer to a multi-step problem, the model has to get it right in one shot with no scratch space. Asking it to "think step by step" forces it to generate intermediate reasoning tokens first — and since each new token is conditioned on all previous tokens (including its own reasoning so far), those intermediate steps act as a working memory that measurably improves accuracy on math/logic tasks.

# 3. Chain-of-Thought
cot_prompt = task + "\n\nLet's think step by step."
print("\n" + "=" * 60)
print("3. CHAIN-OF-THOUGHT")
print("=" * 60)
print(ask(cot_prompt))

# Role/System prompting — the system message sits at the start of the context and every subsequent token generation is conditioned on it. It doesn't grant new capabilities — it re-weights tone, vocabulary, and framing toward what a person with that persona would plausibly say.
# 4. Role/System prompting
print("\n" + "=" * 60)
print("4. ROLE-BASED (System Prompt)")
print("=" * 60)
print(ask(task, system="You are a patient elementary school math teacher. Explain simply, like you're talking to a 9-year-old."))