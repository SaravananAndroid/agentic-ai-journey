SEPARATOR = "=" * 70


def ask_question(vectorstore, question: str, llm):
    """Stage 5 (Retrieval) + Stage 6 (Augmentation) + Stage 7 (Generation)."""

    # --- STAGE 5: RETRIEVAL ---
    print(f"\n{SEPARATOR}\nSTAGE 5: RETRIEVAL\n{SEPARATOR}")
    print(f"Question: {question}")
    print(f"\nSearching for the 3 most relevant chunks...")

    results = vectorstore.similarity_search_with_score(question, k=3)

    retrieved_chunks = []
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Retrieved chunk {i+1} (distance score: {score:.4f}, lower = more similar) ---")
        print(doc.page_content[:250])
        retrieved_chunks.append(doc.page_content)

    # --- STAGE 6: AUGMENTATION ---
    print(f"\n{SEPARATOR}\nSTAGE 6: AUGMENTATION\n{SEPARATOR}")
    context = "\n\n".join(retrieved_chunks)
    final_prompt = f"""Answer the question using ONLY the context below. If the answer isn't
in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    print("EXACT prompt being sent to the LLM:\n")
    print(final_prompt)

    # --- STAGE 7: GENERATION ---
    print(f"\n{SEPARATOR}\nSTAGE 7: GENERATION\n{SEPARATOR}")
    response = llm.invoke(final_prompt)
    print(f"Final Answer:\n{response.content}")

    return response.content