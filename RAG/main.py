import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from document_loader import load_and_chunk
from vector_store import build_vectorstore
from rag_chain import ask_question

load_dotenv()

SEPARATOR = "=" * 70


def main():
    pdf_path = input("Enter the path to your PDF file: ").strip().strip('"')

    if not os.path.exists(pdf_path):
        print(f"File not found: '{pdf_path}'.")
        return

    chunks = load_and_chunk(pdf_path)
    vectorstore = build_vectorstore(chunks)

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
# C:\Users\Dharshith\Downloads\javaCollctions.pdf
    print(f"\n{SEPARATOR}\nSETUP COMPLETE. Ask questions below (type 'quit' to exit).\n{SEPARATOR}")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not question:
            continue

        ask_question(vectorstore, question, llm)


if __name__ == "__main__":
    main()