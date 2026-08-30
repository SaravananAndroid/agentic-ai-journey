from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

SEPARATOR = "=" * 70


def build_vectorstore(chunks):
    """Stage 3 (Embedding) + Stage 4 (Storage). Returns a ready-to-query vectorstore."""

    # --- STAGE 3: EMBEDDING ---
    print(f"{SEPARATOR}\nSTAGE 3: EMBEDDING\n{SEPARATOR}")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    sample_vector = embeddings.embed_query(chunks[0].page_content)
    print(f"Each chunk becomes a vector of {len(sample_vector)} numbers.")
    print(f"First 10 numbers of chunk 1's embedding:\n{sample_vector[:10]}")

    # --- STAGE 4: STORAGE ---
    print(f"\n{SEPARATOR}\nSTAGE 4: STORING IN VECTOR DATABASE\n{SEPARATOR}")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print(f"Stored {len(chunks)} chunks with their embeddings in Chroma.")

    return vectorstore