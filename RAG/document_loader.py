from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARATOR = "=" * 70


def load_and_chunk(pdf_path: str):
    """Stage 1 (Loading) + Stage 2 (Chunking). Returns a list of chunks."""

    # --- STAGE 1: LOADING ---
    print(f"\n{SEPARATOR}\nSTAGE 1: LOADING\n{SEPARATOR}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s).")
    print(f"\nPreview of page 1 (first 300 chars):")
    print(documents[0].page_content[:300])
    print(f"\nMetadata of page 1: {documents[0].metadata}")

    # --- STAGE 2: CHUNKING ---
    print(f"\n{SEPARATOR}\nSTAGE 2: CHUNKING\n{SEPARATOR}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")
    print(f"\nShowing first 2 chunks in full:\n")
    for i, chunk in enumerate(chunks[:2]):
        print(f"--- Chunk {i+1} (length: {len(chunk.page_content)} chars) ---")
        print(chunk.page_content)
        print()

    return chunks