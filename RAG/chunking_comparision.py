from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

SEPARATOR = "=" * 70

def test_chunking_strategy(name, chunks, embeddings, test_question):
    """Builds a temporary vectorstore for ONE chunking strategy and checks retrieval quality."""
    print(f"\n{SEPARATOR}\nSTRATEGY: {name}\n{SEPARATOR}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Average chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    results = vectorstore.similarity_search_with_score(test_question, k=2)

    print(f"\nTop 2 retrieved chunks for question: '{test_question}'")
    for i, (doc, score) in enumerate(results):
        print(f"\n  Result {i+1} (distance score: {score:.4f}):")
        print(f"  {doc.page_content[:200]}")

    return results


# --- Load document once ---
pdf_path = input("Enter the path to your PDF file: ").strip().strip('"')
if not os.path.exists(pdf_path):
    print("File not found.")
    exit()

loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"Loaded {len(documents)} page(s).\n")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
test_question = input("Enter a test question about the document: ").strip()

# --- Strategy 1: Small chunks, no overlap ---
splitter_small = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
chunks_small = splitter_small.split_documents(documents)
test_chunking_strategy("Small chunks (200 chars, no overlap)", chunks_small, embeddings, test_question)

# --- Strategy 2: Medium chunks with overlap (Day 21's default) ---
splitter_medium = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks_medium = splitter_medium.split_documents(documents)
test_chunking_strategy("Medium chunks (500 chars, 50 overlap)", chunks_medium, embeddings, test_question)

# --- Strategy 3: Large chunks with overlap ---
splitter_large = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
chunks_large = splitter_large.split_documents(documents)
test_chunking_strategy("Large chunks (1500 chars, 150 overlap)", chunks_large, embeddings, test_question)

# --- Strategy 4: Naive splitting (ignores natural boundaries) ---
splitter_naive = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
chunks_naive = splitter_naive.split_documents(documents)
test_chunking_strategy("Naive splitting (fixed 500 chars, \\n separator only)", chunks_naive, embeddings, test_question)

print(f"\n{SEPARATOR}\nCOMPARISON COMPLETE\n{SEPARATOR}")
print("Compare the distance scores and retrieved text across strategies above.")
print("Lower distance score = better match. Also judge: does the retrieved")
print("chunk contain a COMPLETE thought, or does it feel cut off mid-idea?")