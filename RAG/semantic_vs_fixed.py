from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import time

SEPARATOR = "=" * 70


def test_strategy(name, chunks, embeddings, test_question):
    print(f"\n{SEPARATOR}\nSTRATEGY: {name}\n{SEPARATOR}")
    print(f"Number of chunks: {len(chunks)}")
    lengths = [len(c.page_content) for c in chunks]
    print(f"Avg chunk length: {sum(lengths)//len(lengths)} chars "
          f"(min: {min(lengths)}, max: {max(lengths)})")

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    results = vectorstore.similarity_search_with_score(test_question, k=2)

    print(f"\nTop 2 retrieved chunks for: '{test_question}'")
    for i, (doc, score) in enumerate(results):
        print(f"\n  Result {i+1} (distance score: {score:.4f}):")
        print(f"  {doc.page_content[:250]}")
        print(f"  [chunk length: {len(doc.page_content)} chars]")


# --- Setup ---
pdf_path = input("Enter the path to your PDF file: ").strip().strip('"')
if not os.path.exists(pdf_path):
    print("File not found.")
    exit()

loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"Loaded {len(documents)} page(s).\n")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
test_question = input("Enter a test question about the document: ").strip()

# --- Strategy A: Fixed-size (Day 21/22's baseline) ---
fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
start = time.time()
fixed_chunks = fixed_splitter.split_documents(documents)
fixed_time = time.time() - start

test_strategy(
    f"Fixed-size (500 chars, 50 overlap) - built in {fixed_time:.2f}s",
    fixed_chunks, embeddings, test_question
)

# --- Strategy B: Semantic chunking ---
print(f"\n{SEPARATOR}\nBuilding semantic chunks (this embeds every sentence - slower)...\n{SEPARATOR}")
semantic_splitter = SemanticChunker(embeddings)
start = time.time()
semantic_chunks = semantic_splitter.split_documents(documents)
semantic_time = time.time() - start

test_strategy(
    f"Semantic chunking - built in {semantic_time:.2f}s",
    semantic_chunks, embeddings, test_question
)

# --- Final comparison summary ---
print(f"\n{SEPARATOR}\nSUMMARY\n{SEPARATOR}")
print(f"Fixed-size:  {len(fixed_chunks)} chunks, built in {fixed_time:.2f}s")
print(f"Semantic:    {len(semantic_chunks)} chunks, built in {semantic_time:.2f}s")
print(f"\nSemantic chunking took {semantic_time/fixed_time:.1f}x longer to build.")
print("Now compare: which strategy's retrieved chunks felt more complete and on-topic?")