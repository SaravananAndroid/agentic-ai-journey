import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the same free local embedding model from Day 6
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Our "document collection" (imagine this is 50,000 support tickets) ---
documents = [
    "How do I request time off for a vacation?",
    "What is the company's paid leave policy?",
    "My laptop screen is broken, can I get a refund?",
    "The customer wants a refund for a damaged electronic item.",
    "How do I reset my account password?",
    "I forgot my login credentials and can't sign in.",
    "What's the weather forecast for tomorrow?",
    "Steps to configure the office printer on Windows.",
    "How to request a refund for a defective product.",
    "I need help changing my forgotten password.",
]

# Step 1: Convert all documents into embeddings
print("Embedding documents...")
doc_embeddings = model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype('float32')  # FAISS requires float32

embedding_dim = doc_embeddings.shape[1]
print(f"Each document embedded into a {embedding_dim}-dimensional vector.\n")

# Step 2: Build a FAISS index
# IndexFlatL2 = exact search using L2 (Euclidean) distance - simplest FAISS index,
# good for learning; production systems with millions of vectors would use
# IndexIVFFlat or IndexHNSWFlat instead for approximate (faster) search.
index = faiss.IndexFlatL2(embedding_dim)
index.add(doc_embeddings)
print(f"FAISS index built with {index.ntotal} vectors.\n")

# Step 3: Search - given a new query, find the most similar documents
def semantic_search(query, top_k=3):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)

    print(f"Query: '{query}'")
    print(f"Top {top_k} results:")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        print(f"  {rank}. (distance={dist:.3f}) {documents[idx]}")
    print()

# --- Test it with queries that share almost NO keywords with the matching docs ---
semantic_search("customer wants money back for broken item")
semantic_search("can't log into my account")
semantic_search("how many vacation days do I get")