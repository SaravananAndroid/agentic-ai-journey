import chromadb
from rank_bm25 import BM25Okapi

client = chromadb.PersistentClient(path="./production_db")

# --- 1. COLLECTIONS: separate namespaces ---
hr_collection = client.get_or_create_collection(name="hr_docs")
product_collection = client.get_or_create_collection(name="product_docs")

print("Created two isolated collections: 'hr_docs' and 'product_docs'\n")

# --- 2. UPSERT: add or update without duplication ---
product_collection.upsert(
    documents=["Error E-4471 occurs when the API rate limit is exceeded."],
    ids=["changelog_v1"]
)
print("Inserted changelog_v1 (first version)")

# Now "update" the SAME id with revised content - this REPLACES, not duplicates
product_collection.upsert(
    documents=["Error E-4471 occurs when the API rate limit is exceeded. Fixed in v2.3 with automatic retry logic."],
    ids=["changelog_v1"]
)
print("Upserted changelog_v1 (revised) - still only ONE entry, not two\n")
print(f"Collection count: {product_collection.count()} (proves no duplicate was created)\n")

# --- 3. COMPOUND METADATA FILTERING ---
product_collection.upsert(
    documents=["Refund processed for damaged laptop screen.", "Refund denied - past 30 day window."],
    metadatas=[
        {"category": "refund", "status": "approved"},
        {"category": "refund", "status": "denied"}
    ],
    ids=["ticket_1", "ticket_2"]
)

results = product_collection.query(
    query_texts=["laptop screen issue"],
    where={"$and": [{"category": "refund"}, {"status": "approved"}]},  # compound filter
    n_results=2
)
print("Compound filter results (category=refund AND status=approved):")
print(results['documents'][0])

# --- 4. HYBRID SEARCH: semantic + keyword (BM25) ---
print(f"\n{'=' * 60}\nHYBRID SEARCH DEMO\n{'=' * 60}")

all_docs = [
    "Error E-4471 occurs when the API rate limit is exceeded.",
    "The application may crash unexpectedly during high load.",
    "Users report slow performance during peak hours.",
]

query = "E-4471"

# Semantic search alone
semantic_results = product_collection.query(query_texts=[query], n_results=2)
print(f"\nPure semantic search for '{query}':")
print(semantic_results['documents'][0])

# Keyword search (BM25) - catches EXACT term matches semantic search might rank lower
tokenized_docs = [doc.split() for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)
bm25_scores = bm25.get_scores(query.split())
print(f"\nBM25 keyword scores for '{query}':")
for doc, score in zip(all_docs, bm25_scores):
    print(f"  {score:.2f} - {doc}")
print("\n(Notice BM25 heavily favors the doc containing the EXACT code 'E-4471',")
print(" which pure semantic search might rank lower since embeddings don't")
print(" specially recognize alphanumeric codes as meaningfully 'similar'.)")