import chromadb

# Step 1: Create a PERSISTENT client - this writes to disk at ./chroma_db
# Run this script twice: first run embeds and stores everything,
# second run will find the data already there (try commenting out
# the "add" section after your first run to prove this).
client = chromadb.PersistentClient(path="./chroma_db")

# Step 2: Create (or get, if it already exists) a "collection" -
# think of this like a table in a traditional database
collection = client.get_or_create_collection(name="support_tickets")

# Step 3: Add documents WITH metadata
# Chroma handles embedding internally by default (using its own model),
# and stores the text + metadata + vector together as one record.
documents = [
    "How do I request time off for a vacation?",
    "What is the company's paid leave policy?",
    "My laptop screen is broken, can I get a refund?",
    "The customer wants a refund for a damaged electronic item.",
    "How do I reset my account password?",
    "I forgot my login credentials and can't sign in.",
    "How to request a refund for a defective product.",
    "I need help changing my forgotten password.",
]

metadatas = [
    {"department": "HR", "category": "leave"},
    {"department": "HR", "category": "leave"},
    {"department": "Finance", "category": "refund"},
    {"department": "Finance", "category": "refund"},
    {"department": "IT", "category": "account"},
    {"department": "IT", "category": "account"},
    {"department": "Finance", "category": "refund"},
    {"department": "IT", "category": "account"},
]

# ids must be unique strings - Chroma requires this to identify each record
ids = [f"doc_{i}" for i in range(len(documents))]

# Only add if the collection is empty (avoids duplicate errors on re-runs)
if collection.count() == 0:
    print("Adding documents to ChromaDB (first run)...")
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
else:
    print(f"Collection already has {collection.count()} documents (loaded from disk).")

print()

# Step 4: Semantic search WITHOUT any filter
print("=" * 60)
print("SEARCH (no filter): 'my password isn't working'")
print("=" * 60)
results = collection.query(
    query_texts=["my password isn't working"],
    n_results=3
)
for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
    print(f"  (dist={dist:.3f}) [{meta['department']}] {doc}")

print()

# Step 5: Semantic search WITH a metadata filter
print("=" * 60)
print("SEARCH (filtered to Finance only): 'I want my money back'")
print("=" * 60)
results = collection.query(
    query_texts=["I want my money back"],
    n_results=3,
    where={"department": "Finance"}   # <-- metadata filter combined with semantic search
)
for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
    print(f"  (dist={dist:.3f}) [{meta['department']}] {doc}")