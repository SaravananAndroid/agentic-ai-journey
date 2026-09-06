import chromadb

client = chromadb.PersistentClient(path="./production_db")  # change path to whichever DB

# List all collections in this database
collections = client.list_collections()
print("Collections found:", [c.name for c in collections])

# Look inside a specific collection
collection = client.get_collection(name="product_docs")  # change to your collection name
print(f"\nTotal documents: {collection.count()}")

# Peek at the actual stored data
data = collection.get(limit=10, include=["documents", "metadatas", "embeddings"])

for i, doc_id in enumerate(data["ids"]):
    print(f"\n--- ID: {doc_id} ---")
    print(f"Document: {data['documents'][i]}")
    print(f"Metadata: {data['metadatas'][i]}")
    print(f"Embedding (first 5 numbers): {data['embeddings'][i][:5]}")