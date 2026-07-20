from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a free, local embedding model (downloads once, ~80MB, runs offline after)
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "How do I request time off?",
    "What is the vacation policy?",
    "How do I reset my password?",
    "I forgot my login credentials.",
    "What's the weather like today?",
]

# Convert each sentence into an embedding vector
embeddings = model.encode(sentences)

print(f"Each sentence became a vector of {embeddings.shape[1]} numbers.")
print(f"Example — first 10 numbers of sentence 1's embedding:\n{embeddings[0][:10]}\n")

# Compute cosine similarity between every pair of sentences
similarity_matrix = cosine_similarity(embeddings)

print("Similarity Matrix (1.0 = identical meaning, 0 = unrelated):\n")
print(" " * 25 + "".join([f"S{i+1}    " for i in range(len(sentences))]))
for i, row in enumerate(similarity_matrix):
    scores = "  ".join([f"{score:.2f}" for score in row])
    print(f"S{i+1} ({sentences[i][:20]:20}) {scores}")

print("\n--- Key observation ---")
print(f"S1 vs S2 (time off vs vacation policy): {similarity_matrix[0][1]:.3f}  <- high, despite NO shared words")
print(f"S3 vs S4 (password reset vs forgot login): {similarity_matrix[2][3]:.3f}  <- high, despite NO shared words")
print(f"S1 vs S5 (time off vs weather): {similarity_matrix[0][4]:.3f}  <- low, unrelated topics")