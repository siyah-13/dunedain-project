import json
from sentence_transformers import SentenceTransformer, util
import torch

# Load PDF chunks
with open("fm5_chunks.json", "r") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]
sources = [chunk["source"] for chunk in chunks]

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all chunks
print("🔄 Embedding chunks...")
embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

# Loop: accept user questions
while True:
    query = input("\n💬 Ask a question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute similarity
    hits = util.semantic_search(query_embedding, embeddings, top_k=5)[0]

    print("\n🔍 Top matching chunks:")
    for i, hit in enumerate(hits):
        score = hit["score"]
        idx = hit["corpus_id"]
        print(f"\n[{i+1}] (Score: {score:.4f}) - {sources[idx]}")
        print(texts[idx])
