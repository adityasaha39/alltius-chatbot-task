import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("embeddings/faiss_index.bin")
with open("embeddings/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

def search(query, top_k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    print(f"Distances: {distances}")
    if distances[0][0] > 1.5: 
        return "I don't know."

    result = chunks[indices[0][0]]
    return result

if __name__ == "__main__":
    while True:
        query = input("Ask a question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = search(query)
        print(f"Bot: {answer}\n")
