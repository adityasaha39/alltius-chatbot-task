import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

with open("data/source.txt", "r") as f:
    full_text = f.read()

chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, "embeddings/faiss_index.bin")

with open("embeddings/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"Successfully built FAISS index with {len(chunks)} chunks!")
