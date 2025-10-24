import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(text):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    vectors = model.encode(chunks)
    return chunks, vectors

def search_in_embeddings(query, stored_chunks, stored_vectors):
    q_vec = model.encode([query])
    index = faiss.IndexFlatL2(stored_vectors.shape[1])
    index.add(stored_vectors)
    D, I = index.search(q_vec, k=1)
    return stored_chunks[I[0][0]]
