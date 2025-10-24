from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

embedding_dim = 384  # for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(embedding_dim)
corpus_texts = []
corpus_embeddings = []

def add_document(text):
    embedding = model.encode([text])
    corpus_texts.append(text)
    corpus_embeddings.append(embedding)
    index.add(embedding)

def search(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [corpus_texts[i] for i in indices[0] if i < len(corpus_texts)]
    return results
