from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Use the same model as main.py for consistency
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# all-mpnet-base-v2 produces 768-dimensional embeddings
embedding_dim = 768

# Create FAISS index
index = faiss.IndexFlatL2(embedding_dim)

# Store texts and embeddings
corpus_texts = []
corpus_embeddings = []

def add_document(text: str):
    """
    Add a document text to the FAISS index and internal corpus.
    """
    embedding = model.encode([text], convert_to_numpy=True)
    corpus_texts.append(text)
    corpus_embeddings.append(embedding)
    index.add(embedding)
    return {"status": "added", "text": text[:50] + "..."}

def search(query: str, top_k: int = 3):
    """
    Search for the top_k most similar texts to the query.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [corpus_texts[i] for i in indices[0] if i < len(corpus_texts)]
    return results
