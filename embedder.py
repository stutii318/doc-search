import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from config import EMBEDDINGS_DIR, EMBEDDING_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def build_index(chunks: list):
    model = get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, f"{EMBEDDINGS_DIR}/index.faiss")
    with open(f"{EMBEDDINGS_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Indexed {len(chunks)} chunks")
    return index, chunks

def load_index():
    index = faiss.read_index(f"{EMBEDDINGS_DIR}/index.faiss")
    with open(f"{EMBEDDINGS_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(query: str, index, chunks, top_k=5) -> list:
    model = get_model()
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append({**chunks[idx], "score": round(float(score), 3)})
    return results