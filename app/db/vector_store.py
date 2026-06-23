"""
Minimal vector index. No external vector DB needed for the demo.

We store embeddings as a single numpy array on disk plus a parallel list of
chunk_ids. For a hackathon-scale corpus (tens to low hundreds of chunks)
brute-force cosine similarity is plenty fast and has zero infra cost.
"""
import numpy as np
from pathlib import Path
import pickle

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
VEC_PATH = DATA_DIR / "vectors.npy"
IDS_PATH = DATA_DIR / "chunk_ids.pkl"


def _load():
    if VEC_PATH.exists() and IDS_PATH.exists():
        vectors = np.load(VEC_PATH)
        with open(IDS_PATH, "rb") as f:
            chunk_ids = pickle.load(f)
        return vectors, chunk_ids
    return np.zeros((0, 384), dtype=np.float32), []


def _save(vectors: np.ndarray, chunk_ids: list[int]):
    np.save(VEC_PATH, vectors)
    with open(IDS_PATH, "wb") as f:
        pickle.dump(chunk_ids, f)


def add_vectors(new_vectors: np.ndarray, new_chunk_ids: list[int]):
    vectors, chunk_ids = _load()
    vectors = np.vstack([vectors, new_vectors]) if vectors.shape[0] else new_vectors
    chunk_ids = chunk_ids + new_chunk_ids
    _save(vectors, chunk_ids)


def search(query_vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
    vectors, chunk_ids = _load()
    if vectors.shape[0] == 0:
        return []

    # cosine similarity (vectors from fastembed are already normalised, but
    # we normalise again defensively)
    norm_v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    norm_q = query_vector / (np.linalg.norm(query_vector) + 1e-10)
    scores = norm_v @ norm_q

    top_idx = np.argsort(-scores)[:top_k]
    return [(chunk_ids[i], float(scores[i])) for i in top_idx]
