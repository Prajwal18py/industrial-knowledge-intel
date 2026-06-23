"""
Embedding wrapper. Uses fastembed (ONNX runtime under the hood) instead of
sentence-transformers + torch, because for a hackathon you don't want a
3GB torch install eating your setup time or your Render/Vercel build minutes.

Model: BAAI/bge-small-en-v1.5 -> 384-dim vectors, good enough for technical
document retrieval at this scale.
"""
import numpy as np
from fastembed import TextEmbedding

_model = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    embeddings = list(model.embed(texts))
    return np.array(embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
