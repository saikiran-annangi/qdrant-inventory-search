"""
Cross-encoder re-ranker.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
       6-layer MiniLM, ~22M params, runs on CPU in ~90ms for 50 candidates.

The reranker is a singleton loaded on first call.
"""

import warnings

warnings.filterwarnings("ignore")

from config import RERANKER_MODEL_NAME

_reranker = None


def get_reranker():
    """Return the CrossEncoder model, loading it on first call."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
    return _reranker


def rerank(query: str, hits: list) -> list:
    """
    Re-rank a list of Qdrant ScoredPoint objects using the cross-encoder.

    Concatenates description, manufacturer, model number, and category
    into a single document string for each candidate, then scores
    (query, document) pairs and returns hits sorted by score descending.
    """
    reranker = get_reranker()

    pairs = []
    for hit in hits:
        doc_text = " ".join(filter(None, [
            hit.payload.get("description", ""),
            hit.payload.get("manufacturer_name", ""),
            hit.payload.get("model_number", ""),
            hit.payload.get("product_category", ""),
        ]))
        pairs.append((query, doc_text))

    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    return [h for _, h in ranked]
