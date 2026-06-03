"""
Cross-encoder re-ranker.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
       6-layer MiniLM, ~22M params, runs on CPU in ~90ms for 50 candidates.

Uses transformers directly (not sentence-transformers) with float64 weights to
avoid float32 numerical instability on torch 2.12+ / transformers 5.x.
"""

import warnings

warnings.filterwarnings("ignore")

from config import RERANKER_MODEL_NAME

_model     = None
_tokenizer = None


def _get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME,
            torch_dtype=torch.float64,
        )
        _model.eval()
    return _model, _tokenizer


def get_reranker():
    """Warm up the reranker (backward-compat shim for scripts that call get_reranker())."""
    _get_model_and_tokenizer()


def rerank(query: str, hits: list) -> list:
    sorted_hits, _ = rerank_with_scores(query, hits)
    return sorted_hits


def rerank_with_scores(query: str, hits: list) -> tuple:
    """
    Re-rank hits and return both the sorted hits and a scores dict.

    Returns:
        sorted_hits   -- list of ScoredPoint, highest CrossEncoder score first
        scores_by_id  -- dict mapping str(hit.id) -> float CrossEncoder logit
                         Range is typically [-12, +12]; higher = more relevant.
    """
    import torch

    model, tokenizer = _get_model_and_tokenizer()

    docs = []
    for hit in hits:
        doc_text = " ".join(filter(None, [
            hit.payload.get("description", ""),
            hit.payload.get("extended_description") or "",
            hit.payload.get("manufacturer_name", ""),
            hit.payload.get("model_number", ""),
            hit.payload.get("product_category", ""),
        ]))
        docs.append(doc_text)

    queries = [query] * len(hits)

    with torch.no_grad():
        enc = tokenizer(
            queries, docs,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        logits = model(**enc).logits.squeeze(-1)
        scores_list = logits.tolist()
        if isinstance(scores_list, float):
            scores_list = [scores_list]

    scores_by_id = {str(hit.id): float(s) for hit, s in zip(hits, scores_list)}
    ranked = sorted(zip(scores_list, hits), key=lambda x: x[0], reverse=True)
    return [h for _, h in ranked], scores_by_id


def score_pairs(pairs: list) -> list:
    """
    Score a list of (query, document) text pairs and return raw logit scores.

    Used by offline taxonomy-building scripts (build_taxonomy_from_descriptions.py)
    that need cross-encoder scoring outside the normal search pipeline.

    Args:
        pairs: list of (query_text, doc_text) tuples

    Returns:
        list of float logit scores, same order as input pairs
    """
    import torch
    import numpy as np

    if not pairs:
        return []

    model, tokenizer = _get_model_and_tokenizer()
    queries = [p[0] for p in pairs]
    docs    = [p[1] for p in pairs]

    with torch.no_grad():
        enc = tokenizer(
            queries, docs,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        logits = model(**enc).logits.squeeze(-1)
        scores = logits.tolist()
        if isinstance(scores, float):
            scores = [scores]

    return [float(s) for s in scores]
