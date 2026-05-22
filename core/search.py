"""
Hybrid search pipeline: dense + sparse_model + sparse_desc -> RRF -> reranker.

Entry point: search()
"""

import warnings
from typing import List, Optional

warnings.filterwarnings("ignore")

from qdrant_client.models import Prefetch, FusionQuery, Fusion

from config import PREFETCH_LIMITS, COLLECTION_NAME
from core.client import get_client
from core.filters import build_filter
from models.classifier import classify_query
from models.embeddings import encode_query
from models.reranker import rerank


def search(
    query: str,
    limit: int = 10,
    query_type: str = None,
    use_reranker: bool = False,
    rerank_top_k: int = 50,
    filter_kwargs: dict = None,
) -> List[dict]:
    """
    Run hybrid search and return a ranked list of results.

    Pipeline:
      1. Classify query -> model_number / technical / descriptive
      2. Encode query -> dense vector + two sparse BM25 vectors
      3. Three parallel Qdrant prefetch queries (dense, sparse_model, sparse_desc)
      4. RRF fusion to produce a single ranked candidate pool
      5. Optional cross-encoder re-ranking on the top rerank_top_k candidates

    Args:
        query:         Natural language or model number query.
        limit:         Number of results to return.
        query_type:    Override auto-classification ('model_number' / 'technical' / 'descriptive').
        use_reranker:  Whether to apply cross-encoder re-ranking.
        rerank_top_k:  Candidate pool size passed to the reranker.
        filter_kwargs: Dict of keyword args forwarded to build_filter().

    Returns:
        List of result dicts with keys: rank, score, id, source, internal_id,
        model_number, description, manufacturer_name, product_category,
        has_stock, total_qoh, currency, query_type.
    """
    client = get_client()

    if query_type is None:
        query_type = classify_query(query)

    limits = PREFETCH_LIMITS[query_type]
    dense_vec, sparse_model_vec, sparse_desc_vec = encode_query(query)
    qdrant_filter = build_filter(**(filter_kwargs or {}))

    prefetch = [
        Prefetch(
            query=dense_vec,
            using="dense",
            limit=limits["dense"],
            filter=qdrant_filter,
        ),
        Prefetch(
            query=sparse_model_vec,
            using="sparse_model",
            limit=limits["sparse_model"],
            filter=qdrant_filter,
        ),
        Prefetch(
            query=sparse_desc_vec,
            using="sparse_desc",
            limit=limits["sparse_desc"],
            filter=qdrant_filter,
        ),
    ]

    fetch_limit = rerank_top_k if use_reranker else limit
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=fetch_limit,
        with_payload=True,
    )

    hits = results.points

    if use_reranker and hits:
        hits = rerank(query, hits)
        hits = hits[:limit]

    return _format_results(hits, query_type)


def _format_results(hits: list, query_type: str) -> List[dict]:
    """Convert a list of Qdrant ScoredPoints to plain dicts."""
    out = []
    for rank, hit in enumerate(hits, 1):
        p = hit.payload
        out.append({
            "rank": rank,
            "score": round(hit.score, 6),
            "id": str(hit.id),
            "source": p.get("source"),
            "internal_id": p.get("internal_id", ""),
            "model_number": p.get("model_number"),
            "description": p.get("description"),
            "manufacturer_name": p.get("manufacturer_name"),
            "product_category": p.get("product_category"),
            "has_stock": p.get("has_stock"),
            "total_qoh": p.get("total_qoh"),
            "currency": p.get("currency"),
            "query_type": query_type,
        })
    return out
