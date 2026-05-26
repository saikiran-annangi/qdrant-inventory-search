"""
Hybrid search pipeline: dense + sparse_model + sparse_desc -> RRF -> reranker.

Public API
----------
search()                    -- used by evaluate.py and scripts; returns plain dicts
search_with_observability() -- used by app.py; includes per-step timings and
                               per-retriever attribution
"""

import time
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore")

from qdrant_client.models import Prefetch, FusionQuery, Fusion

from config import PREFETCH_LIMITS, COLLECTION_NAME
from core.client import get_client
from core.filters import build_filter
from models.classifier import classify_query
from models.embeddings import encode_query
from models.reranker import rerank, rerank_with_scores


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

    prefetch = []
    if limits["dense"] > 0:
        prefetch.append(Prefetch(
            query=dense_vec,
            using="dense",
            limit=limits["dense"],
            filter=qdrant_filter,
        ))
    prefetch.append(Prefetch(
        query=sparse_model_vec,
        using="sparse_model",
        limit=limits["sparse_model"],
        filter=qdrant_filter,
    ))
    if limits["sparse_desc"] > 0:
        prefetch.append(Prefetch(
            query=sparse_desc_vec,
            using="sparse_desc",
            limit=limits["sparse_desc"],
            filter=qdrant_filter,
        ))

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


def search_with_observability(
    query: str,
    limit: int = 3,
    rerank_top_k: int = 50,
    source_filter: str = None,
) -> tuple:
    """
    Run the full search pipeline and return results with per-step timings
    and per-retriever attribution.

    Returns:
        results          -- list of result dicts (see keys below)
        query_type       -- classified query type string
        timings          -- dict of step timings in milliseconds:
                           classify_ms, encode_ms, retrieve_ms, rerank_ms, total_ms
        retriever_counts -- dict with candidate counts per retriever in the RRF pool:
                           dense, sparse_model, sparse_desc, rrf_pool_size

    Result dict keys:
        rank, id, reranker_score, rrf_score,
        dense_score, sparse_model_score, sparse_desc_score, retrieval_path,
        model_number, description, manufacturer_name, product_category,
        source, internal_id, has_stock, total_qoh, min_cost, max_cost,
        currency, locations, raw_payload
    """
    client = get_client()
    timings = {}

    t0 = time.perf_counter()
    query_type = classify_query(query)
    timings["classify_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    dense_vec, sm_vec, sd_vec = encode_query(query)
    timings["encode_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    limits = PREFETCH_LIMITS[query_type]
    qdrant_filter = build_filter(source=source_filter) if source_filter else None

    t0 = time.perf_counter()

    # Run each active retriever individually to capture per-retriever scores
    # for the attribution display. Channels with limit=0 are skipped.
    dense_pts, sm_pts, sd_pts = [], [], []
    if limits["dense"] > 0:
        dense_pts = client.query_points(
            COLLECTION_NAME, query=dense_vec, using="dense",
            limit=limits["dense"], with_payload=False, query_filter=qdrant_filter,
        ).points
    sm_pts = client.query_points(
        COLLECTION_NAME, query=sm_vec, using="sparse_model",
        limit=limits["sparse_model"], with_payload=False, query_filter=qdrant_filter,
    ).points
    if limits["sparse_desc"] > 0:
        sd_pts = client.query_points(
            COLLECTION_NAME, query=sd_vec, using="sparse_desc",
            limit=limits["sparse_desc"], with_payload=False, query_filter=qdrant_filter,
        ).points

    dense_map = {str(p.id): round(float(p.score), 4) for p in dense_pts}
    sm_map    = {str(p.id): round(float(p.score), 4) for p in sm_pts}
    sd_map    = {str(p.id): round(float(p.score), 4) for p in sd_pts}

    # RRF fusion — skip channels whose limit is 0
    prefetch = []
    if limits["dense"] > 0:
        prefetch.append(Prefetch(query=dense_vec, using="dense",        limit=limits["dense"],        filter=qdrant_filter))
    prefetch.append(    Prefetch(query=sm_vec,    using="sparse_model", limit=limits["sparse_model"], filter=qdrant_filter))
    if limits["sparse_desc"] > 0:
        prefetch.append(Prefetch(query=sd_vec,    using="sparse_desc",  limit=limits["sparse_desc"],  filter=qdrant_filter))

    rrf_resp = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=rerank_top_k,
        with_payload=True,
    )
    hits = rrf_resp.points
    rrf_scores = {str(h.id): round(float(h.score), 6) for h in hits}
    timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    rrf_pool_ids = {str(h.id) for h in hits}
    retriever_counts = {
        "dense":         sum(1 for i in rrf_pool_ids if i in dense_map),
        "sparse_model":  sum(1 for i in rrf_pool_ids if i in sm_map),
        "sparse_desc":   sum(1 for i in rrf_pool_ids if i in sd_map),
        "rrf_pool_size": len(hits),
    }

    t0 = time.perf_counter()
    reranker_scores: dict = {}
    if hits:
        hits, reranker_scores = rerank_with_scores(query, hits)
        hits = hits[:limit]
    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timings["total_ms"]  = round(sum(timings.values()), 1)

    results = []
    for rank, hit in enumerate(hits, 1):
        p   = hit.payload
        hid = str(hit.id)

        d_score  = dense_map.get(hid)
        sm_score = sm_map.get(hid)
        sd_score = sd_map.get(hid)

        sources = []
        if d_score  is not None: sources.append("Dense")
        if sm_score is not None: sources.append("BM25-model")
        if sd_score is not None: sources.append("BM25-desc")
        retrieval_path = " + ".join(sources) if sources else "unknown"

        results.append({
            "rank":               rank,
            "id":                 hid,
            "reranker_score":     round(reranker_scores.get(hid, float(hit.score)), 4),
            "rrf_score":          rrf_scores.get(hid, 0.0),
            "dense_score":        round(d_score,  4) if d_score  is not None else None,
            "sparse_model_score": round(sm_score, 4) if sm_score is not None else None,
            "sparse_desc_score":  round(sd_score, 4) if sd_score is not None else None,
            "retrieval_path":     retrieval_path,
            "model_number":       p.get("model_number")      or "",
            "description":        p.get("description")       or "",
            "manufacturer_name":  p.get("manufacturer_name") or "",
            "product_category":   p.get("product_category")  or "",
            "source":             p.get("source")            or "",
            "internal_id":        p.get("internal_id")       or "",
            "has_stock":          p.get("has_stock"),
            "total_qoh":          p.get("total_qoh"),
            "min_cost":           p.get("min_cost"),
            "max_cost":           p.get("max_cost"),
            "currency":           p.get("currency")          or "",
            "locations":          p.get("locations")         or [],
            "raw_payload":        dict(p),
        })

    return results, query_type, timings, retriever_counts


def _format_results(hits: list, query_type: str) -> List[dict]:
    """Convert a list of Qdrant ScoredPoints to plain dicts."""
    out = []
    for rank, hit in enumerate(hits, 1):
        p = hit.payload
        out.append({
            "rank":             rank,
            "score":            round(hit.score, 6),
            "id":               str(hit.id),
            "source":           p.get("source"),
            "internal_id":      p.get("internal_id", ""),
            "model_number":     p.get("model_number"),
            "description":      p.get("description"),
            "manufacturer_name": p.get("manufacturer_name"),
            "product_category": p.get("product_category"),
            "has_stock":        p.get("has_stock"),
            "total_qoh":        p.get("total_qoh"),
            "currency":         p.get("currency"),
            "query_type":       query_type,
        })
    return out
