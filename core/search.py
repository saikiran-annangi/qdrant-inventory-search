"""
Hybrid search pipeline: dense + sparse_model + sparse_desc + taxonomy → RRF → reranker.

Public API
----------
search()                    -- used by scripts; returns plain dicts
search_with_observability() -- used by app.py; includes timings, attribution, taxonomy
"""

import time
import warnings
from typing import List

warnings.filterwarnings("ignore")

from qdrant_client.models import Prefetch, FusionQuery, Fusion, Filter, FieldCondition, MatchValue

from config import PREFETCH_LIMITS, COLLECTION_NAME, QUERY_TAXONOMY_SUBCATEGORY_THRESHOLD
from core.client import get_client
from core.filters import build_filter
from models.classifier import classify_query
from models.embeddings import encode_query
from models.query_taxonomy_llm import classify_query_taxonomy_llm
from models.reranker import rerank, rerank_with_scores
from data.normalizers import size_anchor_tokens, doc_size_anchors


# ---------------------------------------------------------------------------
# Size-aware reranking (senior's addition)
# ---------------------------------------------------------------------------

def _size_relation(hit, want: set) -> str:
    """'match' if a doc size equals a queried size, 'conflict' if it states a
    different size, 'none' if the doc states no size."""
    if not want:
        return "none"
    doc = doc_size_anchors(hit.payload.get("description"))
    if want & doc:
        return "match"
    return "conflict" if doc else "none"


def apply_size_sort(query: str, hits: list, ce_scores: dict) -> list:
    """Re-order reranked hits by size relation to the query.

    tier 2 (top)    -- doc size matches a queried size
    tier 1 (middle) -- doc states no size
    tier 0 (bottom) -- doc states a conflicting size
    CE score is the tiebreaker within each tier. No-op when query has no size.
    """
    if not hits:
        return hits
    want = size_anchor_tokens(query, bridge_metric=True)
    if not want:
        return hits

    def ce(h):
        return ce_scores.get(str(h.id), float(h.score))

    tier = {"match": 2, "none": 1, "conflict": 0}
    return sorted(hits, key=lambda h: (tier[_size_relation(h, want)], ce(h)), reverse=True)


# ---------------------------------------------------------------------------
# Taxonomy filter helper
# ---------------------------------------------------------------------------

def _taxonomy_filter(base_filter, key: str, value: str):
    """Return a filter that adds key == value on top of base_filter."""
    cond = FieldCondition(key=key, match=MatchValue(value=value))
    if base_filter is None:
        return Filter(must=[cond])
    return Filter(must=list(base_filter.must or []) + [cond])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
      1. Classify query → model_number / technical / descriptive
      2. Two-stage LLM taxonomy classifier → domain / category / subcategory
      3. Encode query → dense + sparse_model + sparse_desc vectors
      4. Four RRF prefetch channels (dense, BM25-model, BM25-desc, taxonomy)
      5. RRF fusion
      6. Optional cross-encoder reranking + size-aware sort
    """
    client = get_client()

    if query_type is None:
        query_type = classify_query(query)

    limits       = PREFETCH_LIMITS[query_type]
    dense_vec, sparse_model_vec, sparse_desc_vec = encode_query(query)
    qdrant_filter = build_filter(**(filter_kwargs or {}))

    # Taxonomy 4th channel
    tax_filter = None
    tax_result = classify_query_taxonomy_llm(query, query_type)
    if tax_result:
        tax_conf   = tax_result.get("confidence_score")
        tax_subcat = tax_result.get("taxonomy_subcategory", "")
        tax_cat    = tax_result.get("taxonomy_category",    "")
        if tax_subcat and (tax_conf is None or tax_conf >= QUERY_TAXONOMY_SUBCATEGORY_THRESHOLD):
            tax_filter = _taxonomy_filter(qdrant_filter, "taxonomy_subcategory", tax_subcat)
        elif tax_cat:
            tax_filter = _taxonomy_filter(qdrant_filter, "taxonomy_category", tax_cat)

    prefetch = []
    if limits["dense"] > 0:
        prefetch.append(Prefetch(query=dense_vec, using="dense",
                                 limit=limits["dense"], filter=qdrant_filter))
    prefetch.append(Prefetch(query=sparse_model_vec, using="sparse_model",
                             limit=limits["sparse_model"], filter=qdrant_filter))
    if limits["sparse_desc"] > 0:
        prefetch.append(Prefetch(query=sparse_desc_vec, using="sparse_desc",
                                 limit=limits["sparse_desc"], filter=qdrant_filter))
    if tax_filter is not None:
        prefetch.append(Prefetch(query=dense_vec, using="dense",
                                 limit=50, filter=tax_filter))

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
        hits, ce_scores = rerank_with_scores(query, hits)
        hits = apply_size_sort(query, hits, ce_scores)
        hits = hits[:limit]

    return _format_results(hits, query_type)


def search_with_observability(
    query: str,
    limit: int = 10,
    rerank_top_k: int = 50,
    source_filter: str = None,
) -> tuple:
    """
    Run the full search pipeline with per-step timings and retriever attribution.

    Returns:
        results          -- list of result dicts (top `limit` after reranking)
        query_type       -- classified query type string
        taxonomy_result  -- dict with taxonomy_domain, taxonomy_category,
                           taxonomy_subcategory, filter_level. {} if skipped.
        timings          -- dict of step timings in milliseconds
        retriever_counts -- candidate counts per retriever in the RRF pool
        full_pool        -- all rerank_top_k candidates with rrf_rank + rerank_rank
        channel_hits     -- per-retriever internal_id → rank (for ERP lookup)
    """
    client  = get_client()
    timings = {}

    t0 = time.perf_counter()
    query_type = classify_query(query)
    timings["classify_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    taxonomy_result = classify_query_taxonomy_llm(query, query_type)
    timings["taxonomy_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    dense_vec, sm_vec, sd_vec = encode_query(query)
    timings["encode_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    limits        = PREFETCH_LIMITS[query_type]
    qdrant_filter = build_filter(source=source_filter) if source_filter else None

    # Build taxonomy channel filter with subcategory → category fallback
    tax_filter       = None
    tax_filter_level = None
    if taxonomy_result:
        tax_conf     = taxonomy_result.get("confidence_score")
        tax_subcat   = taxonomy_result.get("taxonomy_subcategory", "")
        tax_category = taxonomy_result.get("taxonomy_category",    "")
        if tax_subcat and (tax_conf is None or tax_conf >= QUERY_TAXONOMY_SUBCATEGORY_THRESHOLD):
            tax_filter       = _taxonomy_filter(qdrant_filter, "taxonomy_subcategory", tax_subcat)
            tax_filter_level = "subcategory"
        elif tax_category:
            tax_filter       = _taxonomy_filter(qdrant_filter, "taxonomy_category", tax_category)
            tax_filter_level = "category"

    t0 = time.perf_counter()

    # Run each retriever individually to capture per-retriever scores
    dense_pts, sm_pts, sd_pts, tax_pts = [], [], [], []
    if limits["dense"] > 0:
        dense_pts = client.query_points(
            COLLECTION_NAME, query=dense_vec, using="dense",
            limit=limits["dense"], with_payload=["internal_id"],
            query_filter=qdrant_filter,
        ).points
    sm_pts = client.query_points(
        COLLECTION_NAME, query=sm_vec, using="sparse_model",
        limit=limits["sparse_model"], with_payload=["internal_id"],
        query_filter=qdrant_filter,
    ).points
    if limits["sparse_desc"] > 0:
        sd_pts = client.query_points(
            COLLECTION_NAME, query=sd_vec, using="sparse_desc",
            limit=limits["sparse_desc"], with_payload=["internal_id"],
            query_filter=qdrant_filter,
        ).points
    if tax_filter is not None:
        tax_pts = client.query_points(
            COLLECTION_NAME, query=dense_vec, using="dense",
            limit=50, with_payload=["internal_id"],
            query_filter=tax_filter,
        ).points

    def _iid_ranks(pts):
        out = {}
        for i, p in enumerate(pts, 1):
            iid = str((p.payload or {}).get("internal_id", "")).strip().lower()
            if iid and iid not in out:
                out[iid] = i
        return out

    channel_hits = {
        "dense":        _iid_ranks(dense_pts),
        "sparse_model": _iid_ranks(sm_pts),
        "sparse_desc":  _iid_ranks(sd_pts),
        "taxonomy":     _iid_ranks(tax_pts),
    }

    dense_map = {str(p.id): round(float(p.score), 4) for p in dense_pts}
    sm_map    = {str(p.id): round(float(p.score), 4) for p in sm_pts}
    sd_map    = {str(p.id): round(float(p.score), 4) for p in sd_pts}
    tax_map   = {str(p.id): round(float(p.score), 4) for p in tax_pts}

    # RRF fusion across all active channels
    prefetch = []
    if limits["dense"] > 0:
        prefetch.append(Prefetch(query=dense_vec, using="dense",
                                 limit=limits["dense"], filter=qdrant_filter))
    prefetch.append(Prefetch(query=sm_vec, using="sparse_model",
                             limit=limits["sparse_model"], filter=qdrant_filter))
    if limits["sparse_desc"] > 0:
        prefetch.append(Prefetch(query=sd_vec, using="sparse_desc",
                                 limit=limits["sparse_desc"], filter=qdrant_filter))
    if tax_filter is not None:
        prefetch.append(Prefetch(query=dense_vec, using="dense",
                                 limit=50, filter=tax_filter))

    rrf_resp = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=rerank_top_k,
        with_payload=True,
    )
    rrf_hits   = rrf_resp.points
    rrf_scores = {str(h.id): round(float(h.score), 6) for h in rrf_hits}
    rrf_rank_map = {str(h.id): i for i, h in enumerate(rrf_hits, 1)}
    timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    rrf_pool_ids = {str(h.id) for h in rrf_hits}
    retriever_counts = {
        "dense":         sum(1 for i in rrf_pool_ids if i in dense_map),
        "sparse_model":  sum(1 for i in rrf_pool_ids if i in sm_map),
        "sparse_desc":   sum(1 for i in rrf_pool_ids if i in sd_map),
        "taxonomy":      sum(1 for i in rrf_pool_ids if i in tax_map),
        "rrf_pool_size": len(rrf_hits),
    }

    t0 = time.perf_counter()
    reranker_scores: dict = {}
    hits = list(rrf_hits)
    if hits:
        hits, reranker_scores = rerank_with_scores(query, hits)
        hits = apply_size_sort(query, hits, reranker_scores)
    rerank_rank_map = {str(h.id): i for i, h in enumerate(hits, 1)}
    display_hits    = hits[:limit]
    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timings["total_ms"]  = round(sum(timings.values()), 1)

    results = []
    for rank, hit in enumerate(display_hits, 1):
        p   = hit.payload
        hid = str(hit.id)

        d_score   = dense_map.get(hid)
        sm_score  = sm_map.get(hid)
        sd_score  = sd_map.get(hid)
        tax_score = tax_map.get(hid)

        sources = []
        if d_score   is not None: sources.append("Dense")
        if sm_score  is not None: sources.append("BM25-model")
        if sd_score  is not None: sources.append("BM25-desc")
        if tax_score is not None: sources.append("Taxonomy")
        retrieval_path = " + ".join(sources) if sources else "unknown"

        results.append({
            "rank":               rank,
            "rrf_rank":           rrf_rank_map.get(hid),
            "id":                 hid,
            "reranker_score":     round(reranker_scores.get(hid, float(hit.score)), 4),
            "rrf_score":          rrf_scores.get(hid, 0.0),
            "dense_score":        round(d_score,   4) if d_score   is not None else None,
            "sparse_model_score": round(sm_score,  4) if sm_score  is not None else None,
            "sparse_desc_score":  round(sd_score,  4) if sd_score  is not None else None,
            "taxonomy_score":     round(tax_score, 4) if tax_score is not None else None,
            "retrieval_path":     retrieval_path,
            "model_number":           p.get("model_number")          or "",
            "description":            p.get("description")           or "",
            "extended_description":   p.get("extended_description"),
            "manufacturer_name":      p.get("manufacturer_name")     or "",
            "product_category":       p.get("product_category")      or "",
            "source":                 p.get("source")                or "",
            "internal_id":            p.get("internal_id")           or "",
            "has_stock":              p.get("has_stock"),
            "total_qoh":              p.get("total_qoh"),
            "min_cost":               p.get("min_cost"),
            "max_cost":               p.get("max_cost"),
            "currency":               p.get("currency")              or "",
            "locations":              p.get("locations")             or [],
            "raw_payload":            dict(p),
        })

    full_pool = []
    for hit in rrf_hits:
        hid = str(hit.id)
        p   = hit.payload
        full_pool.append({
            "rrf_rank":      rrf_rank_map[hid],
            "rerank_rank":   rerank_rank_map.get(hid),
            "internal_id":   p.get("internal_id")  or "",
            "model_number":  p.get("model_number") or "",
            "source":        p.get("source")       or "",
            "description":   str(p.get("description") or "")[:100],
            "rrf_score":     rrf_scores[hid],
            "reranker_score": round(reranker_scores.get(hid, 0.0), 4),
        })

    if taxonomy_result and tax_filter_level:
        taxonomy_result["filter_level"] = tax_filter_level

    return results, query_type, taxonomy_result, timings, retriever_counts, full_pool, channel_hits


def _format_results(hits: list, query_type: str) -> List[dict]:
    out = []
    for rank, hit in enumerate(hits, 1):
        p = hit.payload
        out.append({
            "rank":              rank,
            "score":             round(hit.score, 6),
            "id":                str(hit.id),
            "source":            p.get("source"),
            "internal_id":       p.get("internal_id", ""),
            "model_number":      p.get("model_number"),
            "description":       p.get("description"),
            "manufacturer_name": p.get("manufacturer_name"),
            "product_category":  p.get("product_category"),
            "has_stock":         p.get("has_stock"),
            "total_qoh":         p.get("total_qoh"),
            "currency":          p.get("currency"),
            "query_type":        query_type,
        })
    return out
