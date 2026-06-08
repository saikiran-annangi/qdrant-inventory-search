"""
Hybrid search pipeline: dense + sparse_model + sparse_desc → RRF → reranker
→ taxonomy boost → size sort → attribute sort.

Public API
----------
search()                    -- used by evaluate.py and scripts; returns plain dicts
search_with_observability() -- used by app.py; includes per-step timings,
                               per-retriever attribution, and taxonomy prediction
"""

import math
import time
import warnings
from typing import List

warnings.filterwarnings("ignore")

from qdrant_client.models import (
    Prefetch, FusionQuery, Fusion, SearchParams, QuantizationSearchParams,
)

from config import PREFETCH_LIMITS, COLLECTION_NAME, USE_CLASSIFIER, DEFAULT_PROFILE
from core.client import get_client
from core.filters import build_filter
from models.classifier import classify_query
from models.embeddings import encode_query
from models.query_taxonomy_llm import classify_query_taxonomy_llm
from models.reranker import rerank_with_scores
from data.normalizers import (
    size_anchor_tokens, doc_size_anchors,
    attribute_anchor_tokens, attribute_relation,
    clean_bom_query,
)

# Dense vectors are int8-quantized (see scripts/ingest.py). Rescore re-scores
# the quantized candidate pool against on-disk float32 originals, recovering
# the precision lost to quantization. oversampling=2.0 widens that pool first.
# Applied to dense reads ONLY — BM25 sparse channels are not quantized.
_DENSE_QSP = SearchParams(
    quantization=QuantizationSearchParams(rescore=True, oversampling=2.0)
)

# Score bonus added to a hit's cross-encoder logit when its taxonomy_subcategory
# matches the query's predicted taxonomy. Large enough to swap ranks within ~1
# CE logit of each other; not so large it overrides a strong CE preference.
TAXONOMY_BOOST = 0.8


# ---------------------------------------------------------------------------
# Post-rerank passes (applied in order: taxonomy → size → attribute)
# ---------------------------------------------------------------------------

def _norm_label(s: str) -> str:
    """Normalize a taxonomy label for matching: casefold + collapse whitespace.

    Query-predicted labels and product payload labels both come from the same
    controlled vocabulary (data/taxonomy.py), so they should match exactly — but
    normalizing makes the match robust to incidental casing/whitespace drift so
    a real match can never be missed on a cosmetic difference.
    """
    return " ".join((s or "").strip().casefold().split())


def apply_taxonomy_boost(hits: list, ce_scores: dict, tax_result: dict) -> tuple:
    """
    Soft score bonus for items whose taxonomy matches the predicted label.

    Taxonomy is a ranking signal only — items are never excluded, so recall is
    preserved regardless of prediction accuracy. A wrong prediction just fails
    to boost rather than injecting noise.

    - Subcategory match: +TAXONOMY_BOOST (0.8)
    - Category-only match (no subcategory predicted): +TAXONOMY_BOOST * 0.25
    - No match: no change

    Returns (sorted_hits, updated_scores, boosted_ids_set).
    """
    if not tax_result or not hits:
        return hits, ce_scores, set()

    tax_subcat = _norm_label(tax_result.get("taxonomy_subcategory", ""))
    tax_cat    = _norm_label(tax_result.get("taxonomy_category",    ""))

    if not tax_subcat and not tax_cat:
        return hits, ce_scores, set()

    boosted     = {}
    boosted_ids = set()

    for hit in hits:
        hid = str(hit.id)
        raw = ce_scores.get(hid)
        # Fall back to RRF score when CE score is nan (model numerical instability)
        base = float(hit.score) if (raw is None or (isinstance(raw, float) and math.isnan(raw))) else raw

        payload_subcat = _norm_label(hit.payload.get("taxonomy_subcategory", ""))
        payload_cat    = _norm_label(hit.payload.get("taxonomy_category",    ""))

        if tax_subcat and payload_subcat == tax_subcat:
            boosted[hid] = base + TAXONOMY_BOOST
            boosted_ids.add(hid)
        elif not tax_subcat and tax_cat and payload_cat == tax_cat:
            # Category-level boost fires only when subcategory prediction absent.
            # Kept light to avoid false positives on vague descriptive queries.
            boosted[hid] = base + TAXONOMY_BOOST * 0.25
            boosted_ids.add(hid)
        else:
            boosted[hid] = base

    sorted_hits = sorted(
        hits,
        key=lambda h: boosted.get(str(h.id), float(h.score)),
        reverse=True,
    )
    return sorted_hits, boosted, boosted_ids


def apply_size_sort(query: str, hits: list, ce_scores: dict) -> list:
    """Re-order reranked hits by their size relation to the query.

    The cross-encoder is size-blind — this tiered sort restores size intent:
      tier 2 (top)    -- doc size matches a queried size
      tier 1 (middle) -- doc states no size (silent on the attribute)
      tier 0 (bottom) -- doc states a size and none matches
    CE score is the tiebreaker within each tier. No-op when the query carries
    no size anchor, so it is safe to leave on for every query.
    """
    if not hits:
        return hits
    want = size_anchor_tokens(query, bridge_metric=True)
    if not want:
        return hits

    def _size_relation(hit):
        if not want:
            return "none"
        doc = doc_size_anchors(hit.payload.get("description"))
        if want & doc:
            return "match"
        return "conflict" if doc else "none"

    tier = {"match": 2, "none": 1, "conflict": 0}

    def ce(h):
        return ce_scores.get(str(h.id), float(h.score))

    return sorted(hits, key=lambda h: (tier[_size_relation(h)], ce(h)), reverse=True)


def apply_attribute_sort(query: str, hits: list) -> list:
    """Re-order hits by structured electrical attributes (pole, amp, volt,
    trip curve, NEMA class, IP rating, lamp base, tamper-resistant, knock-out).

    The cross-encoder is attribute-blind — a 15A part can outscore a 20A one.
    This sort re-tiers by (attribute matches desc, conflicts asc): a doc that
    contradicts a queried attribute sinks BELOW one that is merely silent on it.
    Python's stable sort preserves the incoming CE/taxonomy-order as tiebreaker.
    No-op when the query states no recognised attribute. Call AFTER size sort.
    """
    if not hits:
        return hits
    want = attribute_anchor_tokens(query)
    if not want:
        return hits

    def key(h):
        doc = attribute_anchor_tokens(
            " ".join(str(h.payload.get(f) or "") for f in
                     ("description", "extended_description", "product_category"))
        )
        matches, conflicts = attribute_relation(want, doc)
        return (matches, -conflicts)

    return sorted(hits, key=key, reverse=True)


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
      1. Classify query → model_number / technical / descriptive / default
      2. Two-stage LLM taxonomy classifier → domain / category / subcategory
         (skipped for model_number queries; boost disabled for descriptive)
      3. Encode query → dense + sparse_model + sparse_desc vectors
      4. Three parallel Qdrant prefetches (limits per query_type)
      5. RRF fusion → top rerank_top_k candidates
      6. Cross-encoder rerank
      7. Taxonomy soft boost (+0.8 CE logit for subcategory match)
      8. Size-aware sort (exact size > silent > conflicting)
      9. Electrical attribute sort (more matches > fewer conflicts)
    """
    client = get_client()
    query  = clean_bom_query(query)

    if query_type is None:
        query_type = classify_query(query) if USE_CLASSIFIER else DEFAULT_PROFILE

    limits        = PREFETCH_LIMITS[query_type]
    dense_vec, sparse_model_vec, sparse_desc_vec = encode_query(query)
    qdrant_filter = build_filter(**(filter_kwargs or {}))

    # Taxonomy: used as a post-rerank score nudge. Skipped for model_number
    # (query_taxonomy_llm returns {} for those). Also disabled for descriptive
    # because vague queries map unreliably — a wrong prediction hurts more than
    # a right one helps for short freeform queries.
    tax_result = classify_query_taxonomy_llm(query, query_type)
    if query_type == "descriptive":
        tax_result = {}

    prefetch = []
    if limits["dense"] > 0:
        prefetch.append(Prefetch(
            query=dense_vec, using="dense",
            limit=limits["dense"], filter=qdrant_filter, params=_DENSE_QSP,
        ))
    prefetch.append(Prefetch(
        query=sparse_model_vec, using="sparse_model",
        limit=limits["sparse_model"], filter=qdrant_filter,
    ))
    if limits["sparse_desc"] > 0:
        prefetch.append(Prefetch(
            query=sparse_desc_vec, using="sparse_desc",
            limit=limits["sparse_desc"], filter=qdrant_filter,
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
        hits, ce_scores = rerank_with_scores(query, hits)
        hits, ce_scores, _ = apply_taxonomy_boost(hits, ce_scores, tax_result)
        hits = apply_size_sort(query, hits, ce_scores)
        hits = apply_attribute_sort(query, hits)
        hits = hits[:limit]

    return _format_results(hits, query_type)


def search_with_observability(
    query: str,
    limit: int = 10,
    rerank_top_k: int = 50,
    source_filter: str = None,
) -> tuple:
    """
    Run the full search pipeline and return results with per-step timings,
    per-retriever attribution, and taxonomy prediction.

    Returns (7-tuple):
        results          -- list of result dicts (top `limit` after all passes)
        query_type       -- classified query type string
        taxonomy_result  -- dict with taxonomy_domain, taxonomy_category,
                            taxonomy_subcategory. {} when skipped.
        timings          -- dict of step timings in milliseconds
        retriever_counts -- candidate counts per retriever + taxonomy boost count
        full_pool        -- all rerank_top_k candidates with rrf_rank + rerank_rank
        channel_hits     -- per-retriever {internal_id: rank} for ERP lookup
    """
    client  = get_client()
    query   = clean_bom_query(query)
    timings = {}

    t0 = time.perf_counter()
    query_type = classify_query(query) if USE_CLASSIFIER else DEFAULT_PROFILE
    timings["classify_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    taxonomy_result = classify_query_taxonomy_llm(query, query_type)
    if query_type == "descriptive":
        taxonomy_result = {}
    timings["taxonomy_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    dense_vec, sm_vec, sd_vec = encode_query(query)
    timings["encode_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    limits        = PREFETCH_LIMITS[query_type]
    qdrant_filter = build_filter(source=source_filter) if source_filter else None

    t0 = time.perf_counter()

    # Run each retriever individually to capture per-retriever scores for
    # attribution display. Channels with limit=0 are skipped entirely.
    dense_pts, sm_pts, sd_pts = [], [], []
    if limits["dense"] > 0:
        dense_pts = client.query_points(
            COLLECTION_NAME, query=dense_vec, using="dense",
            limit=limits["dense"], with_payload=["internal_id"],
            query_filter=qdrant_filter, search_params=_DENSE_QSP,
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
    }

    dense_map = {str(p.id): round(float(p.score), 4) for p in dense_pts}
    sm_map    = {str(p.id): round(float(p.score), 4) for p in sm_pts}
    sd_map    = {str(p.id): round(float(p.score), 4) for p in sd_pts}

    prefetch = []
    if limits["dense"] > 0:
        prefetch.append(Prefetch(query=dense_vec, using="dense",        limit=limits["dense"],        filter=qdrant_filter, params=_DENSE_QSP))
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
    rrf_hits     = rrf_resp.points
    rrf_scores   = {str(h.id): round(float(h.score), 6) for h in rrf_hits}
    rrf_rank_map = {str(h.id): i for i, h in enumerate(rrf_hits, 1)}
    timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    rrf_pool_ids = {str(h.id) for h in rrf_hits}

    t0 = time.perf_counter()
    reranker_scores: dict = {}
    boosted_ids:    set  = set()
    hits = list(rrf_hits)
    if hits:
        hits, reranker_scores = rerank_with_scores(query, hits)
        hits, reranker_scores, boosted_ids = apply_taxonomy_boost(hits, reranker_scores, taxonomy_result)
        hits = apply_size_sort(query, hits, reranker_scores)
        hits = apply_attribute_sort(query, hits)
    rerank_rank_map = {str(h.id): i for i, h in enumerate(hits, 1)}
    display_hits    = hits[:limit]
    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timings["total_ms"]  = round(sum(timings.values()), 1)

    retriever_counts = {
        "dense":            sum(1 for i in rrf_pool_ids if i in dense_map),
        "sparse_model":     sum(1 for i in rrf_pool_ids if i in sm_map),
        "sparse_desc":      sum(1 for i in rrf_pool_ids if i in sd_map),
        "taxonomy_boosted": len(boosted_ids),
        "rrf_pool_size":    len(rrf_hits),
    }

    results = []
    for rank, hit in enumerate(display_hits, 1):
        p   = hit.payload
        hid = str(hit.id)

        d_score      = dense_map.get(hid)
        sm_score     = sm_map.get(hid)
        sd_score     = sd_map.get(hid)
        tax_boosted  = hid in boosted_ids

        sources = []
        if d_score  is not None: sources.append("Dense")
        if sm_score is not None: sources.append("BM25-model")
        if sd_score is not None: sources.append("BM25-desc")
        if tax_boosted:          sources.append("TaxBoost")
        retrieval_path = " + ".join(sources) if sources else "unknown"

        results.append({
            "rank":               rank,
            "rrf_rank":           rrf_rank_map.get(hid),
            "id":                 hid,
            "reranker_score":     round(reranker_scores.get(hid, float(hit.score)), 4),
            "rrf_score":          rrf_scores.get(hid, 0.0),
            "dense_score":        round(d_score,  4) if d_score  is not None else None,
            "sparse_model_score": round(sm_score, 4) if sm_score is not None else None,
            "sparse_desc_score":  round(sd_score, 4) if sd_score is not None else None,
            "taxonomy_boosted":   tax_boosted,
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
            "rrf_rank":       rrf_rank_map[hid],
            "rerank_rank":    rerank_rank_map.get(hid),
            "internal_id":    p.get("internal_id")   or "",
            "model_number":   p.get("model_number")  or "",
            "source":         p.get("source")        or "",
            "description":    str(p.get("description") or "")[:100],
            "rrf_score":      rrf_scores[hid],
            "reranker_score": round(reranker_scores.get(hid, 0.0), 4),
        })

    return results, query_type, taxonomy_result, timings, retriever_counts, full_pool, channel_hits


def _format_results(hits: list, query_type: str) -> List[dict]:
    """Convert a list of Qdrant ScoredPoints to plain dicts."""
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
