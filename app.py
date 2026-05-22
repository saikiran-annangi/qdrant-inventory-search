"""
Inventory Search -- Streamlit UI

Single search box with distributor filter and full pipeline observability.

Run:
    streamlit run app.py
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure the repo root is in sys.path when Streamlit changes cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from qdrant_client.models import Prefetch, FusionQuery, Fusion

from config import PREFETCH_LIMITS, COLLECTION_NAME
from core.client import get_client
from core.filters import build_filter
from models.classifier import classify_query, CLASSIFY_PROMPT
from models.embeddings import get_dense_model, get_bm25_model, encode_query
from models.reranker import get_reranker, rerank

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Inventory Search",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 1100px; }

    .stTextInput > div > div > input {
        font-size: 1rem;
        padding: 0.6rem 0.9rem;
        border-radius: 6px;
    }

    div[data-testid="column"]:last-child .stButton { margin-top: 0.15rem; }
    div[data-testid="column"]:last-child .stButton > button {
        width: 100%;
        padding: 0.55rem 1.2rem;
        font-size: 1rem;
        border-radius: 6px;
    }

    [data-testid="stMetricLabel"] { font-size: 0.72rem; color: #888; }
    [data-testid="stMetricValue"] { font-size: 1rem; font-weight: 600; }

    .result-header { font-size: 1rem; font-weight: 600; color: #1a1a1a; }
    hr { margin: 0.4rem 0; border-color: #eeeeee; }
    .meta { font-size: 0.8rem; color: #666; }
    .stock-in  { color: #2e7d32; font-weight: 600; font-size: 0.82rem; }
    .stock-out { color: #c62828; font-weight: 600; font-size: 0.82rem; }
    .stock-unk { color: #9e9e9e; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_models():
    get_dense_model()
    get_bm25_model()
    get_reranker()
    return True


# ---------------------------------------------------------------------------
# Search with full pipeline observability
# ---------------------------------------------------------------------------

def search_with_observability(
    query: str,
    limit: int = 5,
    rerank_top_k: int = 50,
    source_filter: str = None,
) -> tuple:
    """
    Run the full search pipeline and return results with timing and
    per-retriever attribution.

    Returns:
        results          -- list of result dicts
        query_type       -- classified query type string
        timings          -- dict of step timing in milliseconds
        retriever_counts -- dict with candidate counts per retriever
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

    # Run active retrievers individually for per-retriever attribution.
    # Channels with limit=0 are skipped (e.g. sparse_desc=0 for model_number queries).
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

    # RRF fusion -- skip channels whose limit is 0
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
    if hits:
        hits = rerank(query, hits)
        hits = hits[:limit]
    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timings["total_ms"]  = round(sum(timings.values()), 1)

    results = []
    for rank, hit in enumerate(hits, 1):
        p   = hit.payload
        hid = str(hit.id)
        results.append({
            "rank":              rank,
            "id":                hid,
            "reranker_score":    round(float(hit.score), 4),
            "rrf_score":         rrf_scores.get(hid, 0.0),
            "model_number":      p.get("model_number")    or "",
            "description":       p.get("description")     or "",
            "manufacturer_name": p.get("manufacturer_name") or "",
            "product_category":  p.get("product_category")  or "",
            "source":            p.get("source")          or "",
            "internal_id":       p.get("internal_id")     or "",
            "has_stock":         p.get("has_stock"),
            "total_qoh":         p.get("total_qoh"),
            "min_cost":          p.get("min_cost"),
            "max_cost":          p.get("max_cost"),
            "currency":          p.get("currency")        or "",
            "locations":         p.get("locations")       or [],
            "raw_payload":       dict(p),
        })

    return results, query_type, timings, retriever_counts


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def stock_label(has_stock) -> str:
    if has_stock is True:
        return '<span class="stock-in">In Stock</span>'
    if has_stock is False:
        return '<span class="stock-out">Out of Stock</span>'
    return '<span class="stock-unk">Unknown</span>'


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

with st.spinner("Loading models..."):
    load_models()

st.title("Inventory Search")

SOURCES = {
    "All distributors":  None,
    "Guillevin (1)":     "guillevin_1",
    "Guillevin (2)":     "guillevin_2",
    "AU Parspec":        "au_parspec",
    "Burnaby DC":        "burnaby_dc",
    "Standard Supply":   "standard_supply",
    "Inventory Sample":  "inventory_sample",
    "Plumbing":          "plumbing",
}

col_input, col_filter, col_btn = st.columns([6, 2, 1])
with col_input:
    query = st.text_input(
        label="Search",
        placeholder="Model number, spec, or description -- e.g. 'Schneider 16A single pole MCB'  or  'K-2084'",
        label_visibility="collapsed",
    )
with col_filter:
    source_label = st.selectbox("Distributor", options=list(SOURCES.keys()), label_visibility="collapsed")
    source_filter = SOURCES[source_label]
with col_btn:
    search_clicked = st.button("Search", use_container_width=True)

if not query or not query.strip():
    st.stop()

# Run only when the button is clicked or the query changes
if not search_clicked and "last_query" in st.session_state and st.session_state.last_query == query:
    st.stop()

st.session_state.last_query = query

# Run search
with st.spinner("Searching..."):
    results, query_type, timings, ret_counts = search_with_observability(
        query.strip(), source_filter=source_filter
    )

# ---------------------------------------------------------------------------
# Pipeline observability
# ---------------------------------------------------------------------------

st.divider()
st.markdown("**Pipeline**")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Query type",  query_type.replace("_", " ").title())
c2.metric("Classify",    f"{timings['classify_ms']} ms")
with c2:
    with st.popover("View prompt", use_container_width=False):
        st.code(CLASSIFY_PROMPT.format(query=query.strip()), language=None)
c3.metric("Encode",   f"{timings['encode_ms']} ms")
c4.metric("Retrieve", f"{timings['retrieve_ms']} ms")
c5.metric("Rerank",   f"{timings['rerank_ms']} ms")
c6.metric("Total",    f"{timings['total_ms']} ms")

pool = ret_counts["rrf_pool_size"]
st.markdown(
    f"**Retriever contribution** -- candidates each retriever passed into the "
    f"{pool}-candidate RRF pool (can overlap)"
)
rc1, rc2, rc3 = st.columns(3)
rc1.metric("Dense",          ret_counts["dense"],        help="Candidates from semantic embedding retriever")
rc2.metric("Sparse (model)", ret_counts["sparse_model"], help="Candidates from BM25 over model number variants")
rc3.metric("Sparse (desc)",  ret_counts["sparse_desc"],  help="Candidates from BM25 over description and specs")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

st.divider()

for r in results:
    header = f"#{r['rank']}  |  {r['model_number']}  |  {r['manufacturer_name']}"

    with st.expander(header, expanded=True):
        left, right = st.columns([3, 1])

        with left:
            st.markdown("**Description**")
            st.write(r["description"])
            st.markdown(
                f'<span class="meta">'
                f'Category: {r["product_category"]}&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'Source: {r["source"]}&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'Internal ID: {r["internal_id"]}'
                f'</span>',
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(stock_label(r["has_stock"]), unsafe_allow_html=True)
            if r["total_qoh"] is not None:
                st.metric("Total QOH", r["total_qoh"])
            if r["min_cost"] is not None:
                curr = r["currency"]
                st.markdown(
                    f'<span class="meta">Cost: {curr} {r["min_cost"]:.2f} '
                    f'- {r["max_cost"]:.2f}</span>',
                    unsafe_allow_html=True,
                )

        if r["locations"]:
            st.markdown("**Branch inventory**")
            loc_rows = []
            for loc in r["locations"]:
                cost = loc.get("cost")
                sell = loc.get("sell_price")
                curr = r["currency"]
                loc_rows.append({
                    "Branch":     loc.get("location_name", ""),
                    "ERP ID":     loc.get("location_erp_id", ""),
                    "In Stock":   "Yes" if loc.get("in_stock") else "No",
                    "QOH":        loc.get("qoh", 0),
                    "Cost":       f"{curr} {cost:.2f}" if cost is not None else "",
                    "Sell Price": f"{curr} {sell:.2f}" if sell is not None else "",
                })
            st.dataframe(loc_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No branch-level inventory data available.")

        with st.expander("Raw Qdrant document (as stored)", expanded=False):
            st.markdown(
                f'<span class="meta">Qdrant point ID: <code>{r["id"]}</code> &nbsp;|&nbsp; '
                f'Vectors: dense (768d cosine) &nbsp;·&nbsp; sparse_model (BM25) '
                f'&nbsp;·&nbsp; sparse_desc (BM25)</span>',
                unsafe_allow_html=True,
            )
            st.json(r["raw_payload"])

# ---------------------------------------------------------------------------
# Evals
# ---------------------------------------------------------------------------

st.divider()
with st.expander("Evals", expanded=False):
    st.caption("90 queries across electrical, mechanical, and plumbing | reranker off | auto classifier | hybrid dense + BM25 + RRF")

    st.markdown("**Overall**")
    st.dataframe([
        {"Metric": "MRR@10",    "Score": 0.6548},
        {"Metric": "Recall@10", "Score": 0.8333},
    ], hide_index=True, use_container_width=False)

    st.markdown("**By domain**")
    st.dataframe([
        {"Domain": "Electrical", "MRR@10": 0.6603, "Recall@10": 0.8333, "N": 30},
        {"Domain": "Mechanical", "MRR@10": 0.6486, "Recall@10": 0.8000, "N": 30},
        {"Domain": "Plumbing",   "MRR@10": 0.6556, "Recall@10": 0.8667, "N": 30},
    ], hide_index=True, use_container_width=False)

    st.markdown("**By query type**")
    st.dataframe([
        {"Query type": "Model number", "MRR@10": 0.7833, "Recall@10": 1.0000, "N": 30},
        {"Query type": "Technical",    "MRR@10": 0.6450, "Recall@10": 0.8000, "N": 30},
        {"Query type": "Descriptive",  "MRR@10": 0.5361, "Recall@10": 0.7000, "N": 30},
    ], hide_index=True, use_container_width=False)

    st.markdown("**By domain x query type**")
    st.dataframe([
        {"Domain": "Electrical", "Query type": "Model number", "MRR@10": 0.8000, "Recall@10": 1.0000, "N": 10},
        {"Domain": "Electrical", "Query type": "Technical",    "MRR@10": 0.6393, "Recall@10": 0.8000, "N": 10},
        {"Domain": "Electrical", "Query type": "Descriptive",  "MRR@10": 0.5417, "Recall@10": 0.7000, "N": 10},
        {"Domain": "Mechanical", "Query type": "Model number", "MRR@10": 0.7000, "Recall@10": 1.0000, "N": 10},
        {"Domain": "Mechanical", "Query type": "Technical",    "MRR@10": 0.5458, "Recall@10": 0.7000, "N": 10},
        {"Domain": "Mechanical", "Query type": "Descriptive",  "MRR@10": 0.7000, "Recall@10": 0.7000, "N": 10},
        {"Domain": "Plumbing",   "Query type": "Model number", "MRR@10": 0.8500, "Recall@10": 1.0000, "N": 10},
        {"Domain": "Plumbing",   "Query type": "Technical",    "MRR@10": 0.7500, "Recall@10": 0.9000, "N": 10},
        {"Domain": "Plumbing",   "Query type": "Descriptive",  "MRR@10": 0.3667, "Recall@10": 0.7000, "N": 10},
    ], hide_index=True, use_container_width=True)
