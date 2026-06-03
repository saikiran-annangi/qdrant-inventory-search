"""
Inventory Search — Streamlit UI

Run:
    streamlit run app.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import streamlit as st

from models.classifier import CLASSIFY_PROMPT
from models.embeddings import get_dense_model, get_bm25_model
from models.reranker import get_reranker
from core.search import search_with_observability
from config import PREFETCH_LIMITS

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
    .tax-badge { font-size: 0.75rem; color: #5c6bc0; background: #e8eaf6;
                 padding: 2px 6px; border-radius: 4px; }
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
# UI helpers
# ---------------------------------------------------------------------------

def stock_label(has_stock) -> str:
    if has_stock is True:
        return '<span class="stock-in">In Stock</span>'
    if has_stock is False:
        return '<span class="stock-out">Out of Stock</span>'
    return '<span class="stock-unk">Unknown</span>'


def taxonomy_label(taxonomy_result: dict) -> str:
    if not taxonomy_result:
        return "—"
    domain  = taxonomy_result.get("taxonomy_domain",      "") or ""
    cat     = taxonomy_result.get("taxonomy_category",    "") or ""
    subcat  = taxonomy_result.get("taxonomy_subcategory", "") or ""
    if subcat:
        return f"{domain} › {cat} › {subcat}"
    if cat:
        return f"{domain} › {cat}"
    return domain or "—"


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
        placeholder="Model number, spec, or description — e.g. 'Schneider 16A single pole MCB'  or  'K-2084'",
        label_visibility="collapsed",
    )
with col_filter:
    source_label = st.selectbox("Distributor", options=list(SOURCES.keys()), label_visibility="collapsed")
    source_filter = SOURCES[source_label]
with col_btn:
    search_clicked = st.button("Search", use_container_width=True)

if not query or not query.strip():
    st.stop()

_cache_key = (query.strip(), source_filter)
if search_clicked or st.session_state.get("_search_key") != _cache_key:
    st.session_state["_search_key"] = _cache_key
    st.session_state["erp_lookup"]  = ""
    with st.spinner("Searching..."):
        _r = search_with_observability(query.strip(), source_filter=source_filter)
    # search_with_observability returns 7 values:
    # results, query_type, taxonomy_result, timings, retriever_counts, full_pool, channel_hits
    st.session_state["_search_results"]  = _r[0]
    st.session_state["_query_type"]      = _r[1]
    st.session_state["_taxonomy"]        = _r[2]
    st.session_state["_timings"]         = _r[3]
    st.session_state["_ret_counts"]      = _r[4]
    st.session_state["_full_pool"]       = _r[5]
    st.session_state["_channel_hits"]    = _r[6]

if "_search_results" not in st.session_state:
    st.stop()

results         = st.session_state["_search_results"]
query_type      = st.session_state["_query_type"]
taxonomy_result = st.session_state.get("_taxonomy", {})
timings         = st.session_state["_timings"]
ret_counts      = st.session_state["_ret_counts"]
full_pool       = st.session_state["_full_pool"]

# ---------------------------------------------------------------------------
# Pipeline observability
# ---------------------------------------------------------------------------

st.divider()
st.markdown("**Pipeline**")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Query type",  query_type.replace("_", " ").title())
c2.metric("Classify",    f"{timings['classify_ms']} ms")
with c2:
    with st.popover("View prompt", use_container_width=False):
        st.code(CLASSIFY_PROMPT.format(query=query.strip()), language=None)
c3.metric("Taxonomy",   f"{timings.get('taxonomy_ms', 0)} ms",
          help=taxonomy_label(taxonomy_result))
c4.metric("Encode",     f"{timings['encode_ms']} ms")
c5.metric("Retrieve",   f"{timings['retrieve_ms']} ms")
c6.metric("Rerank",     f"{timings['rerank_ms']} ms")
c7.metric("Total",      f"{timings['total_ms']} ms")

if taxonomy_result and taxonomy_result.get("taxonomy_domain"):
    st.caption(
        f"Taxonomy prediction: **{taxonomy_label(taxonomy_result)}** "
        f"— items matching this subcategory received a +0.8 score boost after reranking."
    )

pool = ret_counts["rrf_pool_size"]
st.markdown(
    f"**Retriever contribution** — candidates each retriever passed into the "
    f"{pool}-candidate RRF pool (can overlap)"
)
rc1, rc2, rc3 = st.columns(3)
rc1.metric("Dense",          ret_counts["dense"],
           help="Candidates from semantic embedding retriever")
rc2.metric("Sparse (model)", ret_counts["sparse_model"],
           help="Candidates from BM25 over model number variants")
rc3.metric("Sparse (desc)",  ret_counts["sparse_desc"],
           help="Candidates from BM25 over description and specs")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

st.divider()

for r in results:
    rrf_rank  = r.get("rrf_rank") or "?"
    ce_score  = r["reranker_score"]
    rrf_score = r["rrf_score"]
    header = (
        f"#{r['rank']}  |  {r['model_number']}  |  {r['manufacturer_name']}  "
        f"  CE: {ce_score:.2f}  ·  RRF score: {rrf_score:.4f}  (was RRF rank #{rrf_rank})"
    )

    with st.expander(header, expanded=True):
        left, right = st.columns([3, 1])

        with left:
            st.markdown("**Description**")
            st.write(r["description"])
            st.markdown("**Extended Description**")
            if r.get("extended_description"):
                st.write(r["extended_description"])
            else:
                st.caption("—")
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
                    f'— {r["max_cost"]:.2f}</span>',
                    unsafe_allow_html=True,
                )

        # Retrieval path observability
        st.markdown("**Retrieval path**")
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric(
            "Dense",
            f"{r['dense_score']:.3f}" if r["dense_score"] is not None else "—",
            help="Cosine similarity from the dense retriever.",
        )
        rc2.metric(
            "BM25 model",
            f"{r['sparse_model_score']:.3f}" if r["sparse_model_score"] is not None else "—",
            help="BM25 score from the model-number retriever.",
        )
        rc3.metric(
            "BM25 desc",
            f"{r['sparse_desc_score']:.3f}" if r["sparse_desc_score"] is not None else "—",
            help="BM25 score from the description retriever.",
        )
        rc4.metric(
            "RRF fusion",
            f"{r['rrf_score']:.4f}",
            help="Reciprocal Rank Fusion score after merging all active retriever pools.",
        )
        rc5.metric(
            "Reranker",
            f"{r['reranker_score']:.4f}",
            help="Cross-encoder score (ms-marco-MiniLM-L-6-v2). Raw logit; higher = better match.",
        )
        st.caption(f"Retrieved by: **{r['retrieval_path']}**")

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
                f'Vectors: dense (768d cosine, int8-quantized) &nbsp;·&nbsp; '
                f'sparse_model (BM25) &nbsp;·&nbsp; sparse_desc (BM25)</span>',
                unsafe_allow_html=True,
            )
            # Reorder so extended_description appears right after description
            _p = r["raw_payload"]
            _ordered = {}
            for k, v in _p.items():
                _ordered[k] = v
                if k == "description":
                    _ordered["extended_description"] = _p.get("extended_description")
            st.json(_ordered)

# ---------------------------------------------------------------------------
# ERP ID lookup — trace one ERP ID through this query's pipeline
# ---------------------------------------------------------------------------

st.divider()
with st.expander("ERP ID lookup", expanded=False):
    st.caption(
        "Enter an ERP ID to trace it through THIS query's pipeline: "
        "(1) which retrievers surfaced it, (2) whether it reached the RRF pool, "
        "(3) where the reranker placed it."
    )
    lookup_id = st.text_input(
        "ERP ID",
        placeholder="e.g. 30-0101",
        label_visibility="collapsed",
        key="erp_lookup",
    )

    channel_hits = st.session_state.get("_channel_hits", {})
    pool         = st.session_state.get("_full_pool", [])

    if lookup_id and lookup_id.strip():
        needle = lookup_id.strip().lower()

        def _matches(iid: str) -> bool:
            iid = str(iid).lower()
            return iid == needle or iid.startswith(needle + "_")

        def _channel_rank(ch: str):
            for iid, rnk in channel_hits.get(ch, {}).items():
                if _matches(iid):
                    return rnk
            return None

        d_rank  = _channel_rank("dense")
        sm_rank = _channel_rank("sparse_model")
        sd_rank = _channel_rank("sparse_desc")
        pool_match    = next((e for e in pool if _matches(e["internal_id"])), None)
        retrieved_any = any(r is not None for r in (d_rank, sm_rank, sd_rank))
        limits        = PREFETCH_LIMITS.get(query_type, {})

        st.markdown("**1. Retrieved by the retrievers?**")
        def _line(name, rnk, lim):
            if lim == 0:
                return f"- ⚪ **{name}** — not used for `{query_type}` queries"
            if rnk is not None:
                return f"- ✅ **{name}** — retrieved at rank #{rnk} (of top {lim})"
            return f"- ⛔ **{name}** — not retrieved (outside top {lim})"
        st.markdown(_line("Dense",          d_rank,  limits.get("dense", 0)))
        st.markdown(_line("Sparse · model", sm_rank, limits.get("sparse_model", 0)))
        st.markdown(_line("Sparse · desc",  sd_rank, limits.get("sparse_desc", 0)))

        st.markdown("**2. In the RRF candidate pool?**")
        if pool_match:
            st.markdown(
                f"- ✅ Yes — RRF rank **#{pool_match['rrf_rank']}** of {len(pool)} "
                f"(RRF score {pool_match['rrf_score']})"
            )
        else:
            st.markdown(f"- ⛔ No — did not make the top-{len(pool)} RRF pool")

        st.markdown("**3. Reranker rank?**")
        if pool_match and pool_match.get("rerank_rank"):
            st.markdown(
                f"- Final rank after reranking: **#{pool_match['rerank_rank']}** of {len(pool)} "
                f"(CE score {pool_match['reranker_score']:.4f})"
            )
        else:
            st.markdown("- — only RRF-pool candidates are reranked")

        if pool_match:
            st.caption(f"`{pool_match['source']}` · {pool_match['description']}")
        elif not retrieved_any:
            st.warning(
                "Not surfaced by any retriever for this query. Either it's too "
                "dissimilar, or the ERP ID isn't in the index."
            )
