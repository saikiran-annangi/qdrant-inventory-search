"""
Inventory Search -- Streamlit UI

Run:
    streamlit run app.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure the repo root is in sys.path when Streamlit changes cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from models.classifier import CLASSIFY_PROMPT
from models.embeddings import get_dense_model, get_bm25_model
from models.reranker import get_reranker
from core.search import search_with_observability

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
c1.metric("Query type", query_type.replace("_", " ").title())
c2.metric("Classify",   f"{timings['classify_ms']} ms")
with c2:
    with st.popover("View prompt", use_container_width=False):
        st.code(CLASSIFY_PROMPT.format(query=query.strip()), language=None)
c3.metric("Encode",   f"{timings['encode_ms']} ms")
c4.metric("Retrieve", f"{timings['retrieve_ms']} ms")
c5.metric("Rerank",   f"{timings['rerank_ms']} ms")
c6.metric("Total",    f"{timings['total_ms']} ms")

pool = ret_counts["rrf_pool_size"]
st.markdown(
    f"**Retriever contribution** — candidates each retriever passed into the "
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
                    f'— {r["max_cost"]:.2f}</span>',
                    unsafe_allow_html=True,
                )

        # Retrieval path observability
        st.markdown("**Retrieval path**")
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric(
            "Dense",
            f"{r['dense_score']:.3f}" if r["dense_score"] is not None else "—",
            help="Cosine similarity from the dense retriever. '—' means this doc was not in the dense candidate pool.",
        )
        rc2.metric(
            "BM25 model",
            f"{r['sparse_model_score']:.3f}" if r["sparse_model_score"] is not None else "—",
            help="BM25 score from the model-number retriever. '—' means this doc was not in the BM25-model candidate pool.",
        )
        rc3.metric(
            "BM25 desc",
            f"{r['sparse_desc_score']:.3f}" if r["sparse_desc_score"] is not None else "—",
            help="BM25 score from the description retriever. '—' means this doc was not in the BM25-desc candidate pool (or desc channel is disabled for this query type).",
        )
        rc4.metric(
            "RRF fusion",
            f"{r['rrf_score']:.4f}",
            help="Reciprocal Rank Fusion score after merging all active retriever pools.",
        )
        rc5.metric(
            "Reranker",
            f"{r['reranker_score']:.4f}",
            help="Cross-encoder score (ms-marco-MiniLM-L-6-v2). Raw logit — typically [-5, +10], higher = better match.",
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
    st.caption("90 queries across electrical, mechanical, and plumbing | Gemini classifier | hybrid dense + BM25 + RRF")

    st.markdown("**Overall**")
    st.dataframe([
        {"Metric": "MRR@3",     "@3": 0.7481, "@10": 0.7575, "@50": 0.7592},
        {"Metric": "Recall",    "@3": 0.8111, "@10": 0.8667, "@50": 0.8889},
        {"Metric": "Miss",      "@3": 0.1889, "@10": 0.1333, "@50": 0.1111},
    ], hide_index=True, use_container_width=False)

    st.markdown("**By domain**")
    st.dataframe([
        {"Domain": "Electrical", "MRR@3": 0.7611, "R@3": 0.8333, "Miss@3": 0.1667, "MRR@10": 0.7644, "R@10": 0.8667, "Miss@10": 0.1333, "MRR@50": 0.7694, "R@50": 0.9333, "Miss@50": 0.0667, "N": 30},
        {"Domain": "Mechanical", "MRR@3": 0.7667, "R@3": 0.8333, "Miss@3": 0.1667, "MRR@10": 0.7733, "R@10": 0.8667, "Miss@10": 0.1333, "MRR@50": 0.7733, "R@50": 0.8667, "Miss@50": 0.1333, "N": 30},
        {"Domain": "Plumbing",   "MRR@3": 0.7167, "R@3": 0.7667, "Miss@3": 0.2333, "MRR@10": 0.7347, "R@10": 0.8667, "Miss@10": 0.1333, "MRR@50": 0.7347, "R@50": 0.8667, "Miss@50": 0.1333, "N": 30},
    ], hide_index=True, use_container_width=True)

    st.markdown("**By query type**")
    st.dataframe([
        {"Query type": "Model number", "MRR@3": 0.9833, "R@3": 1.0000, "Miss@3": 0.0000, "MRR@10": 0.9833, "R@10": 1.0000, "Miss@10": 0.0000, "MRR@50": 0.9833, "R@50": 1.0000, "Miss@50": 0.0000, "N": 30},
        {"Query type": "Technical",    "MRR@3": 0.7000, "R@3": 0.7667, "Miss@3": 0.2333, "MRR@10": 0.7067, "R@10": 0.8000, "Miss@10": 0.2000, "MRR@50": 0.7117, "R@50": 0.8667, "Miss@50": 0.1333, "N": 30},
        {"Query type": "Descriptive",  "MRR@3": 0.5611, "R@3": 0.6667, "Miss@3": 0.3333, "MRR@10": 0.5825, "R@10": 0.8000, "Miss@10": 0.2000, "MRR@50": 0.5825, "R@50": 0.8000, "Miss@50": 0.2000, "N": 30},
    ], hide_index=True, use_container_width=True)

    st.markdown("**By domain × query type**")
    st.dataframe([
        {"Domain": "Electrical", "Type": "Model number", "MRR@3": 1.0000, "R@3": 1.0000, "Miss@3": 0.0000, "MRR@10": 1.0000, "R@10": 1.0000, "Miss@10": 0.0000, "MRR@50": 1.0000, "R@50": 1.0000, "Miss@50": 0.0000, "N": 10},
        {"Domain": "Electrical", "Type": "Technical",    "MRR@3": 0.7000, "R@3": 0.7000, "Miss@3": 0.3000, "MRR@10": 0.7000, "R@10": 0.7000, "Miss@10": 0.3000, "MRR@50": 0.7150, "R@50": 0.9000, "Miss@50": 0.1000, "N": 10},
        {"Domain": "Electrical", "Type": "Descriptive",  "MRR@3": 0.5833, "R@3": 0.8000, "Miss@3": 0.2000, "MRR@10": 0.5933, "R@10": 0.9000, "Miss@10": 0.1000, "MRR@50": 0.5933, "R@50": 0.9000, "Miss@50": 0.1000, "N": 10},
        {"Domain": "Mechanical", "Type": "Model number", "MRR@3": 0.9500, "R@3": 1.0000, "Miss@3": 0.0000, "MRR@10": 0.9500, "R@10": 1.0000, "Miss@10": 0.0000, "MRR@50": 0.9500, "R@50": 1.0000, "Miss@50": 0.0000, "N": 10},
        {"Domain": "Mechanical", "Type": "Technical",    "MRR@3": 0.6500, "R@3": 0.7000, "Miss@3": 0.3000, "MRR@10": 0.6700, "R@10": 0.8000, "Miss@10": 0.2000, "MRR@50": 0.6700, "R@50": 0.8000, "Miss@50": 0.2000, "N": 10},
        {"Domain": "Mechanical", "Type": "Descriptive",  "MRR@3": 0.7000, "R@3": 0.8000, "Miss@3": 0.2000, "MRR@10": 0.7000, "R@10": 0.8000, "Miss@10": 0.2000, "MRR@50": 0.7000, "R@50": 0.8000, "Miss@50": 0.2000, "N": 10},
        {"Domain": "Plumbing",   "Type": "Model number", "MRR@3": 1.0000, "R@3": 1.0000, "Miss@3": 0.0000, "MRR@10": 1.0000, "R@10": 1.0000, "Miss@10": 0.0000, "MRR@50": 1.0000, "R@50": 1.0000, "Miss@50": 0.0000, "N": 10},
        {"Domain": "Plumbing",   "Type": "Technical",    "MRR@3": 0.7500, "R@3": 0.9000, "Miss@3": 0.1000, "MRR@10": 0.7500, "R@10": 0.9000, "Miss@10": 0.1000, "MRR@50": 0.7500, "R@50": 0.9000, "Miss@50": 0.1000, "N": 10},
        {"Domain": "Plumbing",   "Type": "Descriptive",  "MRR@3": 0.4000, "R@3": 0.4000, "Miss@3": 0.6000, "MRR@10": 0.4542, "R@10": 0.7000, "Miss@10": 0.3000, "MRR@50": 0.4542, "R@50": 0.7000, "Miss@50": 0.3000, "N": 10},
    ], hide_index=True, use_container_width=True)
