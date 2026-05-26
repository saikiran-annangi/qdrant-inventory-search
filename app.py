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

import json
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

# Re-run the search only when the button is clicked or the query/filter changed.
# On any other rerun (e.g. ERP ID box, expander toggle) we render from session state.
_cache_key = (query.strip(), source_filter)
if search_clicked or st.session_state.get("_search_key") != _cache_key:
    st.session_state["_search_key"] = _cache_key
    # Clear the ERP ID lookup so a stale ID from the previous search
    # is not immediately evaluated against the new query's candidate pool.
    st.session_state["erp_lookup"] = ""
    with st.spinner("Searching..."):
        _r = search_with_observability(query.strip(), source_filter=source_filter)
    st.session_state["_search_results"]  = _r[0]
    st.session_state["_query_type"]      = _r[1]
    st.session_state["_timings"]         = _r[2]
    st.session_state["_ret_counts"]      = _r[3]
    st.session_state["_full_pool"]       = _r[4]

# Always render from session state — safe across any rerun
if "_search_results" not in st.session_state:
    st.stop()

results    = st.session_state["_search_results"]
query_type = st.session_state["_query_type"]
timings    = st.session_state["_timings"]
ret_counts = st.session_state["_ret_counts"]
full_pool  = st.session_state["_full_pool"]

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
# ERP ID / model-number lookup
# ---------------------------------------------------------------------------

st.divider()
with st.expander("ERP ID / Model number position lookup", expanded=False):
    st.caption(
        "Enter an internal ID or model number to check if it appeared in the "
        "top-50 RRF candidate pool and where the reranker placed it."
    )
    lookup_id = st.text_input(
        "ERP ID or model number",
        placeholder="e.g. 12345  or  12345_0  or  AB-XYZ",
        label_visibility="collapsed",
        key="erp_lookup",
    )

    pool = st.session_state.get("_full_pool", [])

    if lookup_id and lookup_id.strip() and pool:
        needle = lookup_id.strip().lower()

        def _pool_match(entry: dict) -> bool:
            iid = str(entry["internal_id"]).lower()
            mn  = str(entry["model_number"]).lower()
            if iid == needle or mn == needle:
                return True
            # Burnaby DC variant: "99999_0" matches "99999"
            if iid.startswith(needle + "_"):
                return True
            return False

        matches = [e for e in pool if _pool_match(e)]

        if not matches:
            st.error(f"**Not found** — `{lookup_id}` was not in the top-{len(pool)} RRF candidates for this query.")
        else:
            for m in matches:
                rrf_r    = m["rrf_rank"]
                rerank_r = m["rerank_rank"] or "?"
                pool_n   = len(pool)
                ce       = m["reranker_score"]
                rrf_s    = m["rrf_score"]
                st.success(
                    f"**Found!**  Internal ID: `{m['internal_id']}`  |  Model: `{m['model_number']}`  |  Source: `{m['source']}`\n\n"
                    f"- **RRF pool**: rank **#{rrf_r}** / {pool_n}   (RRF score: {rrf_s})\n"
                    f"- **After reranking**: rank **#{rerank_r}** / {pool_n}   (CE score: {ce:.4f})"
                )
                st.caption(m["description"])

    elif lookup_id and lookup_id.strip() and not pool:
        st.warning("Run a search first — the candidate pool is empty.")


# ---------------------------------------------------------------------------
# Evals
# ---------------------------------------------------------------------------

_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")

@st.cache_data(show_spinner=False)
def _load_eval() -> dict:
    """Load eval summary from eval_results.json (cached until file changes)."""
    if not os.path.exists(_EVAL_PATH):
        return {}
    with open(_EVAL_PATH) as f:
        return json.load(f).get("summary", {})

def _row(s: dict, label_key: str, label_val: str) -> dict:
    return {
        label_key:  label_val,
        "MRR@3":   s["MRR@3"],   "R@3":   s["Recall@3"],   "Miss@3":   s["Miss@3"],
        "MRR@10":  s["MRR@10"],  "R@10":  s["Recall@10"],  "Miss@10":  s["Miss@10"],
        "MRR@50":  s["MRR@50"],  "R@50":  s["Recall@50"],  "Miss@50":  s["Miss@50"],
        "N": s["n"],
    }

st.divider()
with st.expander("Evals", expanded=False):
    ev = _load_eval()
    if not ev:
        st.warning("eval_results.json not found — run `python scripts/evaluate.py` to generate it.")
        st.stop()

    st.caption("90 queries across electrical, mechanical, and plumbing | Gemini classifier | hybrid dense + BM25 + RRF")

    st.markdown("**Overall**")
    ov = ev["overall"]
    st.dataframe([
        {"Metric": "MRR",    "@3": ov["MRR@3"],    "@10": ov["MRR@10"],    "@50": ov["MRR@50"]},
        {"Metric": "Recall", "@3": ov["Recall@3"],  "@10": ov["Recall@10"],  "@50": ov["Recall@50"]},
        {"Metric": "Miss",   "@3": ov["Miss@3"],    "@10": ov["Miss@10"],    "@50": ov["Miss@50"]},
    ], hide_index=True, use_container_width=False)

    st.markdown("**By domain**")
    st.dataframe([
        _row(ev["electrical"], "Domain", "Electrical"),
        _row(ev["mechanical"], "Domain", "Mechanical"),
        _row(ev["plumbing"],   "Domain", "Plumbing"),
    ], hide_index=True, use_container_width=True)

    st.markdown("**By query type**")
    st.dataframe([
        _row(ev["model_number"], "Query type", "Model number"),
        _row(ev["technical"],    "Query type", "Technical"),
        _row(ev["descriptive"],  "Query type", "Descriptive"),
    ], hide_index=True, use_container_width=True)

    st.markdown("**By domain × query type**")
    domain_type_rows = []
    for domain in ("electrical", "mechanical", "plumbing"):
        for qtype in ("model_number", "technical", "descriptive"):
            key = f"{domain}/{qtype}"
            if key in ev:
                r = _row(ev[key], "Type", qtype.replace("_", " ").title())
                r["Domain"] = domain.title()
                # reorder columns
                domain_type_rows.append({k: r[k] for k in
                    ["Domain", "Type", "MRR@3", "R@3", "Miss@3",
                     "MRR@10", "R@10", "Miss@10", "MRR@50", "R@50", "Miss@50", "N"]})
    st.dataframe(domain_type_rows, hide_index=True, use_container_width=True)
