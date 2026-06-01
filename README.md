# Inventory Search

Hybrid vector search over 39,108 distributor inventory products (electrical,
mechanical, plumbing). Built around Qdrant with three vectors per product —
dense semantic + two BM25 fields — fused via RRF and re-ranked by a
cross-encoder. Single search bar handles part-number lookup, technical-spec
queries, and natural-language descriptions.

## Eval results — 300 queries (mep_eval_300_v3)

Production code path: OpenRouter Gemini 2.5 Flash classifier + 3-retriever
RRF + cross-encoder rerank + size-aware sort.

| Metric    | @5    | @10   | @50   |
|-----------|------:|------:|------:|
| MRR       | 0.799 | 0.803 | 0.805 |
| Recall    | 0.883 | 0.913 | **0.950** |
| Miss      | 0.117 | 0.087 | 0.050 |

**By query type**

| Query type    |    n | MRR@10 | R@10  | R@50  |
|---------------|-----:|------:|------:|------:|
| Model number  |   99 | 0.865 | 0.929 | 0.950 |
| Technical     |   99 | 0.774 | 0.899 | 0.929 |
| Descriptive   |  102 | 0.770 | 0.912 | 0.971 |

---

## Quick start

```bash
# 1. Install deps (uses pint for unit parsing)
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Edit .env -> set OPENROUTER_API_KEY, QDRANT_URL, QDRANT_API_KEY

# 3. Build the inventory collection (wipes + rebuilds)
python scripts/ingest.py

# 4. Run the search UI
streamlit run app.py
```

Want to develop offline without Qdrant Cloud? Set
`QDRANT_LOCAL_PATH=/abs/path/to/local_storage` in your `.env`, ingest, and the
app will use an embedded file store instead.

### Run the eval

```bash
python scripts/run_eval.py scripts/mep_eval_300_v3.csv
```

Eval CSV columns: `query_id, query_text, expected_erp_id, domain, query_type, expected_description`.

Scoring is pure string match between the retrieved point's `internal_id` /
`model_number` and the CSV's `expected_erp_id`.

---

## Architecture

```
INGESTION:   8 source CSV/XLSX  ->  Embed each product 3 ways  ->  Qdrant
QUERY:       User query  ->  Encode + 3-retriever search  ->  Rerank top-50  ->  Top 10
```

**Three vectors per product** (Qdrant named vectors, one record per product):

| Vector         | Model                                | What it indexes                                       |
|----------------|--------------------------------------|-------------------------------------------------------|
| `dense`        | `all-mpnet-base-v2` (768-d, int8)    | description + manufacturer + category + model_number  |
| `sparse_model` | FastEmbed BM25                       | model-number variants (case, separator, alphanumeric) |
| `sparse_desc`  | FastEmbed BM25                       | `normalize_specs(description + mfr + category)`       |

**Query-time pipeline** ([core/search.py](core/search.py)):

1. **Classify** the query as `model_number` / `technical` / `descriptive`
   via OpenRouter Gemini 2.5 Flash ([models/classifier.py](models/classifier.py)).
   No regex or LR fallback — failure here raises.
2. **Encode** into the same three vectors.
3. **Retrieve** via three parallel prefetches with per-type weighted limits
   (see `PREFETCH_LIMITS` in [config.py](config.py)).
4. **Fuse** with RRF (`FusionQuery(Fusion.RRF)`).
5. **Rerank** top 50 with `ms-marco-MiniLM-L-6-v2`.
6. **Size-aware sort** — pint-backed `normalize_specs` extracts canonical
   `sizeNNN` / `mmNNN` tokens from both query and doc; matches outrank
   silent-on-size docs, which outrank size-conflicts. Tiebreak is CE score.

---

## Unit-aware dimension parsing (pint)

`normalize_specs` ([data/normalizers.py](data/normalizers.py)) finds every
`<number><unit>` span and uses `pint` to canonicalize. Handles:

- inch / inches / `"` / IN (case-insensitive)
- decimals (`2.5`), fractions (`3/4`), mixed numbers (`1-1/2`)
- hyphenated separators (`2-inch`)
- millimeter (`50mm`)
- correctly REJECTS `2.5MM2` (wire cross-section in mm², not 2.5 mm length)

Outputs canonical `size200` / `mm50` tokens used by both the BM25 sparse_desc
vector and the post-rerank `apply_size_sort` step. Query side adds metric
bridging (`2 inch` → also `mm50/51/52`); doc side stays imperial-only.

---

## Repository structure

```
qdrant-inventory-search/
├── app.py                        Streamlit UI
├── config.py                     PREFETCH_LIMITS, model names, env config
├── requirements.txt              Pinned deps
├── .env.example                  Copy to .env
│
├── core/
│   ├── client.py                 Qdrant client (cloud by default, local override)
│   ├── filters.py                build_filter() for source/stock/price
│   └── search.py                 search() + search_with_observability()
│
├── models/
│   ├── classifier.py             OpenRouter Gemini 2.5 Flash query classifier
│   ├── embeddings.py             Dense + BM25 encoders, encode_query()
│   └── reranker.py               Cross-encoder rerank_with_scores()
│
├── data/
│   └── normalizers.py            pint-backed normalize_specs + model variants
│
├── scripts/
│   ├── ingest.py                 Canonical re-ingest from raw CSV/XLSX -> Qdrant
│   ├── setup_collection.py       Schema (alternative entry point; ingest.py creates the collection too)
│   ├── evaluate.py               is_hit / mrr_at_k / recall_at_k / miss_at_k
│   └── run_eval.py               Run a CSV eval through the production pipeline
│
├── tests/
│   └── test_dimension_normalization.py    34 cases for normalize_specs
│
└── inventory_data/               Gitignored - raw CSV/XLSX go here
```

---

## Demo URL

The Streamlit demo runs from the developer's laptop, gated to the Parspec
Tailscale network. Ask Sai Kiran for access.
