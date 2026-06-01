# Inventory Search

A hybrid vector search service over 39,108 distributor inventory products
across electrical, mechanical, and plumbing categories. One search bar
handles three different query styles: part numbers, technical specs, and
plain-English descriptions.

## Headline numbers

Measured on 300 buyer-style queries through the production pipeline (OpenRouter
Gemini 2.5 Flash classifier, three-retriever RRF, cross-encoder reranker,
size-aware sort).

| Metric                    | @5    | @10   | @50   |
| ------------------------- | ----: | ----: | ----: |
| Mean Reciprocal Rank      | 0.799 | 0.803 | 0.805 |
| Recall                    | 0.883 | 0.913 | 0.950 |
| Miss rate                 | 0.117 | 0.087 | 0.050 |

By query type:

| Query type                 |  n  | MRR@10 | Recall@10 | Recall@50 |
| -------------------------- | --: | -----: | --------: | --------: |
| Part number                |  99 | 0.865  | 0.929     | 0.950     |
| Technical spec             |  99 | 0.774  | 0.899     | 0.929     |
| Plain English description  | 102 | 0.770  | 0.912     | 0.971     |

Reading the numbers in plain English: for 95 percent of queries the right
product appears somewhere in the top 50 results, and for 91 percent it is
in the top 10.

## How it works

```
INGESTION
  Source files (CSV, XLSX)
      -> Per-source loaders (in scripts/ingest.py)
      -> Embed each product three ways
           1. dense  : all-mpnet-base-v2 (768-dim cosine, int8 quantized)
           2. sparse_model : BM25 over model-number variants
           3. sparse_desc  : BM25 over spec-normalized text (pint-backed)
      -> Upsert to Qdrant collection 'inventory'

QUERY
  User query
      -> Classify (Gemini 2.5 Flash via OpenRouter)
           model_number | technical | descriptive
      -> Encode the query into the same three vectors
      -> Three parallel retrievals (per-type weighted prefetch limits)
      -> RRF fusion (server-side, single call)
      -> Cross-encoder rerank on top 50 (ms-marco-MiniLM-L-6-v2)
      -> Size-aware sort (exact-size match wins over near-size mismatches)
      -> Top 10 results
```

A more detailed architecture diagram lives in `presentation/architecture.png`,
generated from the Mermaid source in `presentation/CONFLUENCE_PAGE.md`.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
#    Fill in OPENROUTER_API_KEY and either QDRANT_URL + QDRANT_API_KEY (cloud)
#    or QDRANT_LOCAL_PATH (embedded local store for offline development).

# 3. Build the inventory collection
#    Reads raw CSV/XLSX files from the path in INVENTORY_DATA env var (defaults
#    to ./inventory_data/), wipes the target collection, and rebuilds it.
python scripts/ingest.py

# 4. Run the search UI
streamlit run app.py
```

By default the app talks to Qdrant Cloud (URL and API key from `.env`).
To develop offline against an embedded file-based store, set
`QDRANT_LOCAL_PATH=/abs/path/to/local_storage` in `.env`; `scripts/ingest.py`
and `app.py` will both use that store automatically.

## Running the eval

```bash
python scripts/run_eval.py scripts/mep_eval_300_v3.csv
```

The eval CSV needs these columns: `query_id`, `query_text`, `expected_erp_id`,
`domain`, `query_type`, `expected_description`.

Scoring is a pure string match between the retrieved point's `internal_id`
or `model_number` and the CSV's `expected_erp_id`. The script prints overall
metrics, breakdowns by domain and query type, and classifier agreement
between Gemini's prediction and the CSV label.

## Configuration

Most settings live in `config.py` and are not env-overridable on purpose
(see the design notes below). The two things you do configure at runtime
are secrets and the Qdrant target, both via `.env`:

| Variable           | Purpose                                          |
| ------------------ | ------------------------------------------------ |
| OPENROUTER_API_KEY | Required. Gemini 2.5 Flash query classifier.    |
| QDRANT_URL         | Cloud cluster URL (recommended for production). |
| QDRANT_API_KEY     | Cloud cluster API key.                          |
| QDRANT_LOCAL_PATH  | Optional. Path to an embedded file store.       |
| INVENTORY_DATA     | Optional. Directory containing source CSV/XLSX. |

If `QDRANT_LOCAL_PATH` is set, the client uses the embedded store. Otherwise
it talks to `QDRANT_URL`.

## Repository structure

```
qdrant-inventory-search/
  app.py                          Streamlit UI
  config.py                       Constants: model names, prefetch limits
  requirements.txt                Pinned dependencies
  .env.example                    Copy to .env, fill in secrets

  core/
    client.py                     Qdrant client (cloud by default, local override)
    filters.py                    Source, stock, and price filters
    search.py                     search() and search_with_observability()

  models/
    classifier.py                 OpenRouter Gemini 2.5 Flash query classifier
    embeddings.py                 Dense and BM25 encoders, encode_query()
    reranker.py                   Cross-encoder rerank_with_scores()

  data/
    normalizers.py                Unit-aware dimension parsing (pint-backed),
                                  model-number variants, manufacturer aliases

  scripts/
    ingest.py                     Build the collection from raw CSV/XLSX
    setup_collection.py           Schema-only setup (ingest.py does this too)
    evaluate.py                   is_hit, mrr_at_k, recall_at_k, miss_at_k
    run_eval.py                   Run a CSV eval through the production pipeline
    mep_eval_300_v3.csv           300 buyer-style queries with expected_erp_id

  tests/
    test_dimension_normalization.py   34 cases for normalize_specs

  inventory_data/                 Gitignored; raw CSV/XLSX go here
```

## Design notes

### One point per unique product per source

Each row in Qdrant represents one SKU from one distributor. Per-branch detail
(quantity on hand, cost, sell price) lives in a nested `locations[]` array
on the same point, with top-level rollups (`has_stock`, `total_qoh`,
`min_cost`, `max_cost`, `location_count`).

Cross-source duplicates are intentionally kept as separate points. If both
Guillevin and Standard Supply carry the same Schneider breaker, that produces
two findable records, because the buyer needs to compare per-distributor
stock and price.

### Canonical IDs

`internal_id` is the raw value from the source CSV's ID column, verbatim.
For AU Parspec, Plumbing, Standard Supply, Guillevin\_2 it is the
`Product ERP Code`. For Guillevin\_1 it is the `Item Id`. No abbreviation
prefixing, no hyphen stripping. This makes external evals work without any
patching: an eval that quotes the source CSV's ERP code matches the
indexed `internal_id` by string equality.

### Unit-aware dimension parsing

`normalize_specs` in `data/normalizers.py` is backed by the `pint` library.
One regex finds `<number><unit>` spans (handles plurals, hyphens, fractions,
mixed numbers, double-quote, mm), pint canonicalizes to inches or
millimeters, and the result is emitted as a high-IDF anchor token
(`size200`, `mm50`) that BM25 can match exactly. The same normalization runs
at ingest and at query time, so a query of `2 inch locknut` finds the
indexed `STEEL LOCKNUT 2IN` even though their text differs.

The parser correctly rejects forms like `2.5MM2` (wire cross-section in
square millimeters, not a 2.5 mm length). Tests in
`tests/test_dimension_normalization.py` cover 34 surface forms.

### Size-aware sort

The cross-encoder reranker is size-blind: it can rank a 1/2 inch part above
a 2 inch part if the surrounding text is more aligned. `apply_size_sort` in
`core/search.py` corrects this by re-ordering reranked candidates into
three tiers:

1. doc has a size that matches the query's size
2. doc states no size (silent on the attribute)
3. doc states a size and none matches the query

Cross-encoder score is the tiebreaker within each tier. The sort is a
no-op when the query carries no size token, so it stays on for every query.

### One classifier, no fallback

The query classifier is OpenRouter Gemini 2.5 Flash and nothing else.
There is no logistic regression model, no regex fallback. If OpenRouter
is unreachable the call raises rather than silently degrading. This
deliberate brittleness means the production code path has fewer surfaces
to debug.

## Hosting and access

The Streamlit demo runs from a developer laptop, gated to the Parspec
Tailscale network. Ask Sai Kiran for access.

Production deployment to an always-on host is not yet set up. The
canonical Qdrant Cloud collection has the full 39,108 points, so any host
that can reach the cloud cluster and has the dependencies installed can
serve the app.
