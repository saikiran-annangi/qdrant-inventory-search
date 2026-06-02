# Inventory Search

Hybrid vector search over 39,108 distributor inventory products across
electrical, mechanical, and plumbing categories. A single search bar
handles part numbers, technical specs, and natural-language descriptions.

## Performance

Measured on 300 buyer-style queries.

| Metric  | @5    | @10   | @50   |
| ------- | ----: | ----: | ----: |
| MRR     | 0.799 | 0.803 | 0.805 |
| Recall  | 0.883 | 0.913 | 0.950 |

By query type:

| Query type                 |  n  | Recall@10 | Recall@50 |
| -------------------------- | --: | --------: | --------: |
| Part number                |  99 | 0.929     | 0.950     |
| Technical spec             |  99 | 0.899     | 0.929     |
| Plain-English description  | 102 | 0.912     | 0.971     |

## Pipeline

```
User query
  Classify   Gemini 2.5 Flash (model_number / technical / descriptive)
  Encode     three vectors per query: dense, sparse_model, sparse_desc
  Retrieve   three parallel prefetches with per-type weighted limits
             (dense channel: int8-quantized + rescored against on-disk originals)
  Fuse       reciprocal rank fusion, server-side
  Rerank     cross-encoder on top 50
  Sort       size-aware reordering for queries with size tokens
Top 10 results
```

Architecture diagram: [`presentation/architecture.png`](presentation/architecture.png).

## Stack

| Component          | Choice                                    |
| ------------------ | ----------------------------------------- |
| Vector database    | Qdrant Cloud                              |
| Dense embeddings   | sentence-transformers/all-mpnet-base-v2   |
| Dense quantization | int8 scalar, originals on disk + rescore  |
| Sparse vectors    | FastEmbed BM25 (two fields)               |
| Query classifier   | Gemini 2.5 Flash via OpenRouter           |
| Reranker           | cross-encoder/ms-marco-MiniLM-L-6-v2      |
| UI                 | Streamlit                                 |

### Dense vector quantization

The dense (mpnet) vectors are stored with int8 scalar quantization: the
quantized copy is pinned in RAM (`always_ram`) while the original float32
vectors live on disk (`on_disk`). This shrinks the in-RAM working set ~4x so
the cluster stays on a smaller RAM tier as the catalog grows — the main lever
on Qdrant Cloud cost at scale. At query time the dense channel rescores its
candidate pool against the on-disk originals (`rescore`, `oversampling=2.0`) to
recover the precision lost to quantization. Measured impact on the 300-query
eval: Recall@{5,10,50} unchanged, MRR within +0.001. Binary quantization is
deliberately not used — at 768 dimensions it degrades recall (Qdrant recommends
it only for ~1024d+). The BM25 sparse channels are not quantized.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env             # fill in OPENROUTER_API_KEY, QDRANT_URL, QDRANT_API_KEY
python scripts/ingest.py         # build the inventory collection from raw CSV/XLSX
streamlit run app.py
```

The app reads the Qdrant cluster URL and API key from `.env`. To develop
against an embedded file-based store instead, set `QDRANT_LOCAL_PATH` in
`.env`.

## Running the eval

```bash
python scripts/run_eval.py scripts/mep_eval_300_v3.csv
```

Eval CSV columns: `query_id, query_text, expected_erp_id, domain,
query_type, expected_description`. Scoring is a string match between the
retrieved `internal_id` (or `model_number`) and `expected_erp_id`.

## Repository structure

```
qdrant-inventory-search/
  app.py                          Streamlit UI
  config.py                       Constants and prefetch limits
  requirements.txt
  .env.example

  core/
    client.py                     Qdrant client
    filters.py                    Source, stock, price filters
    search.py                     Public search functions

  models/
    classifier.py                 Query classifier
    embeddings.py                 Dense and BM25 encoders
    reranker.py                   Cross-encoder reranker

  data/
    normalizers.py                Unit-aware dimension parsing,
                                  model-number variants,
                                  manufacturer aliases

  scripts/
    ingest.py                     Build the collection from raw CSV/XLSX
    run_eval.py                   Run an eval CSV through the pipeline
    setup_collection.py
    evaluate.py
    mep_eval_300_v3.csv

  tests/
    test_dimension_normalization.py

  inventory_data/                 Gitignored: raw CSV/XLSX
```

## Data model

Each Qdrant point represents one SKU from one distributor. Branch-level
detail (stock, cost, sell price) lives in a `locations[]` array on the
same point, with rollups (`has_stock`, `total_qoh`, `min_cost`, `max_cost`)
at the top level.

The same SKU carried by multiple distributors produces multiple points,
one per source, so buyers can compare per-distributor stock and price.
