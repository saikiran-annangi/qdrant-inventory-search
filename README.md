# Inventory Search

Hybrid vector search over industrial inventory (electrical, mechanical, plumbing). Combines dense semantic embeddings, two BM25 sparse fields, RRF fusion, and cross-encoder re-ranking to handle model number lookups, technical spec queries, and natural-language descriptions in a single pipeline.

## Eval results — 90 queries (electrical / mechanical / plumbing)

| Metric    | Score  |
|-----------|--------|
| MRR@3     | 0.748  |
| Recall@3  | 0.811  |
| Miss@3    | 0.189  |

**By domain**

| Domain      | MRR@3 | Recall@3 | Miss@3 |
|-------------|-------|----------|--------|
| Electrical  | 0.761 | 0.833    | 0.167  |
| Mechanical  | 0.767 | 0.833    | 0.167  |
| Plumbing    | 0.717 | 0.767    | 0.233  |

**By query type**

| Query type   | MRR@3 | Recall@3 | Miss@3 |
|--------------|-------|----------|--------|
| Model number | 0.983 | 1.000    | 0.000  |
| Technical    | 0.700 | 0.767    | 0.233  |
| Descriptive  | 0.561 | 0.667    | 0.333  |

---

## Repository structure

```
qdrant-inventory-search/
├── config.py                   Constants: URLs, model names, prefetch limits
├── app.py                      Streamlit UI
│
├── core/
│   ├── client.py               Qdrant HTTP client singleton
│   ├── filters.py              build_filter() for source/stock/price filters
│   └── search.py               search() and search_with_observability()
│
├── models/
│   ├── classifier.py           Gemini 2.5 Flash query classifier (via OpenRouter)
│   ├── embeddings.py           Dense + BM25 encoders, encode_query()
│   └── reranker.py             CrossEncoder reranker, rerank_with_scores()
│
├── data/
│   ├── normalizers.py          Spec expansion, model variants, manufacturer aliases
│   └── loader.py               Source-specific loaders + load_all()
│
├── scripts/
│   ├── setup_collection.py     Create the Qdrant collection (run once)
│   ├── ingest.py               Embed and upsert all inventory records
│   ├── evaluate.py             90-query eval benchmark
│   └── migrate_to_server.py    One-time migration from local SQLite to HTTP server
│
├── inventory_data/             Gitignored — place raw CSV/XLSX files here
└── eval_queries.json           90 labeled evaluation queries
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Add OPENROUTER_API_KEY to .env

# 3. Start Qdrant server
./qdrant --config-path qdrant_config.yaml

# 4. Create collection + ingest
python scripts/setup_collection.py
python scripts/ingest.py

# 5. Run the app
streamlit run app.py
```

To run evaluations:

```bash
python scripts/evaluate.py --rerank
```
