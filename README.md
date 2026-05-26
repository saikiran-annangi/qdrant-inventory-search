# Inventory Search

A hybrid vector search system for industrial inventory (electrical, mechanical, plumbing). Combines dense semantic embeddings, two BM25 sparse fields, and cross-encoder re-ranking to handle model number lookups, technical spec queries, and natural-language descriptions in a single pipeline.

**Current eval results (90-query benchmark, auto classifier):**

| Metric    | Score  |
|-----------|--------|
| MRR@3     | 0.748  |
| Recall@3  | 0.811  |
| Miss@3    | 0.189  |

By domain (reranker ON):

| Domain      | MRR@3 | Recall@3 | Miss@3 |
|-------------|-------|----------|--------|
| Electrical  | 0.761 | 0.833    | 0.167  |
| Mechanical  | 0.767 | 0.833    | 0.167  |
| Plumbing    | 0.717 | 0.767    | 0.233  |

By query type (reranker ON):

| Query type   | MRR@3 | Recall@3 | Miss@3 |
|--------------|-------|----------|--------|
| Model number | 0.983 | 1.000    | 0.000  |
| Technical    | 0.700 | 0.767    | 0.233  |
| Descriptive  | 0.561 | 0.667    | 0.333  |

---

## Repository structure

```
qdrant-inventory-search/
|
+-- config.py                   All constants: URLs, model names, limits
|
+-- models/
|   +-- embeddings.py           Dense + BM25 model loaders, encode_query()
|   +-- reranker.py             CrossEncoder loader, rerank()
|   +-- classifier.py           3-tier query classifier
|
+-- core/
|   +-- client.py               Qdrant HTTP client singleton
|   +-- filters.py              build_filter() for source/stock/price filters
|   +-- search.py               Main search() function
|
+-- data/
|   +-- normalizers.py          Spec expansion, model variants, manufacturer aliases
|   +-- loader.py               Source-specific loaders + load_all()
|
+-- app.py                      Streamlit UI
|
+-- scripts/
|   +-- setup_collection.py     Create the Qdrant collection (run once)
|   +-- ingest.py               Embed and upsert all inventory records
|   +-- evaluate.py             Run the 90-query eval benchmark
|   +-- build_classifier.py     Train the logistic regression classifier
|   +-- migrate_to_server.py    One-time migration from local SQLite to HTTP server
|
+-- inventory_data/             Gitignored -- place raw CSV/XLSX files here
+-- eval_queries.json           90 labeled evaluation queries
+-- .env.example                Template for environment variables
+-- requirements.txt
```

---

## Prerequisites

- Python 3.11 or 3.12
- A running [Qdrant](https://qdrant.tech/documentation/guides/installation/) server (v1.12+)
- Raw inventory data files placed in `inventory_data/`
- An [OpenRouter](https://openrouter.ai) API key (recommended for best classifier accuracy)

---

## Setup from scratch

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_ORG/qdrant-inventory-search.git
cd qdrant-inventory-search
```

### 2. Create a Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```
OPENROUTER_API_KEY=sk-or-v1-...   # Required for Gemini 2.5 Flash classifier
GEMINI_API_KEY=...                 # Optional, only for description generation during ingest
```

Get an OpenRouter key at https://openrouter.ai. The `google/gemini-2.5-flash` model is used for classification (costs fractions of a cent per query). Without this key, the system falls back to the regex classifier at ~55% accuracy.

### 5. Start the Qdrant server

Download the Qdrant binary for your platform from https://github.com/qdrant/qdrant/releases (v1.12 or later).

**macOS ARM64:**
```bash
curl -L https://github.com/qdrant/qdrant/releases/download/v1.16.1/qdrant-aarch64-apple-darwin.tar.gz | tar xz
chmod +x qdrant
```

Create a config file (optional -- defaults work fine):

```yaml
# qdrant_config.yaml
storage:
  storage_path: ./qdrant_storage
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
log_level: WARN
```

Start the server:

```bash
./qdrant --config-path qdrant_config.yaml
```

Verify it is running:

```bash
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine","version":"..."}
```

### 6. Place inventory data files

Copy your raw data files into the `inventory_data/` folder. The loader expects these file names:

```
inventory_data/
  AU Parspec inventory load 10032026 SEND AB V 1 Test copy for demo.csv
  Burnaby DC Lighting Inventory 9 17.xlsx
  Guillevin_inventory_data_1.xlsx
  Guillevin_inventory_data_2_utf8.csv
  INVENTORY SAMPLE.xlsx
  Plumbing Inventory example.xlsx
  Standard Supply_inventory_data_1.csv
```

If your file names differ, update the paths at the top of each loader function in `data/loader.py`.

### 7. Create the Qdrant collection

```bash
python scripts/setup_collection.py
```

This creates the collection with all three vector fields and payload indexes. It drops and recreates the collection if it already exists.

### 8. Run ingestion

```bash
python scripts/ingest.py
```

This reads all inventory files, generates embeddings (dense + two BM25 sparse), and upserts everything into Qdrant in batches of 256. For ~35k products, expect 10-15 minutes on CPU.

Ingestion is idempotent -- re-running it overwrites existing records via upsert.

### 9. Start the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Optional: train the logistic regression classifier

The LR classifier runs locally with no API calls and is ~5-10x faster than OpenRouter. Train it after ingestion (it reads model numbers directly from Qdrant):

```bash
python scripts/build_classifier.py          # train and save
python scripts/build_classifier.py --eval   # also evaluate on the 90 eval queries
```

This saves `query_classifier.joblib` to the repo root. Once present, it takes priority over the OpenRouter classifier automatically.

---

## Running evaluations

```bash
python scripts/evaluate.py                # auto classifier, no reranker
python scripts/evaluate.py --rerank       # with cross-encoder reranker (slower but more accurate)
python scripts/evaluate.py --gt-type      # oracle: use ground-truth query type
```

Results are saved to `eval_results.json` in the repo root.

---

## Adding a new inventory source

1. Add a loader function in `data/loader.py` following the existing pattern.
2. Register it in the `LOADERS` dict at the bottom of that file.
3. Add the source name to the `SOURCES` dict in `app.py` for UI filtering.
4. Re-run ingestion (`python scripts/ingest.py`) -- existing records are not affected.

---

## Deduplication notes

Records are deduplicated **within each source** by `internal_id` (manufacturer abbreviation + model number, or the source's ERP code). A product that appears in multiple distributor files will have **one Qdrant point per source**. This is intentional: each source has its own pricing, stock levels, and branch inventory.

Point IDs are deterministic: `md5(f"{source}:{internal_id}")`. Re-ingesting the same record always upserts to the same point -- no duplicates accumulate across runs.

---

## Environment variables reference

| Variable           | Required | Default               | Description                             |
|--------------------|----------|-----------------------|-----------------------------------------|
| OPENROUTER_API_KEY | Recommended | (none)             | Enables Gemini 2.5 Flash classifier     |
| GEMINI_API_KEY     | No       | (none)                | Enables LLM description generation      |
| QDRANT_URL         | No       | http://localhost:6333 | URL of the Qdrant HTTP server           |
