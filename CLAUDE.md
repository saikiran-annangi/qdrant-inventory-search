# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Hybrid vector search over ~39,000 distributor inventory SKUs (electrical / mechanical / plumbing). Merges the baseline `qdrant-inventory-search` pipeline with two enhancements:

1. **LLM-enriched descriptions** — Gemini generates rich product text for terse/abbreviated items, stored as `extended_description` and included in dense encoding.
2. **Taxonomy soft boost** — a two-stage Gemini classifier predicts the product category of the query; items whose `taxonomy_subcategory` matches get a +0.8 CE-score boost after reranking.

## Commands

```bash
streamlit run app.py                                       # run the UI
python scripts/ingest.py                                   # (re)build the 'inventory' collection
python scripts/run_eval.py scripts/mep_eval_300_v3.csv     # 300-query eval
python tests/test_dimension_normalization.py               # test suite (exits 1 on failure)
```

One-time cache builders (run in this order before ingest):

```bash
python scripts/enrich_descriptions.py          # → enrichment_cache.json  (LLM, ~1h)
python scripts/build_taxonomy_embeddings.py    # → taxonomy_embeddings.json (~1 min)
python scripts/build_taxonomy_from_descriptions.py  # → taxonomy_cache.json (~5 min, local only)
python scripts/ingest.py                       # ingests all three caches
```

Installing deps (override Parspec CodeArtifact index with public PyPI if needed):

```bash
pip install --index-url https://pypi.org/simple/ -r requirements.txt
```

## Required environment (`.env`)

- `OPENROUTER_API_KEY` — Gemini 2.5 Flash, used by: query classifier (`USE_CLASSIFIER=True`), taxonomy query classifier, and `scripts/enrich_descriptions.py`.
- `QDRANT_URL` + `QDRANT_API_KEY` — Qdrant Cloud (production). Copy `.env.example` → `.env` and fill in the cluster URL and API key.
- `QDRANT_LOCAL_PATH` — dev override for an embedded file-based store (leave unset in production).

## Architecture

### Search pipeline (`core/search.py`)

Every query flows through seven stages:

1. **Classify** — `USE_CLASSIFIER=False` (default): every query uses `DEFAULT_PROFILE = "default"` = `{dense:50, sparse_model:50, sparse_desc:40}`. No OpenRouter call, no latency. Set `USE_CLASSIFIER=True` to enable per-type routing via Gemini 2.5 Flash — only helps `model_number` queries meaningfully; hurts recall overall due to 61.7% classifier agreement.
2. **Taxonomy classify** (`models/query_taxonomy_llm.py`) — two-stage Gemini call: Stage 1 returns domain, Stage 2 returns category/subcategory from labels that exist in `taxonomy_cache.json`. Returns `{}` for `model_number` queries. Also returns `{}` for `descriptive` queries (short vague queries map unreliably — wrong prediction hurts more than right one helps).
3. **Encode** (`models/embeddings.py`) — dense mpnet, BM25 `sparse_model` (model variants), BM25 `sparse_desc` (spec-normalized text + electrical attribute anchors).
4. **Retrieve** — three parallel Qdrant prefetches, per-channel limits from `PREFETCH_LIMITS[query_type]`.
5. **Fuse** — server-side Reciprocal Rank Fusion.
6. **Rerank + taxonomy boost + size sort + attribute sort**:
   - Cross-encoder (`models/reranker.py`, float64) rescores top ~50
   - `apply_taxonomy_boost`: +0.8 CE logit for subcategory match, +0.2 for category-only. Items never excluded — wrong predictions just fail to boost.
   - `apply_size_sort`: re-tiers by physical size match (match > silent > conflict)
   - `apply_attribute_sort`: re-tiers by electrical attributes (pole/amp/volt/curve/NEMA/IP/base)

`search_with_observability()` returns a 7-tuple:
`(results, query_type, taxonomy_result, timings, retriever_counts, full_pool, channel_hits)`

### Normalization (`data/normalizers.py`)

- `normalize_specs` collapses size surface forms into anchor tokens (`size75`, `mm50`). Pint-backed.
- **Metric bridging (`bridge_metric=True`) is query-side ONLY.** Do not enable at ingest.
- `spec_text_with_attributes` = normalize_specs output + canonical electrical attribute anchors (`amp20`, `volt125`, `pole3`, `curvec`, `nema4x`, `ip66`, `baseg24q3`). Used for `sparse_desc` at both ingest and query time.
- `attribute_anchor_tokens` / `attribute_relation` back `apply_attribute_sort`.
- `model_number_variants` expands model numbers into casing/separator/alphanum variants.
- `make_id` is deterministic MD5 → UUID of `(source, internal_id)` — ingest is idempotent via upsert.

### Loaders (`data/loaders.py`)

All source loaders are in `data/loaders.py` (not inline in ingest.py). `load_all()` is the single entry point used by `scripts/ingest.py` and `scripts/enrich_descriptions.py`. Two caches are merged at load time if present: `enrichment_cache.json` and `taxonomy_cache.json`.

### Ingestion (`scripts/ingest.py`)

Calls `load_all(attach_caches=True)`, builds three vector fields per product, creates the collection with **int8 scalar quantization** on dense vectors (`always_ram=True`, `on_disk=True`), and upserts. Payload index is created for `taxonomy_domain`, `taxonomy_category`, `taxonomy_subcategory` in addition to the base fields.

### CE reranker — float64 workaround

torch 2.12+ / transformers 5.x causes NaN in float32 BERT attention on CPU. `models/reranker.py` loads the cross-encoder with `torch_dtype=torch.float64` directly via transformers (not sentence-transformers). Do NOT simplify this back to sentence-transformers.

## Gotchas

- **Production target is Qdrant Cloud.** Use `QDRANT_LOCAL_PATH` for local dev only. Cloud collection has 39,108 points with `extended_description` and taxonomy fields fully populated.
- **Taxonomy boost is OFF for descriptive queries.** This is intentional gating in `core/search.py` — vague 2-4 word queries don't reliably predict subcategories.
- **`build_taxonomy_from_descriptions.py` reads raw files (not Qdrant).** It calls `data/loaders.py` directly and infers domain via keyword matching on `product_category` + `description`. No prior ingest required.
- **Backend selection:** `scripts/ingest.py` and `app.py` both use `QDRANT_LOCAL_PATH` if set, else `QDRANT_URL` + `QDRANT_API_KEY` (cloud). Same priority order via `core/client.py`.
- **`scripts/run_eval.py` is the canonical eval entry point.** It reads a CSV and prints Overall + By Domain + By Query Type metrics tables.
- **`data/loaders.py` is the single source of truth for all loader logic.** `scripts/ingest.py` imports from it. If you add a new source file, add the loader there.
- **`app.py` expects 7 values from `search_with_observability`.** The signature is: `(results, query_type, taxonomy_result, timings, retriever_counts, full_pool, channel_hits)`.
