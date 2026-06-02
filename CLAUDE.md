# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Hybrid vector search over ~39,000 distributor inventory SKUs (electrical / mechanical / plumbing). A single query box handles part numbers, technical specs, and plain-English descriptions. Backed by Qdrant (cloud or embedded), a Streamlit UI, and an LLM query classifier.

## Commands

Always use the project virtualenv at `.venv/`. Dependencies are heavy (torch, sentence-transformers).

```bash
.venv/bin/streamlit run app.py                          # run the UI (loads models on first request)
.venv/bin/python scripts/ingest.py                      # (re)build the 'inventory' collection from inventory_data/
.venv/bin/python scripts/run_eval.py scripts/mep_eval_300_v3.csv   # 300-query eval -> scripts/eval_results.json
.venv/bin/python tests/test_dimension_normalization.py  # the test suite (plain script, exits 1 on failure — NOT pytest)
```

Installing deps: the machine's global pip points at an expired Parspec CodeArtifact index, so always override with public PyPI:

```bash
.venv/bin/pip install --index-url https://pypi.org/simple/ -r requirements.txt
```

## Required environment (`.env`)

- `OPENROUTER_API_KEY` — **hard requirement**. `models/classifier.py` raises `EnvironmentError` without it; there is no fallback path.
- `QDRANT_URL` + `QDRANT_API_KEY` — for Qdrant Cloud.
- `QDRANT_LOCAL_PATH` — dev override for an embedded file-based store (see backend-selection gotcha below).

## Architecture

### Search pipeline (`core/search.py`)
Every query flows through five stages:
1. **Classify** (`models/classifier.py`) — Gemini 2.5 Flash via OpenRouter returns exactly one of `model_number` / `technical` / `descriptive`. In-process cached. An unexpected token raises.
2. **Encode** (`models/embeddings.py` → `encode_query`) — produces **three** vectors per query: a dense mpnet embedding, a BM25 `sparse_model` vector (over model-number variants), and a BM25 `sparse_desc` vector (over spec-normalized text).
3. **Retrieve** — three parallel Qdrant prefetches. The classified `query_type` selects per-channel limits from `config.py:PREFETCH_LIMITS`. Note `model_number` queries set `sparse_desc` limit to 0 (siblings in a family share descriptions and would pollute rank 1); channels with limit 0 are skipped entirely.
4. **Fuse** — server-side Reciprocal Rank Fusion (`FusionQuery(Fusion.RRF)`).
5. **Rerank + size-sort + attribute-sort** — cross-encoder (`models/reranker.py`) rescores the top ~50, then two deterministic re-tiering passes run in `core/search.py`: `apply_size_sort` (by physical size) and `apply_attribute_sort` (by electrical attributes — pole/amp/volt/curve/NEMA/IP/lamp base). The cross-encoder is blind to both. `apply_attribute_sort` sorts by `(attribute matches, −conflicts)` via `attribute_relation` in `data/normalizers.py`: a doc that contradicts a queried attribute (15A when 20A asked) sinks below one merely silent on it. Both passes are no-ops when the query states no size/attribute token, so they're safe for every domain.

**Dense quantization.** The dense vectors are int8 scalar-quantized with originals on disk (`on_disk=True`, `always_ram=True` on the quantized copy) — set at collection creation in `ingest.py`. Every dense read in `core/search.py` attaches `_DENSE_QSP` (`rescore=True, oversampling=2.0`) so the quantized candidate pool is rescored against the full-precision originals. This is a RAM/cost lever for scale (~4× smaller in-RAM dense footprint), not an accuracy change — the eval A/B showed Recall@{5,10,50} unchanged. Quantization touches dense only; the BM25 sparse channels are untouched. Binary quantization is intentionally avoided (mpnet is 768d, below the ~1024d where binary holds up).

Two public entry points: `search()` (plain dicts, used by scripts/eval) and `search_with_observability()` (adds per-step timings, per-retriever attribution, and the full candidate pool — used by the Streamlit UI).

### Normalization (`data/normalizers.py`) — the core retrieval trick
This module is shared by **both** ingest and query encoding, and getting it right is what drives recall:
- `normalize_specs` collapses inconsistent size surface forms (`3/4IN`, `1-1/2"`, `50MM`, `2 inches`) into punctuation-free high-IDF anchor tokens (`size75`, `mm50`) so BM25 doesn't shatter them. Uses pint for unit math + a single regex scanner for spans.
- **Metric bridging (`bridge_metric=True`) is query-side ONLY.** Documents stay canonical at ingest; only the query cross-emits the other unit system (imperial↔metric). Don't enable it on the document side.
- `model_number_variants` expands a model number into casing/separator/alphanum variants for the `sparse_model` field.
- `spec_text_with_attributes` = `normalize_specs` output + canonical electrical-attribute anchors (`baseg24q3`, `nema4_1`, `pole1`, `curvec`, …). Used to build the `sparse_desc` field at both ingest (`scripts/ingest.py`) and query (`models/embeddings.py:encode_query`), so punctuation-heavy attributes BM25 would shatter become matchable. `attribute_anchor_tokens` / `attribute_relation` back the post-rerank `apply_attribute_sort`.
- `make_id` is a deterministic MD5 → UUID of `(source, internal_id)`, which makes ingest idempotent via upsert.

### Ingestion (`scripts/ingest.py`)
Each of the 9 raw files has a dedicated `load_*` function mapping that distributor's columns to a canonical record (the per-source column maps live inline in each loader). Rows are grouped by canonical ID; per-branch detail (stock/cost/price) is aggregated into a `locations[]` array with top-level rollups (`has_stock`, `total_qoh`, `min_cost`, `max_cost`). The same SKU from multiple distributors yields multiple points (one per `source`). Raw files go in `inventory_data/` (gitignored); override the dir with `$INVENTORY_DATA`.

## Gotchas

- **`ingest.py` creates the collection itself** (delete + recreate, wiping existing data) and is the source of truth for the live schema, including the int8 quantization config. `scripts/setup_collection.py` is a separate standalone schema script that `ingest.py` does NOT call; it's kept in sync by hand (both now set `on_disk=True`), so if you change the schema in one, mirror it in the other.
- **Backend selection differs between ingest and the app.** `ingest.py` uses cloud if `QDRANT_URL` is set, else falls back to a fixed `local_storage/` dir — it ignores `QDRANT_LOCAL_PATH`. The app/`core/client.py` uses `QDRANT_LOCAL_PATH` if set, else `QDRANT_URL`. For end-to-end local dev, unset `QDRANT_URL` for ingest and point `QDRANT_LOCAL_PATH` at the repo's `local_storage/`.
- **Dense encoder is a hand-rolled `_MpnetEncoder`, not `SentenceTransformer`.** This is deliberate: torch 2.8 + transformers leaves meta tensors that break `SentenceTransformer.__init__`, and Streamlit can trigger torch dynamo fake-tensor dispatch. The module sets `TORCHDYNAMO_DISABLE=1` and wraps the forward pass in `torch.compiler.disable`. Don't "simplify" it back to `SentenceTransformer`.
- **`scripts/evaluate.py` vs `scripts/run_eval.py`:** `run_eval.py` (CSV-based, the documented entry point) imports scoring helpers from `evaluate.py`. `evaluate.py`'s own `main()` expects an `eval_queries.json` that may not be present — prefer `run_eval.py`.
