# Latest Version — Full Pipeline

```
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                  CACHE BUILDING  ·  one-time before enriched ingest                         ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────────────────────┐
  │  RAW SOURCE FILES  (CSV / XLSX)                                                          │
  │  guillevin_1/2  ·  burnaby_dc  ·  au_parspec  ·  inventory_sample  ·  standard_supply   │
  │  plumbing  ·  plumbing_2                                                                 │
  └──────────────────────────────┬──────────────────────────────────────────────────────────┘
                                 │
               ┌─────────────────┴──────────────────┐
               │                                     │
               ▼                                     ▼
  ┌────────────────────────────┐       ┌─────────────────────────────────────────────────┐
  │  enrich_descriptions.py    │       │  build_taxonomy_embeddings.py                   │
  │  ─────────────────────     │       │  ──────────────────────────                     │
  │  load_all() → ~45k prods   │       │  Read PRODUCT_TAXONOMY from config.py           │
  │                            │       │  213 predefined nodes                           │
  │  _needs_enrichment(desc)?  │       │  (Electrical / Mechanical / Plumbing)           │
  │  ├─ < 5 words   → YES      │       │                                                 │
  │  ├─ 5-15 words             │       │  Embed each node label with all-mpnet-base-v2   │
  │  │  >60% UPPERCASE → YES   │       │  → 768-dim vector per node                      │
  │  └─ rich text   → SKIP     │       │                                                 │
  │                            │       │  Save → taxonomy_embeddings.json                │
  │  Already in cache → SKIP   │       │  ~1 min · no API key needed                     │
  │                            │       └──────────────────┬──────────────────────────────┘
  │  Per flagged product:       │                         │
  │  ┌──────────────────────┐  │                         ▼
  │  │ Gemini 2.5 Flash     │  │       ┌─────────────────────────────────────────────────┐
  │  │ via OpenRouter        │  │       │  build_taxonomy_from_descriptions.py            │
  │  │                      │  │       │  ────────────────────────────────────            │
  │  │ Input (compact):      │  │       │  load_all(attach_caches=False)                  │
  │  │  desc | mfr |         │  │       │                                                 │
  │  │  model | category     │  │       │  Step 1 — Domain inference (keyword match)      │
  │  │                      │  │       │  category + description text                    │
  │  │ System prompt cached │  │       │  count hits: Electrical / Mechanical / Plumbing │
  │  │ Rules: expand abbrev,│  │       │  highest count wins → domain assigned           │
  │  │ lowercase, factual,  │  │       │                                                 │
  │  │ 1-2 sentences        │  │       │  Step 2 — Embed product text (batched)          │
  │  │                      │  │       │  text = desc | ext_desc | mfr | category        │
  │  │ Output:              │  │       │  all-mpnet → 768-dim per product                │
  │  │  extended_description│  │       │  batch_size=256                                 │
  │  └──────────────────────┘  │       │                                                 │
  │                            │       │  Step 3 — Top-3 cosine candidates               │
  │  retry once if bad output  │       │  domain_matrix @ product_vecs.T                 │
  │  fallback: original desc   │       │  argsort → top 3 node indices per product       │
  │                            │       │                                                 │
  │  150 threads concurrently  │       │  Step 4 — Cross-encoder rerank                  │
  │  checkpoint save /100 prods│       │  score_pairs([(prod_text, node_text) × 3])      │
  │  ~1 hour · ~37,498 entries │       │  float64 CE → logit per candidate               │
  │                            │       │  best = argmax(logits)                          │
  │  Save →                    │       │                                                 │
  │  enrichment_cache.json     │       │  Step 5 — Confidence threshold                  │
  │  { uuid: ext_description } │       │  logit ≥ -5.5 → domain+category+subcategory     │
  └────────────┬───────────────┘       │  logit  < -5.5 → domain only (blank cat/subcat) │
               │                       │  unknown domain → all blank                     │
               │                       │                                                 │
               │                       │  Save →                                         │
               │                       │  taxonomy_cache.json                            │
               │                       │  { uuid: {domain,category,subcategory,score} }  │
               │                       │  ~5 min · ~38,094 entries                       │
               │                       └──────────────────┬──────────────────────────────┘
               │                                          │
               └──────────────────┬───────────────────────┘
                                  │
                                  ▼  both caches now exist on disk


╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                         INGESTION PIPELINE  ·  scripts/ingest.py                            ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────────────────────┐
  │  RAW SOURCE FILES  (CSV / XLSX)                                                          │
  └──────────────────────────────────────────────────┬──────────────────────────────────────┘
                                                     │
                                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  data/loaders.load_all(attach_caches=True)                                               │
  │                                                                                          │
  │  Parse all CSV/XLSX  →  aggregate branch rows into locations[] per product               │
  │                                                                                          │
  │       enrichment_cache.json exists?                taxonomy_cache.json exists?           │
  │       ┌──────────┬────────────────┐               ┌──────────┬────────────────────┐     │
  │       │  YES     │  NO            │               │  YES     │  NO                │     │
  │       │  merge   │  ext_desc=None │               │  merge   │  taxonomy=None     │     │
  │       │  ext_desc│                │               │  domain/ │                    │     │
  │       │  per UUID│                │               │  cat/sub │                    │     │
  │       └──────────┴────────────────┘               └──────────┴────────────────────┘     │
  └──────────────────────────────────────────────────┬───────────────────────────────────────┘
                                                     │
                                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  BUILD 3 TEXT VARIANTS per product                                                       │
  │                                                                                          │
  │  dense_text ──── description + extended_description (if cached) + manufacturer           │
  │                  + product_category + model_number                                       │
  │                  (richer semantic context for embedding)                                 │
  │                                                                                          │
  │  sm_text ──────  model_number_variants()                                                 │
  │                  casing variants · separator variants · alphanum variants                │
  │                  e.g.  "A9F74116" → "a9f74116" "A9F-74116" "A9F 74116"                  │
  │                                                                                          │
  │  sd_text ──────  spec_text_with_attributes(description, bridge_metric=False)             │
  │                  normalise size units → anchor tokens  size75  mm50                      │
  │                  electrical anchors → amp16  volt125  pole3  curvec  nema4x  ip66        │
  └──────────────────────────────────────────────────┬───────────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────┼────────────────────────────────┐
                    │                                │                                │
                    ▼                                ▼                                ▼
       ┌────────────────────────┐    ┌───────────────────────────┐   ┌───────────────────────────┐
       │  Dense Encoder          │    │  BM25  sparse_model       │   │  BM25  sparse_desc        │
       │  all-mpnet-base-v2      │    │  Qdrant/bm25 FastEmbed    │   │  Qdrant/bm25 FastEmbed    │
       │  AutoModel (not ST)     │    │                           │   │                           │
       │  tokenise → forward     │    │  embed  sm_text           │   │  embed  sd_text           │
       │  mean-pool + attn mask  │    │  → sparse vector          │   │  → sparse vector          │
       │  L2-normalise           │    │  (indices + values)       │   │  (indices + values)       │
       │  → 768-dim float32      │    └───────────────────────────┘   └───────────────────────────┘
       └────────────────────────┘
                    │                                │                                │
                    └────────────────────────────────┼────────────────────────────────┘
                                                     │
                                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  QDRANT COLLECTION SETUP  (delete + recreate "inventory")                                │
  │                                                                                          │
  │  dense vector ─────  768-dim · COSINE distance                                           │
  │                       HNSW  m=16  ef_construct=200                                       │
  │                       INT8 scalar quantization                                           │
  │                         quantile=0.99  always_ram=True  on_disk=True                     │
  │                         → int8 copy pinned in RAM   (fast ANN search)                    │
  │                         → float32 originals on disk  (accurate rescore)                  │
  │                                                                                          │
  │  sparse_model ─────  in-memory BM25 index                                                │
  │  sparse_desc  ─────  in-memory BM25 index                                                │
  │                                                                                          │
  │  payload indexes ──  source · manufacturer · category · currency · has_stock             │
  │                       taxonomy_domain · taxonomy_category · taxonomy_subcategory          │
  └──────────────────────────────────────────────────┬───────────────────────────────────────┘
                                                     │
                                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  UPSERT  ·  batch size 256                                                               │
  │                                                                                          │
  │  id  =  make_id(source, internal_id)  →  deterministic MD5 → UUID  (idempotent)         │
  │  vectors:   dense  +  sparse_model  +  sparse_desc                                       │
  │  payload:   all product fields  +  taxonomy fields  +  locations[]                       │
  │                                                                                          │
  │  ~45,280 points total                                                                    │
  └──────────────────────────────────────────────────────────────────────────────────────────┘


╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                           QUERY PIPELINE  ·  core/search.py                                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  RAW QUERY STRING                                                                        │
  │  e.g.  "Schneider 16A single pole MCB"   or   "K-2084"                                  │
  └──────────────────────────────────────────────┬─────────────────────────────────────────-─┘
                                                 │
                                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  STAGE 1 — Query Type Classification   ·   models/classifier.py                         │
  │                                                                                          │
  │  USE_CLASSIFIER = False  (default)                                                       │
  │  └─  every query → "default" profile  ·  0 ms  ·  no API call                           │
  │                                                                                          │
  │  USE_CLASSIFIER = True                                                                   │
  │  └─  Gemini 2.5 Flash via OpenRouter                                                     │
  │      → model_number  |  technical  |  descriptive                                        │
  └──────────────────────────────────────────────┬─────────────────────────────────────────-─┘
                                                 │
                                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  STAGE 2 — Taxonomy Classification (LLM)   ·   models/query_taxonomy_llm.py             │
  │                                                                                          │
  │  query_type = model_number  →  SKIP  (return {})                                        │
  │  query_type = descriptive   →  SKIP  (vague queries → unreliable prediction)            │
  │  query looks like model no  →  SKIP  (pn: / stk no. / cat# prefix patterns)            │
  │  no OPENROUTER_API_KEY      →  SKIP  (return {})                                        │
  │                                                                                          │
  │  query_type = technical  →  2 Gemini Flash calls via OpenRouter:                        │
  │                                                                                          │
  │   Call 1 — Stage 1 (domain)                                                             │
  │   ┌─────────────────────────────────────────────────────────┐                           │
  │   │ "Classify query into: Electrical / Mechanical /          │                           │
  │   │  Plumbing / Unknown"                                     │                           │
  │   │  → { "domain": "Electrical" }                           │                           │
  │   └─────────────────────────────────────────────────────────┘                           │
  │                                                                                          │
  │   Call 2 — Stage 2 (category / subcategory)                                             │
  │   ┌─────────────────────────────────────────────────────────┐                           │
  │   │ "Pick best node from domain-filtered label list          │                           │
  │   │  (labels that exist in taxonomy_cache.json only —       │                           │
  │   │  no invented labels)"                                    │                           │
  │   │  → { "taxonomy_category":    "Circuit Breakers",        │                           │
  │   │       "taxonomy_subcategory": "MCB" }                   │                           │
  │   └─────────────────────────────────────────────────────────┘                           │
  │                                                                                          │
  │  Returns: { taxonomy_domain, taxonomy_category, taxonomy_subcategory }                  │
  └──────────────────────────────────────────────┬─────────────────────────────────────────-─┘
                                                 │
                                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  STAGE 3 — Encode Query into 3 Vectors   ·   models/embeddings.py                       │
  │                                                                                          │
  │  dense_vec  ───   all-mpnet-base-v2 on raw query text                                   │
  │                   → 768-dim float32  (mean-pool + L2-norm)                               │
  │                                                                                          │
  │  sparse_model_vec  ──  BM25 on  model_number_variants(query)                            │
  │                        → sparse vector (indices + values)                               │
  │                                                                                          │
  │  sparse_desc_vec  ───  BM25 on  spec_text_with_attributes(query, bridge_metric=True)    │
  │                        bridge_metric=True ← QUERY SIDE ONLY                             │
  │                        imperial query → also emits metric anchor tokens                  │
  │                        e.g. "3/4 inch" → size75 + mm19  (±1mm window)                  │
  └──────────────────────────────────────────────┬─────────────────────────────────────────-─┘
                                                 │
                                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  STAGE 4 — 3-Channel Qdrant Prefetch                                                     │
  │                                                                                          │
  │  Prefetch limits  (from config.PREFETCH_LIMITS[query_type]):                             │
  │                                                                                          │
  │   profile        dense   sparse_model   sparse_desc                                     │
  │   ──────────     ─────   ────────────   ───────────                                     │
  │   default          50         50             40    ← USE_CLASSIFIER=False                │
  │   technical        50         50             40                                          │
  │   model_number     10         80              0    ← BM25 model is primary               │
  │   descriptive      20         50             80    ← BM25 desc is primary                │
  │                                                                                          │
  │  Dense channel uses:                                                                     │
  │    QuantizationSearchParams(rescore=True, oversampling=2.0)                             │
  │    → 2× candidates fetched from int8 index                                              │
  │    → rescored against on-disk float32 originals  (accuracy recovery)                    │
  │                                                                                          │
  │  Sparse channels:  standard BM25 retrieval (no quantization)                            │
  └──────────────────────────────────────────────┬─────────────────────────────────────────-─┘
                                                 │
                                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  STAGE 5 — Server-Side RRF Fusion   ·   Qdrant FusionQuery                              │
  │                                                                                          │
  │  Reciprocal Rank Fusion merges all 3 channel result lists                               │
  │  score(item) = Σ  1 / (rank_in_channel + 60)   across channels that retrieved it        │
  │                                                                                          │
  │  Output:  ranked pool of ~50–140 candidates with rrf_score + rrf_rank                   │
  └──────────────────────────────────────────────┬─────────────────────────────────────────-─┘
                                                 │
                                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  STAGE 6 — Cross-Encoder Rerank   ·   models/reranker.py                                │
  │                                                                                          │
  │  Model:  cross-encoder/ms-marco-MiniLM-L-6-v2  (6-layer MiniLM, ~22M params)           │
  │  Loaded with:  torch_dtype=torch.float64  ← avoids float32 NaN bug torch 2.12+ on CPU  │
  │                                                                                          │
  │  Input per pair:                                                                         │
  │    query  ⊕  description + extended_description + manufacturer + model + category       │
  │                                                                                          │
  │  Forward pass → logit per hit  (range ~−12 to +12)                                      │
  │  Resort descending by logit                                                              │
  └──────────────────────────────────────────────┬─────────────────────────────────────────-─┘
                                                 │
                                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  STAGE 7a — Taxonomy Boost   ·   apply_taxonomy_boost()                                 │
  │                                                                                          │
  │  taxonomy_result empty (model_number / descriptive / no key)  →  pass-through           │
  │                                                                                          │
  │  taxonomy_result present (technical query with API key):                                │
  │    subcategory match  →  CE logit  +0.8                                                 │
  │    category-only match →  CE logit  +0.2                                                │
  │    no match           →  no change                                                      │
  │                                                                                          │
  │  Items never excluded — wrong prediction just fails to boost                            │
  │  Re-sort by updated CE logit                                                             │
  └──────────────────────────────────────────────┬─────────────────────────────────────────-─┘
                                                 │
                                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  STAGE 7b — Size Sort   ·   apply_size_sort()                                           │
  │                                                                                          │
  │  Extract size anchors from query  (size75  mm50  etc.)                                  │
  │                                                                                          │
  │  Tier 1  ──  exact size match between query and product                                 │
  │  Tier 2  ──  no size signal in query or product                                         │
  │  Tier 3  ──  size conflict (query has 3/4" but product is 1")                           │
  │                                                                                          │
  │  CE score order preserved within each tier                                               │
  └──────────────────────────────────────────────┬─────────────────────────────────────────-─┘
                                                 │
                                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  STAGE 7c — Attribute Sort   ·   apply_attribute_sort()                                 │
  │                                                                                          │
  │  Extract electrical attribute anchors from query:                                        │
  │    pole  ·  amp  ·  volt  ·  curve  ·  NEMA  ·  IP rating  ·  lamp base                │
  │                                                                                          │
  │  Tier 1  ──  more attribute matches                                                      │
  │  Tier 2  ──  fewer / no attribute signals                                               │
  │  Tier 3  ──  attribute conflicts                                                         │
  │                                                                                          │
  │  CE score order preserved within each tier                                               │
  └──────────────────────────────────────────────┬─────────────────────────────────────────-─┘
                                                 │
                                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  OUTPUT  ·  Top-10 Final Results                                                         │
  │                                                                                          │
  │  Each hit contains:                                                                      │
  │    rank · CE score · RRF score · RRF rank · retrieval_path                              │
  │    dense_score · sparse_model_score · sparse_desc_score                                  │
  │    description · extended_description · taxonomy fields · raw_payload                    │
  │                                                                                          │
  │  search()                    →  plain dicts          (eval scripts)                     │
  │  search_with_observability() →  7-tuple with timings  (UI / app.py)                     │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
```
