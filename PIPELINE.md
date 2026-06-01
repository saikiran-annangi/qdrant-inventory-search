# Inventory Search — Enhanced Pipeline (Phase 2 + Phase 3)

Extends the baseline hybrid search with three additions:
1. **LLM-enriched descriptions** — Gemini generates rich product text for sparse descriptions
2. **Taxonomy classification at ingestion** — every product tagged with domain/category/subcategory
3. **LLM query taxonomy classifier** — two-stage Gemini call maps every query into existing taxonomy labels, enabling a reliable 4th RRF channel

---

## Eval results — mep_eval_300_v3 (300 queries, fuzzy ID matching)

| Metric | Overall | Model Number | Technical | Descriptive |
|--------|---------|--------------|-----------|-------------|
| MRR@3  | 0.6739  | 0.8148       | 0.5488    | 0.6585      |
| R@3    | 0.7267  | 0.8788       | 0.5960    | 0.7059      |
| R@10   | 0.7667  | 0.8788       | 0.6667    | 0.7549      |
| Miss@10| 23.3%   | —            | —         | —           |

Fuzzy ID matching recovers 31 false misses caused by internal_id format differences (doubled prefixes, hyphen variants). Raw strict matching gives R@10 = 0.667.

---

## Architecture overview

```
INGESTION (offline, one-time)              QUERY (real-time, per search)
─────────────────────────────              ──────────────────────────────
Raw product                                User query
    │                                           │
    ▼                                           ▼
LLM enrichment          ──────────────►  classify_query()
(extended_description)                    model_number / technical / descriptive
    │                                           │
    ▼                                           ▼
LLM attribute extraction                  Two-stage LLM taxonomy classifier
{domain, explicit, inferred}               Stage 1: domain (Electrical/Mechanical/Plumbing)
    │                                      Stage 2: category > subcategory
    ▼                                      (picks from labels that exist in collection)
Taxonomy assignment                             │
  Path A: cosine sim + cross-encoder            ▼
  Path B: LLM fallback (if A fails)       encode_query()
    │                                      dense + sparse_model + sparse_desc
    ▼                                           │
Encode vectors                                  ▼
dense + sparse_model + sparse_desc         Qdrant Prefetch (4 channels)
    │                                      Ch1: Dense
    ▼                                      Ch2: BM25-model
Qdrant upsert                              Ch3: BM25-desc
(35,989 points)                            Ch4: Dense + taxonomy filter  ← new
                                                │
                                                ▼
                                           RRF Fusion → top 50
                                                │
                                                ▼
                                           Cross-encoder Reranker → top 10
```

---

## Ingestion pipeline (detailed)

### Step 1 — Description enrichment (`scripts/enrich_descriptions.py`)

Products often have sparse descriptions like `"LEDHIGHBAY-150"` or just a model code. These carry no semantic meaning for the dense encoder.

**What it does:** Calls Gemini to generate a 2-3 sentence product description covering specs, application, and technology. Stored in `extended_description`.

**Why it matters:** The dense vector is built from `description + extended_description`. A rich LLM-generated description gives the all-mpnet encoder actual semantics to embed — wattage, application, product type — so queries like `"LED warehouse fixture"` find the right product instead of noise.

**Output:** `enrichment_cache.json` (32,362 entries)

---

### Step 2 — Attribute extraction (`scripts/build_attributes_cache.py`)

**What it does:** Calls Gemini 2.5 Flash (via OpenRouter) to extract structured attributes from each product:

```json
{
  "domain": "Electrical",
  "explicit": { "voltage": "120V", "wattage": "150W", "poles": "2" },
  "inferred": { "product_type": "circuit breaker", "mounting": "panel" }
}
```

**Why it matters:** Used as input to the taxonomy mapper (Step 3). Structured key-value pairs are more precise signal than raw description text for classification.

**Output:** `attributes_cache.json` (35,989 entries)

---

### Step 3 — Taxonomy classification (`scripts/build_taxonomy_cache.py`)

Every product gets assigned `taxonomy_domain`, `taxonomy_category`, `taxonomy_subcategory`. Two paths:

**Path A — Cosine similarity (no LLM cost)**

1. Embed each attribute text (`"voltage: 120V"`, `"product_type: breaker"`) with all-mpnet
2. Cosine similarity against pre-defined taxonomy node vectors, filtered by domain
3. RRF across all attributes → top-3 candidates
4. Cross-encoder rerank (ms-marco) → pick best node
5. If confidence ≥ −5.5 → assign predefined taxonomy, `taxonomy_source: "cosine"`

**Path B — LLM fallback (for products that fail Path A)**

Products with short/generic attributes, unknown domain, or poor cosine matches fall through to:

```
Input to Gemini 2.5 Flash:
  Model Number: FORTIS-LEDHIGHBAY-150
  Description: LED High Bay 150W...
  Extended Description: warehouse fixture, IP65...
  Manufacturer: Fortis
  Extracted Attributes: wattage:150W, product_type:LED fixture

  + 213 predefined nodes shown as style reference only
    (LLM is told to INVENT a new label, not reuse existing ones)

Output:
  { domain: "Electrical",
    category: "Luminaires & Lighting Controls",
    subcategory: "High-Bay Luminaires" }
  taxonomy_source: "llm_fallback"
```

**Why show predefined nodes to LLM fallback:** Ensures stylistic consistency. LLM-invented labels follow the same format (2-5 words, title case) as predefined ones, preventing fragmented vocabulary.

**Coverage result:** 8.5% → **100%** (35,989 / 35,989 products fully classified)

**Output:** `taxonomy_cache.json` (35,989 entries with `taxonomy_source` field)

---

### Step 3b — Rebuild taxonomy embeddings (`scripts/build_taxonomy_embeddings.py`)

After Step 3, there are now thousands of LLM-invented taxonomy labels in addition to 213 predefined ones. The query-side classifier needs to know all of them.

**What it does:** Reads all unique `(domain, category, subcategory)` combinations from `taxonomy_cache.json`, embeds each with all-mpnet, saves to `taxonomy_embeddings.json`.

- Before: 213 predefined nodes
- After: **10,546 nodes** (213 predefined + 10,333 LLM-invented)

---

### Step 4 — Ingest (`scripts/ingest.py`)

For each product, reads all three caches and builds the Qdrant point:

```python
payload = {
    "model_number":         "FORTIS-LEDHIGHBAY-150",
    "description":          "LED High Bay 150W...",
    "extended_description": "Warehouse-grade LED fixture...",  # from enrichment
    "manufacturer_name":    "Fortis",
    "taxonomy_domain":      "Electrical",                      # from taxonomy
    "taxonomy_category":    "Luminaires & Lighting Controls",
    "taxonomy_subcategory": "High-Bay Luminaires",
    "has_stock":            True,
    "total_qoh":            24,
    ...
}

vectors = {
    "dense":        all_mpnet(description + extended_description),
    "sparse_model": bm25(model_number_variants),
    "sparse_desc":  bm25(description + extended_description),
}
```

---

## Query pipeline (detailed)

### Step 1 — Query classification (`models/classifier.py`)

Rule-based joblib classifier assigns one of three types:

| Type | Example | Behaviour |
|------|---------|-----------|
| `model_number` | `"Q2T3225"` | BM25-model dominant, no taxonomy |
| `technical` | `"20A 2-pole breaker 120V"` | Balanced dense + BM25 |
| `descriptive` | `"LED fixture for warehouse"` | BM25-desc + dense dominant |

Adjusts prefetch limits per type (e.g. `sparse_desc` is 0 for model_number queries).

---

### Step 2 — Two-stage LLM taxonomy classifier (`models/query_taxonomy_llm.py`)

**Why replace the old embedding classifier:**

The old classifier embedded the query and did cosine similarity against 213 node labels. Short label strings like `"High-bay"` score poorly with the ms-marco cross-encoder trained on long web passages. Most queries scored below the −4.0 confidence threshold → Ch4 never fired.

**New approach — two stages:**

**Stage 1 — Domain** (~50 token prompt, ~80ms)
```
Query: "LED high-bay fixture for warehouse"
Options: Electrical / Mechanical / Plumbing / Unknown
→ "Electrical"
```

**Stage 2 — Category/Subcategory** (~500 token prompt, ~100ms)
```
Available Electrical nodes:
  Luminaires & Lighting Controls > High-Bay Luminaires
  Luminaires & Lighting Controls > LED Troffers
  Conductors, Cable & Raceways > Building Wire
  ... (8,595 Electrical labels, domain-filtered)

Query: "LED high-bay fixture for warehouse"
Pick from the list above — use exact text, do not invent.
→ { category: "Luminaires & Lighting Controls",
    subcategory: "High-Bay Luminaires" }
```

**Why domain-filtered for Stage 2:** Passing all 10,546 nodes per query would be ~50,000 tokens. Filtering to one domain reduces to ~500 tokens. 100x cheaper and faster.

**Why LLM must pick from existing labels:** Ch4 is an exact string match filter in Qdrant. An invented label like `"High Bay Fixtures"` won't match products tagged `"High-Bay Luminaires"`. Loading the label list from `taxonomy_cache.json` guarantees every label the LLM returns exists in the collection.

**Why no confidence threshold:** The old classifier had a −4.0 threshold that most queries failed. The LLM always returns a result, so Ch4 fires on every non-model-number query.

**Result:**
- `{ taxonomy_domain: "Electrical", taxonomy_category: "Luminaires & Lighting Controls", taxonomy_subcategory: "High-Bay Luminaires" }`

---

### Step 3 — Query encoding (`models/embeddings.py`)

Produces three vectors from the raw query text:
- `dense_vec` — all-mpnet-base-v2 (768d)
- `sparse_model_vec` — BM25 over model number variants
- `sparse_desc_vec` — BM25 over normalized description text

---

### Step 4 — Qdrant prefetch (4 channels)

```python
prefetch = [
    # Ch1: Semantic similarity
    Prefetch(query=dense_vec, using="dense", limit=20),

    # Ch2: Exact model number / ERP code match
    Prefetch(query=sparse_model_vec, using="sparse_model", limit=50),

    # Ch3: Keyword match in description + extended_description
    Prefetch(query=sparse_desc_vec, using="sparse_desc", limit=80),

    # Ch4: Taxonomy channel (NEW)
    # Dense retrieval filtered to only products in the same category
    Prefetch(
        query=dense_vec,
        using="dense",
        limit=50,
        filter=Filter(must=[
            FieldCondition(
                key="taxonomy_subcategory",
                match=MatchValue(value="High-Bay Luminaires")
            )
        ])
    ),
]
```

**How Ch4 helps:** Without it, a product described only as `"Industrial LED Fixture 150W IP65"` (no mention of "high-bay" or "warehouse") would be missed by BM25 and ranked low by dense. Ch4 finds ALL products tagged `High-Bay Luminaires` and re-ranks them by semantic similarity to the query — surfacing correctly-categorised products regardless of their description text.

**Subcategory → category fallback:** If subcategory confidence is low, the filter falls back to `taxonomy_category`. Broader coverage, less precision — better than no filter.

**Why soft (RRF) not hard filter:** A hard filter would exclude products outside the category entirely, potentially missing the correct product if taxonomy was mis-assigned. RRF just boosts — Ch4 products get extra ranking credit, products from Ch1-3 still appear.

---

### Step 5 — RRF fusion

All 4 channel results merged with Reciprocal Rank Fusion (k=60). A product appearing in multiple channels gets a higher combined score. Returns top 50 candidates.

---

### Step 6 — Cross-encoder reranker (`models/reranker.py`)

ms-marco-MiniLM-L-6-v2 cross-encoder scores `(query, product_description)` pairs for all 50 candidates. Returns final top 10.

**Why rerank last:** BM25 and dense are fast but imprecise. The cross-encoder reads both query and product text together and produces a true relevance score. Running it on 50 candidates instead of 35,989 keeps latency acceptable (~200ms).

---

## Decision log

| Decision | Reason |
|----------|--------|
| LLM enrichment for sparse descriptions | Model codes have no semantic content. Dense retrieval is useless without real text to embed. |
| Attribute extraction before taxonomy | Structured `{voltage: 120V, product_type: breaker}` is better input for classification than raw description text. |
| Cosine first, LLM fallback second | Cosine is free (no API cost). Only pay for LLM when cosine fails. Minimises cost while maximising coverage. |
| Show predefined nodes to fallback LLM as style reference | Ensures invented labels are stylistically consistent. Prevents vocabulary fragmentation. |
| `taxonomy_source` field in cache | Track which path (cosine vs LLM) assigned each product's taxonomy for debugging and future normalization. |
| Rebuild taxonomy embeddings after Step 3 | Query side needs a complete list of all labels in the collection — not just 213 predefined. |
| Two-stage query taxonomy (domain then category) | Domain classification costs ~50 tokens. Filtering to one domain reduces Stage 2 from 50,000 tokens to 500 tokens. |
| Query LLM picks from existing labels only | Ch4 is exact string match. Invented labels return zero results. |
| Ch4 as RRF channel (soft) not hard filter | Hard filter risks missing correct products due to taxonomy mis-assignment. Soft signal boosts without excluding. |
| Subcategory → category fallback | Subcategory is precise but has fewer products. Category provides broader coverage when subcategory confidence is low. |

---

## Repository structure (updated)

```
├── config.py                        Constants, taxonomy definition, thresholds
├── app.py                           Streamlit UI with observability panel
│
├── core/
│   ├── client.py                    Qdrant singleton
│   ├── filters.py                   build_filter()
│   └── search.py                    search() and search_with_observability()
│
├── models/
│   ├── classifier.py                Query type classifier (model_number/technical/descriptive)
│   ├── embeddings.py                Dense + BM25 encoders
│   ├── reranker.py                  Cross-encoder reranker
│   ├── extractor.py                 LLM attribute extraction (Gemini 2.5 Flash)
│   ├── taxonomy_mapper.py           Cosine+cross-encoder taxonomy mapper + LLM fallback
│   ├── query_taxonomy.py            Old embedding-based query classifier (superseded)
│   └── query_taxonomy_llm.py        Two-stage LLM query taxonomy classifier (active)
│
├── data/
│   ├── normalizers.py               Spec expansion, model variants
│   └── loader.py                    Source loaders + load_all()
│
├── scripts/
│   ├── setup_collection.py          Create Qdrant collection (run once)
│   ├── ingest.py                    Embed and upsert all products
│   ├── build_attributes_cache.py    LLM attribute extraction for all products
│   ├── build_taxonomy_cache.py      Taxonomy classification (cosine + LLM fallback)
│   ├── build_taxonomy_embeddings.py Embed all taxonomy node labels
│   ├── evaluate.py                  Original 90-query eval
│   └── evaluate_improvements.py     28-query improvement eval + mep_eval_300_v3
│
├── eval_improvements.json           28 targeted queries (enriched desc + taxonomy)
├── enrichment_cache.json            gitignored — regenerate with enrich_descriptions.py
├── attributes_cache.json            gitignored — regenerate with build_attributes_cache.py
├── taxonomy_cache.json              gitignored — regenerate with build_taxonomy_cache.py
└── taxonomy_embeddings.json         gitignored — regenerate with build_taxonomy_embeddings.py
```

---

## Setup from scratch

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Add OPENROUTER_API_KEY (for LLM calls)
# Add QDRANT_URL (default: http://localhost:6333)

# 3. Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# 4. Place inventory CSV/XLSX files in inventory_data/

# 5. Build caches (one-time, run in order)
python scripts/build_attributes_cache.py     # ~30 min, LLM calls
python scripts/build_taxonomy_cache.py       # ~10 min, cosine + LLM fallback
python scripts/build_taxonomy_embeddings.py  # ~1 min, local

# 6. Create collection + ingest
python scripts/setup_collection.py --confirm-drop
python scripts/ingest.py                     # ~8 min

# 7. Run the app
streamlit run app.py
```

## Run evaluations

```bash
# Full eval (improvement queries + mep_eval_300_v3)
python scripts/evaluate_improvements.py

# mep_eval_300_v3 only (faster with parallel workers)
python scripts/evaluate_improvements.py --mep-only

# Improvement queries only
python scripts/evaluate_improvements.py --improvements-only
```
