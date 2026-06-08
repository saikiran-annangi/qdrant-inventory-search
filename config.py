"""
Central configuration for the inventory search system.
All constants, paths, and tuning parameters live here.
"""

import os

# Load .env file when present (OPENROUTER_API_KEY, QDRANT_URL, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", None)
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "")
COLLECTION_NAME   = "inventory"
DENSE_DIM         = 768  # all-mpnet-base-v2 output dimension

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

DENSE_MODEL_NAME    = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_MODEL_NAME     = "Qdrant/bm25"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw inventory CSV/XLSX files go here (gitignored)
DATA_DIR = os.path.join(REPO_ROOT, "inventory_data")

# Generated cache files — rebuild using scripts in scripts/
ENRICHMENT_CACHE_PATH    = os.path.join(REPO_ROOT, "enrichment_cache.json")
TAXONOMY_EMBEDDINGS_PATH = os.path.join(REPO_ROOT, "taxonomy_embeddings.json")
TAXONOMY_CACHE_PATH      = os.path.join(REPO_ROOT, "taxonomy_cache.json")
# Open, self-growing taxonomy store (seeded from PRODUCT_TAXONOMY, grows at
# ingest time). taxonomy_store.json holds nodes+embeddings (ingest side);
# taxonomy_labels.json is the labels-only projection the query side reads.
TAXONOMY_STORE_PATH      = os.path.join(REPO_ROOT, "taxonomy_store.json")
TAXONOMY_LABELS_PATH     = os.path.join(REPO_ROOT, "taxonomy_labels.json")
# Confidence gates for the open vocabulary (see data/taxonomy_store.py).
TAXONOMY_ASSIGN_THRESHOLD = 0.55   # product↔node cosine to assign an existing node
TAXONOMY_DEDUP_THRESHOLD  = 0.86   # new-node↔existing cosine to reuse instead of mint

# ---------------------------------------------------------------------------
# Retrieval tuning: prefetch limits per query type
# ---------------------------------------------------------------------------
# dense        -- semantic embedding (all-mpnet, int8-quantized)
# sparse_model -- BM25 over model number variants
# sparse_desc  -- BM25 over normalized description + spec attribute anchors
# ---------------------------------------------------------------------------

PREFETCH_LIMITS = {
    # model_number: sparse_model is the primary signal. Dense kept small for
    # tokenization variants. sparse_desc excluded — product family siblings
    # share descriptions and would push incorrect siblings to rank 1.
    "model_number": {"dense": 10, "sparse_model": 80, "sparse_desc": 0},
    "technical":    {"dense": 50, "sparse_model": 50, "sparse_desc": 40},
    "descriptive":  {"dense": 20, "sparse_model": 50, "sparse_desc": 80},
    # Single fixed profile used when the classifier is bypassed. Balanced
    # across all three channels — A/B across two held-out 300-query eval sets
    # showed this matches/beats per-type routing while removing ~600ms latency
    # and the OpenRouter dependency.
    "default":      {"dense": 50, "sparse_model": 50, "sparse_desc": 40},
}

# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------
# When USE_CLASSIFIER=False (the default), every query uses DEFAULT_PROFILE
# and no OpenRouter call is made. Set USE_CLASSIFIER=True to restore per-type
# routing via Gemini 2.5 Flash.
# ---------------------------------------------------------------------------

USE_CLASSIFIER  = False
DEFAULT_PROFILE = "default"

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

INGEST_BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Product taxonomy — domain → category → [subcategories]
# Used by models/query_taxonomy_llm.py (query side) and
# scripts/build_taxonomy_embeddings.py / build_taxonomy_from_descriptions.py
# (ingest side).
# ---------------------------------------------------------------------------

# The product taxonomy is the SINGLE SOURCE OF TRUTH and lives in its own
# module (data/taxonomy.py). The query classifier, the embedding builder, the
# product classifier, and the search-time boost all derive from it — so they
# can never drift apart. CATEGORY_MAP gives a deterministic ERP-category ->
# taxonomy-node mapping for products that already carry a category.
from data.taxonomy import PRODUCT_TAXONOMY, CATEGORY_MAP  # noqa: E402,F401
