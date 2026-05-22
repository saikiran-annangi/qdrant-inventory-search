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

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "inventory"
DENSE_DIM = 768  # all-mpnet-base-v2 output dimension

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

DENSE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_MODEL_NAME = "Qdrant/bm25"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw inventory CSVs / XLSXs go here (gitignored)
DATA_DIR = os.path.join(REPO_ROOT, "inventory_data")

# Trained LR classifier (produced by scripts/build_classifier.py)
CLASSIFIER_PATH = os.path.join(REPO_ROOT, "query_classifier.joblib")

# ---------------------------------------------------------------------------
# Retrieval tuning: prefetch limits per query type
# ---------------------------------------------------------------------------
# dense       -- semantic embedding (all-mpnet)
# sparse_model -- BM25 over model number variants
# sparse_desc  -- BM25 over normalized description + specs
# ---------------------------------------------------------------------------

PREFETCH_LIMITS = {
    "model_number": {"dense": 80, "sparse_model": 80, "sparse_desc": 20},
    "technical":    {"dense": 50, "sparse_model": 50, "sparse_desc": 40},
    "descriptive":  {"dense": 20, "sparse_model": 50, "sparse_desc": 80},
}

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

INGEST_BATCH_SIZE = 256
