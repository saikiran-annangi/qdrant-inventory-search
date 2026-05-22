"""
Embed all inventory records and upsert them into Qdrant.

Run after setup_collection.py:
    python scripts/ingest.py

Ingestion is idempotent -- re-running overwrites existing points via upsert
because point IDs are deterministic hashes of (source, internal_id).

Vectors generated:
  dense        -- all-mpnet-base-v2 (768d) on description + manufacturer + category + model
  sparse_model -- FastEmbed BM25 on model number variants
  sparse_desc  -- FastEmbed BM25 on spec-normalized description + manufacturer + category
"""

import os
import sys
import time
import uuid
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client.models import PointStruct, SparseVector

from config import COLLECTION_NAME, DENSE_MODEL_NAME, BM25_MODEL_NAME, INGEST_BATCH_SIZE
from core.client import get_client
from data.loader import load_all


def _load_dense_model():
    from sentence_transformers import SentenceTransformer
    print(f"Loading dense model: {DENSE_MODEL_NAME}...")
    return SentenceTransformer(DENSE_MODEL_NAME, device="cpu")


def _load_bm25_model():
    from fastembed import SparseTextEmbedding
    print("Loading BM25 model...")
    return SparseTextEmbedding(model_name=BM25_MODEL_NAME)


def _embed_dense(model, texts: list) -> list:
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return vecs.tolist()


def _embed_sparse(bm25_model, texts: list) -> list:
    return [
        SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
        for r in bm25_model.embed(texts)
    ]


def _dense_text(record: dict) -> str:
    parts = [
        record.get("description", ""),
        record.get("manufacturer_name", ""),
        record.get("product_category", ""),
        record.get("model_number", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _to_payload(record: dict) -> dict:
    """Strip internal helper fields before storing as Qdrant payload."""
    return {k: v for k, v in record.items() if not k.startswith("_") and k != "id"}


def ingest(sources: list = None, verbose: bool = True) -> int:
    print("Loading inventory data...")
    records = load_all(sources=sources, verbose=verbose)
    print(f"Total records to ingest: {len(records)}")

    dense_model = _load_dense_model()
    bm25_model  = _load_bm25_model()
    client      = get_client()

    total    = len(records)
    ingested = 0
    t0       = time.time()

    for batch_start in range(0, total, INGEST_BATCH_SIZE):
        batch = records[batch_start: batch_start + INGEST_BATCH_SIZE]

        dense_texts      = [_dense_text(r) for r in batch]
        bm25_model_texts = [r.get("_bm25_model", "") or "" for r in batch]
        bm25_desc_texts  = [r.get("_bm25_desc",  "") or "" for r in batch]

        dense_vecs        = _embed_dense(dense_model, dense_texts)
        sparse_model_vecs = _embed_sparse(bm25_model, bm25_model_texts)
        sparse_desc_vecs  = _embed_sparse(bm25_model, bm25_desc_texts)

        points = [
            PointStruct(
                id=str(uuid.UUID(record["id"])),
                vector={
                    "dense":        dense_vecs[i],
                    "sparse_model": sparse_model_vecs[i],
                    "sparse_desc":  sparse_desc_vecs[i],
                },
                payload=_to_payload(record),
            )
            for i, record in enumerate(batch)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        ingested += len(batch)

        elapsed = time.time() - t0
        rate    = ingested / elapsed if elapsed > 0 else 0
        eta     = (total - ingested) / rate if rate > 0 else 0

        if verbose:
            print(f"  [{ingested:>6}/{total}]  {rate:.0f} rec/s  ETA {eta:.0f}s", end="\r", flush=True)

    elapsed = time.time() - t0
    print(f"\nIngested {ingested} records in {elapsed:.1f}s ({ingested / elapsed:.0f} rec/s)")

    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' now has {info.points_count} points")
    return ingested


if __name__ == "__main__":
    ingest(verbose=True)
