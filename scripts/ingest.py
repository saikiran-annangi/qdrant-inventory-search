"""
Embed and upsert all inventory records into Qdrant.

Reads from raw CSV/XLSX files via data.loaders, which also merges in
enrichment_cache.json, attributes_cache.json, and taxonomy_cache.json
when those files are present.

Usage:
    python scripts/ingest.py
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_R, ".env"))
except ImportError:
    pass

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from config import COLLECTION_NAME, INGEST_BATCH_SIZE, QDRANT_URL, QDRANT_API_KEY, QDRANT_LOCAL_PATH
from data.loaders import load_all
from data.normalizers import normalize_specs, model_number_variants
from models.embeddings import get_dense_model, get_bm25_model


def _get_client():
    if QDRANT_LOCAL_PATH:
        import shutil
        if os.path.isdir(QDRANT_LOCAL_PATH):
            shutil.rmtree(QDRANT_LOCAL_PATH)
        return QdrantClient(path=QDRANT_LOCAL_PATH), f"embedded local ({QDRANT_LOCAL_PATH})"
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY,
                        check_compatibility=False, timeout=120), f"server ({QDRANT_URL})"


def main():
    print("Loading inventory records…")
    records = load_all(verbose=True, attach_caches=True)
    print()

    # Build text fields for each encoder
    dense_texts, sm_texts, sd_texts = [], [], []
    for r in records:
        ext  = r.get("extended_description") or ""
        desc = r.get("description") or ""
        mfr  = r.get("manufacturer_name") or ""
        cat  = r.get("product_category") or ""
        model = r.get("model_number") or ""

        # Dense: description + extended_description give semantic richness
        dense_base = " ".join(x for x in [desc, ext, mfr, cat, model] if x)
        dense_texts.append(dense_base.strip() or r["internal_id"])

        # sparse_model: model number + internal_id variants for exact lookup
        sm = model_number_variants(model) or model or r["internal_id"]
        sm_texts.append(sm)

        # sparse_desc: normalized description + extended_description for keyword search
        desc_text = " ".join(x for x in [desc, ext, mfr, cat] if x)
        sd_texts.append(normalize_specs(desc_text) or desc or r["internal_id"])

    # Encode
    dense_model = get_dense_model()
    bm25        = get_bm25_model()

    print("Encoding dense vectors (all-mpnet)…")
    t0 = time.time()
    dense_vecs = []
    for i, text in enumerate(dense_texts):
        dense_vecs.append(dense_model.encode(text, normalize_embeddings=True).tolist())
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(dense_texts)}  ({(i+1)/elapsed:.1f}/s  {elapsed:.0f}s)")
    print(f"  Dense done in {time.time()-t0:.0f}s")

    print("Encoding sparse_model (BM25)…")
    sm_vecs = [SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
               for v in bm25.embed(sm_texts)]

    print("Encoding sparse_desc (BM25)…")
    sd_vecs = [SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
               for v in bm25.embed(sd_texts)]

    # Upsert
    client, target = _get_client()
    print(f"\nTarget: {target}")

    print("Upserting…")
    t0 = time.time()
    for i in range(0, len(records), INGEST_BATCH_SIZE):
        batch = records[i:i + INGEST_BATCH_SIZE]
        points = []
        for j, r in enumerate(batch):
            payload = {k: v for k, v in r.items() if k not in ("id",)}
            points.append(PointStruct(
                id=r["id"],
                vector={
                    "dense":        dense_vecs[i + j],
                    "sparse_model": sm_vecs[i + j],
                    "sparse_desc":  sd_vecs[i + j],
                },
                payload=payload,
            ))
        client.upsert(
            COLLECTION_NAME,
            points=points,
            wait=(i + INGEST_BATCH_SIZE >= len(records)),
        )
        print(f"\r  {min(i + INGEST_BATCH_SIZE, len(records))}/{len(records)}", end="", flush=True)

    n = client.count(COLLECTION_NAME).count
    print(f"\n\nDone. {n} points in collection '{COLLECTION_NAME}'  ({time.time()-t0:.0f}s)")
    client.close()


if __name__ == "__main__":
    main()
