"""
Embed and upsert all inventory records into Qdrant.

Reads raw CSV/XLSX files via data.loaders and merges in two optional caches
when present:

    enrichment_cache.json  → extended_description (LLM-generated rich text)
    taxonomy_cache.json    → taxonomy_domain / taxonomy_category / taxonomy_subcategory

Rebuild the caches first (one-time, see CLAUDE.md) to get the full benefit of
LLM enrichment and taxonomy soft boost. Ingest works without them — products
will just have empty extended_description and taxonomy fields.

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
from qdrant_client.models import (
    PointStruct, SparseVector, VectorParams, Distance,
    SparseVectorParams, SparseIndexParams, HnswConfigDiff, PayloadSchemaType,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
)

from config import COLLECTION_NAME, INGEST_BATCH_SIZE, QDRANT_URL, QDRANT_API_KEY, QDRANT_LOCAL_PATH
from data.loaders import load_all
from data.normalizers import model_number_variants, spec_text_with_attributes
from models.embeddings import get_dense_model, get_bm25_model


def _get_client():
    if QDRANT_LOCAL_PATH:
        import shutil
        if os.path.isdir(QDRANT_LOCAL_PATH):
            shutil.rmtree(QDRANT_LOCAL_PATH)
        return QdrantClient(path=QDRANT_LOCAL_PATH), f"embedded local ({QDRANT_LOCAL_PATH})"
    return (
        QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY,
                     check_compatibility=False, timeout=120),
        f"server ({QDRANT_URL})",
    )


def main():
    print("Loading inventory records…")
    records = load_all(verbose=True, attach_caches=True)
    print()

    # Build text fields for each encoder.
    # Dense includes extended_description when available — richer semantics
    # for products whose original description is a bare model code or abbreviation.
    dense_texts, sm_texts, sd_texts = [], [], []
    for r in records:
        desc  = r.get("description")           or ""
        ext   = r.get("extended_description")  or ""
        mfr   = r.get("manufacturer_name")     or ""
        cat   = r.get("product_category")      or ""
        model = r.get("model_number")          or ""

        dense_base = " ".join(x for x in [desc, ext, mfr, cat, model] if x)
        dense_texts.append(dense_base.strip() or r["internal_id"])

        sm_texts.append(model_number_variants(model) or model or r["internal_id"])

        desc_text = " ".join(x for x in [desc, ext, mfr, cat] if x)
        sd_texts.append(
            spec_text_with_attributes(desc_text) or desc or r["internal_id"]
        )

    # Encode
    dense_model = get_dense_model()
    bm25        = get_bm25_model()

    print("Encoding dense vectors (all-mpnet, batched)…")
    t0 = time.time()
    dense_vecs = []
    for i, t in enumerate(dense_texts):
        dense_vecs.append(dense_model.encode(t, normalize_embeddings=True).tolist())
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(dense_texts)}  ({(i+1)/(time.time()-t0):.1f}/s)")
    print(f"  Dense done in {time.time()-t0:.0f}s")

    print("Encoding sparse_model (BM25)…")
    sm_vecs = [SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
               for r in bm25.embed(sm_texts)]

    print("Encoding sparse_desc (BM25 + attribute anchors)…")
    sd_vecs = [SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
               for r in bm25.embed(sd_texts)]

    client, target = _get_client()
    print(f"\nTarget: {target}")

    # Recreate collection from scratch
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    # Dense vectors use int8 scalar quantization: the quantized copy is
    # pinned in RAM (always_ram) while the original float32 vectors live on
    # disk (on_disk=True). Rescore (enabled query-side in core/search.py) reads
    # the on-disk originals to recover accuracy. ~4x smaller in-RAM footprint.
    # Binary quantization intentionally avoided: mpnet is 768d (below ~1024d
    # where binary holds up).
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(
            size=768, distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8, quantile=0.99, always_ram=True,
                )
            ),
            on_disk=True,
        )},
        sparse_vectors_config={
            "sparse_model": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            "sparse_desc":  SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )

    # Payload indexes speed up filter-based queries (source, stock, taxonomy)
    for fld, sch in [
        ("source",              PayloadSchemaType.KEYWORD),
        ("manufacturer_name",   PayloadSchemaType.KEYWORD),
        ("product_category",    PayloadSchemaType.KEYWORD),
        ("currency",            PayloadSchemaType.KEYWORD),
        ("has_stock",           PayloadSchemaType.BOOL),
        ("taxonomy_domain",     PayloadSchemaType.KEYWORD),
        ("taxonomy_category",   PayloadSchemaType.KEYWORD),
        ("taxonomy_subcategory", PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(COLLECTION_NAME, fld, field_schema=sch)

    print("Upserting…")
    t0 = time.time()
    for i in range(0, len(records), INGEST_BATCH_SIZE):
        batch = records[i:i + INGEST_BATCH_SIZE]
        points = []
        for j, r in enumerate(batch):
            payload = {k: v for k, v in r.items() if k != "id"}
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
            COLLECTION_NAME, points=points,
            wait=(i + INGEST_BATCH_SIZE >= len(records)),
        )
        print(f"\r  {min(i + INGEST_BATCH_SIZE, len(records))}/{len(records)}", end="", flush=True)

    n = client.count(COLLECTION_NAME).count
    print(f"\n\nDone. Collection rebuilt with {n} points in {time.time()-t0:.0f}s.")
    client.close()


if __name__ == "__main__":
    main()
