"""
One-time migration: local SQLite Qdrant storage -> Qdrant HTTP server.

Use this only if you have an existing local SQLite store and want to move
it to the HTTP server without re-ingesting. If you are starting fresh,
run setup_collection.py and ingest.py instead.

Usage:
    python scripts/migrate_to_server.py

Reads every point from the local file-based client and upserts it into the
running Qdrant server at QDRANT_URL. Takes around 2-5 minutes for 35k points.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector,
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    HnswConfigDiff,
)

from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, DENSE_DIM

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_PATH = os.path.join(REPO_ROOT, "qdrant_storage")
BATCH_SIZE   = 500


def main():
    print("Connecting to local SQLite client...")
    local = QdrantClient(path=STORAGE_PATH)

    print(f"Connecting to Qdrant server at {QDRANT_URL}...")
    server = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

    existing = [c.name for c in server.get_collections().collections]
    if COLLECTION_NAME in existing:
        count = server.count(COLLECTION_NAME).count
        print(f"Collection '{COLLECTION_NAME}' already exists on server with {count} points.")
        ans = input("Re-migrate? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
        server.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")

    print("Creating collection on server...")
    server.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
            )
        },
        sparse_vectors_config={
            "sparse_model": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            "sparse_desc":  SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )

    total_local = local.count(COLLECTION_NAME).count
    print(f"Migrating {total_local:,} points in batches of {BATCH_SIZE}...\n")

    offset   = None
    migrated = 0

    while True:
        results, offset = local.scroll(
            collection_name=COLLECTION_NAME,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not results:
            break

        points = []
        for pt in results:
            vectors = {}
            if "dense" in pt.vector:
                vectors["dense"] = pt.vector["dense"]
            for sv_name in ("sparse_model", "sparse_desc"):
                if sv_name in pt.vector:
                    sv = pt.vector[sv_name]
                    vectors[sv_name] = SparseVector(indices=sv.indices, values=sv.values)

            points.append(PointStruct(id=pt.id, vector=vectors, payload=pt.payload))

        server.upsert(collection_name=COLLECTION_NAME, points=points)
        migrated += len(points)

        pct = migrated / total_local * 100
        print(f"\r  {migrated:>6,} / {total_local:,}  ({pct:.1f}%)", end="", flush=True)

        if offset is None:
            break

    print(f"\n\nDone. Migrated {migrated:,} points.")
    server_count = server.count(COLLECTION_NAME).count
    print(f"Server reports: {server_count:,} points in '{COLLECTION_NAME}'.")

    local.close()


if __name__ == "__main__":
    main()
