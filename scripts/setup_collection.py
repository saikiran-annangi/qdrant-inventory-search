"""
Create the Qdrant collection with full schema.

Run this once before ingestion:
    python scripts/setup_collection.py

Collection schema:
  dense        -- 768d cosine, HNSW m=16 ef_construct=200, int8 scalar quantization
  sparse_model -- BM25 over model number variants
  sparse_desc  -- BM25 over normalized description + specs

Payload indexes are created for filtering on source, manufacturer, category,
has_stock, total_qoh, min_cost, max_cost, and nested location fields.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add repo root to path so package imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Distance,
    HnswConfigDiff,
    ScalarQuantizationConfig,
    ScalarType,
    ScalarQuantization,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
)

from config import QDRANT_URL, COLLECTION_NAME, DENSE_DIM

client = QdrantClient(url=QDRANT_URL, check_compatibility=False)

existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    print(f"Dropping existing collection '{COLLECTION_NAME}'...")
    client.delete_collection(COLLECTION_NAME)

print(f"Creating collection '{COLLECTION_NAME}'...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": VectorParams(
            size=DENSE_DIM,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
            on_disk=False,
        )
    },
    sparse_vectors_config={
        "sparse_model": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        "sparse_desc":  SparseVectorParams(index=SparseIndexParams(on_disk=False)),
    },
)

print("Collection created. Adding payload indexes...")

# Keyword indexes for exact-match filters
for field in ("source", "manufacturer_name", "product_category", "currency"):
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field,
        field_schema=PayloadSchemaType.KEYWORD,
    )

# Text index for model_number (prefix / whitespace tokenization)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="model_number",
    field_schema=TextIndexParams(
        type="text",
        tokenizer=TokenizerType.WHITESPACE,
        min_token_len=1,
        max_token_len=30,
        lowercase=True,
    ),
)

# Boolean and numeric indexes
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="has_stock",
    field_schema=PayloadSchemaType.BOOL,
)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="total_qoh",
    field_schema=PayloadSchemaType.INTEGER,
)
for field in ("min_cost", "max_cost"):
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field,
        field_schema=PayloadSchemaType.FLOAT,
    )

# Nested location-level indexes
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="locations[].location_erp_id",
    field_schema=PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="locations[].in_stock",
    field_schema=PayloadSchemaType.BOOL,
)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="locations[].qoh",
    field_schema=PayloadSchemaType.INTEGER,
)

print("All payload indexes created.")

info = client.get_collection(COLLECTION_NAME)
print(f"\nCollection '{COLLECTION_NAME}' ready:")
print(f"  Status         : {info.status}")
print(f"  Dense vectors  : {DENSE_DIM}d cosine, HNSW m=16 ef_construct=200")
print(f"  Sparse vectors : sparse_model, sparse_desc (BM25)")
print(f"  Points count   : {info.points_count}")
