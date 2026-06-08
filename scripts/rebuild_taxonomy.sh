#!/usr/bin/env bash
#
# Full clean taxonomy rebuild — runs the whole pipeline in order.
#
# Use this whenever data/taxonomy.py (the single source of truth) changes, or a
# new inventory source is added. Each step runs in its own process so model
# state never leaks between stages.
#
#   1. build_taxonomy_embeddings.py        embed the seed controlled vocabulary
#   2. build_taxonomy_from_descriptions.py label every product into the OPEN
#                                          taxonomy store (deterministic ERP map
#                                          → match existing node → mint new node)
#   3. ingest.py                           upsert products + taxonomy into Qdrant
#
# The taxonomy is an OPEN vocabulary: the store (taxonomy_store.json) is seeded
# from the curated nodes but GROWS as step 2 mints nodes for product types that
# fit nothing existing. The store PERSISTS across runs and accumulates — adding a
# new source file just extends it. To rebuild the vocabulary from scratch
# (discard auto-created nodes), pass FRESH=1.
#
# Prereqs: run from any dir; the script cd's to the repo root. Needs the data
# files present, OPENROUTER_API_KEY for LLM node-naming (optional — falls back to
# ERP-category naming), and (for step 3) QDRANT_URL/QDRANT_API_KEY (or
# QDRANT_LOCAL_PATH) in .env. Step 3 writes to whatever backend .env points at —
# point it at a staging collection first if you don't want to touch production.
#
#   SKIP_INGEST=1   stop before pushing to Qdrant
#   FRESH=1         discard the existing store + auto nodes, rebuild from seed
#
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${FRESH:-0}" = "1" ]; then
    echo "FRESH=1 — discarding existing taxonomy store (auto nodes will be re-minted)."
    rm -f taxonomy_store.json taxonomy_labels.json
fi

echo "=============================================================="
echo "[1/3] Embedding seed controlled-vocabulary taxonomy nodes"
echo "=============================================================="
python scripts/build_taxonomy_embeddings.py

echo "=============================================================="
echo "[2/3] Classifying products into the OPEN taxonomy store"
echo "       (deterministic ERP map → match existing → mint new)"
echo "=============================================================="
python scripts/build_taxonomy_from_descriptions.py

if [ "${SKIP_INGEST:-0}" = "1" ]; then
    echo "SKIP_INGEST=1 set — stopping before ingest."
    echo "taxonomy_cache.json is rebuilt; run scripts/ingest.py when ready."
    exit 0
fi

echo "=============================================================="
echo "[3/3] Re-ingesting products + taxonomy into Qdrant"
echo "=============================================================="
python scripts/ingest.py

echo "=============================================================="
echo "Done. Taxonomy rebuilt and ingested."
echo "=============================================================="
