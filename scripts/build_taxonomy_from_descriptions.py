"""
Build taxonomy_cache.json — maps every product to a controlled vocabulary node
from PRODUCT_TAXONOMY (213 predefined nodes).

No LLM API calls. Pipeline per product:
  1. Infer domain from product_category + description via keyword matching
  2. Embed product text with all-mpnet (batched)
  3. Cosine similarity against domain-filtered predefined nodes → top-3
  4. Cross-encoder rerank (score_pairs) → best node
  5. If CE confidence ≥ threshold → assign (domain, category, subcategory)
     else → store domain only (blank category/subcategory)

The script reads raw source files via data/loaders.py — no Qdrant running required.
Re-run whenever the inventory source files change.

Run time: ~3-5 minutes for 35k products on CPU.

Usage:
    python scripts/build_taxonomy_from_descriptions.py
"""

import os
import sys
import json
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

import numpy as np
from config import (TAXONOMY_EMBEDDINGS_PATH, TAXONOMY_CACHE_PATH,
                    DENSE_MODEL_NAME, PRODUCT_TAXONOMY)
from data.loaders import load_all

# ms-marco logits: ~[-10, +10]; -5.5 is low confidence
CONFIDENCE_THRESHOLD = -5.5

# ---------------------------------------------------------------------------
# 1. Load predefined taxonomy node embeddings
# ---------------------------------------------------------------------------
print("Loading taxonomy node embeddings...")
with open(TAXONOMY_EMBEDDINGS_PATH) as f:
    tax_emb = json.load(f)

nodes = []
by_domain = {}
for entry in tax_emb.values():
    idx = len(nodes)
    nodes.append({
        "domain":      entry["domain"],
        "category":    entry["category"],
        "subcategory": entry["subcategory"],
        "text":        entry["text"],
        "vector":      np.array(entry["vector"], dtype=np.float32),
    })
    by_domain.setdefault(entry["domain"], []).append(idx)

for d, idxs in by_domain.items():
    print(f"  {d}: {len(idxs)} predefined nodes")

# Pre-stack domain matrices for fast cosine similarity
domain_matrices = {
    d: np.stack([nodes[i]["vector"] for i in idxs])
    for d, idxs in by_domain.items()
}

# ---------------------------------------------------------------------------
# 2. Domain inference — keyword matching on product_category + description
#    No LLM calls; good enough for first-pass classification.
# ---------------------------------------------------------------------------

_ELEC = {
    "electrical", "lighting", "light", "circuit", "breaker", "panel", "wire",
    "cable", "conduit", "transformer", "switchgear", "fuse", "relay", "contactor",
    "outlet", "switch", "receptacle", "luminaire", "fixture", "ballast", "led",
    "motor control", "mcc", "busway", "mcb", "rcbo", "rccb", "isolator",
}
_PLUMB = {
    "plumbing", "water", "pipe", "valve", "faucet", "drain", "fitting",
    "coupling", "toilet", "lavatory", "sink", "pump", "sewer", "backflow",
    "pex", "cpvc", "pvc", "copper", "shower", "urinal", "flush", "hydrant",
}
_MECH = {
    "hvac", "air", "duct", "fan", "mechanical", "heating", "cooling",
    "ahu", "chiller", "boiler", "coil", "damper", "vav", "vrf", "thermostat",
    "ventilation", "exhaust", "supply air", "return air", "actuator",
}


def _infer_domain(category: str, description: str) -> str:
    text = (category + " " + description).lower()
    e = sum(1 for k in _ELEC  if k in text)
    p = sum(1 for k in _PLUMB if k in text)
    m = sum(1 for k in _MECH  if k in text)
    best = max(e, p, m)
    if best == 0:
        return "Unknown"
    if e == best:
        return "Electrical"
    if p == best:
        return "Plumbing"
    return "Mechanical"

# ---------------------------------------------------------------------------
# 3. Load all products from raw source files
# ---------------------------------------------------------------------------
print("\nLoading inventory records...")
records = load_all(verbose=True, attach_caches=False)
print()

# Assign domain per product
for r in records:
    r["_inferred_domain"] = _infer_domain(
        r.get("product_category", "") or "",
        r.get("description", "") or "",
    )

domain_groups = {d: [r for r in records if r["_inferred_domain"] == d]
                 for d in ("Electrical", "Mechanical", "Plumbing")}
unknown = [r for r in records if r["_inferred_domain"] == "Unknown"]

for d, rs in domain_groups.items():
    print(f"  {d}: {len(rs)}")
print(f"  Unknown: {len(unknown)}")

# ---------------------------------------------------------------------------
# 4. Embed + cosine + cross-encoder per domain
# ---------------------------------------------------------------------------
print(f"\nLoading embedding model: {DENSE_MODEL_NAME}")
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer(DENSE_MODEL_NAME)

print("Loading cross-encoder for reranking...")
from models.reranker import score_pairs  # float64 safe, returns list[float]

taxonomy_cache = {}
total_mapped = total_low_conf = 0

for domain, recs in domain_groups.items():
    if not recs:
        continue
    print(f"\nProcessing {domain} ({len(recs)} products)...")

    d_idxs   = by_domain[domain]
    d_matrix = domain_matrices[domain]

    # Build description text per product
    texts = []
    for r in recs:
        parts = [x for x in [
            r.get("description", "") or "",
            r.get("extended_description", "") or "",
            r.get("manufacturer_name", "") or "",
            r.get("product_category", "") or "",
        ] if x]
        texts.append(" | ".join(parts) if parts else r["internal_id"])

    # Batch embed
    t0 = time.time()
    vecs = encoder.encode(texts, batch_size=256, normalize_embeddings=True,
                          show_progress_bar=True)
    print(f"  Embedded in {time.time()-t0:.1f}s")

    print("  Reranking top-3 candidates with cross-encoder...")
    t0 = time.time()
    mapped = low_conf = 0

    BATCH = 256
    for bi in range(0, len(recs), BATCH):
        batch_recs  = recs[bi:bi + BATCH]
        batch_vecs  = vecs[bi:bi + BATCH]
        batch_texts = texts[bi:bi + BATCH]

        # Top-3 cosine candidates per product
        sims_batch = d_matrix @ batch_vecs.T             # (N_domain, batch_size)
        top3_idxs  = np.argsort(sims_batch, axis=0)[-3:, :].T  # (batch_size, 3)

        for rec, prod_text, top3 in zip(batch_recs, batch_texts, top3_idxs):
            candidate_indices = [d_idxs[i] for i in top3]
            pairs  = [(prod_text, nodes[i]["text"]) for i in candidate_indices]
            scores = score_pairs(pairs)
            best_pos   = int(np.argmax(scores))
            best_idx   = candidate_indices[best_pos]
            best_score = float(scores[best_pos])

            node = nodes[best_idx]
            if best_score < CONFIDENCE_THRESHOLD:
                taxonomy_cache[rec["id"]] = {
                    "taxonomy_domain":      domain,
                    "taxonomy_category":    "",
                    "taxonomy_subcategory": "",
                    "confidence_score":     round(best_score, 4),
                    "taxonomy_source":      "desc_mapper_low_conf",
                }
                low_conf += 1
            else:
                taxonomy_cache[rec["id"]] = {
                    "taxonomy_domain":      node["domain"],
                    "taxonomy_category":    node["category"],
                    "taxonomy_subcategory": node["subcategory"],
                    "confidence_score":     round(best_score, 4),
                    "taxonomy_source":      "desc_mapper",
                }
                mapped += 1

        if (bi // BATCH + 1) % 10 == 0:
            done = min(bi + BATCH, len(recs))
            rate = done / (time.time() - t0 + 0.001)
            print(f"    {done}/{len(recs)}  ({rate:.0f}/s)", flush=True)

    total_mapped   += mapped
    total_low_conf += low_conf
    print(f"  Done: {mapped} mapped, {low_conf} low-confidence (domain only)")

# Unknown domain — store empty taxonomy
for r in unknown:
    taxonomy_cache[r["id"]] = {
        "taxonomy_domain":      "",
        "taxonomy_category":    "",
        "taxonomy_subcategory": "",
        "taxonomy_source":      "no_domain",
    }

# ---------------------------------------------------------------------------
# 5. Save taxonomy_cache.json
# ---------------------------------------------------------------------------
tmp = TAXONOMY_CACHE_PATH + ".tmp"
with open(tmp, "w") as f:
    json.dump(taxonomy_cache, f, indent=2)
os.replace(tmp, TAXONOMY_CACHE_PATH)

print(f"\n{'='*60}")
print(f"taxonomy_cache.json saved: {len(taxonomy_cache)} products")
print(f"  Mapped to controlled vocabulary: {total_mapped}")
print(f"  Low confidence (domain only):    {total_low_conf}")
print(f"  Unknown domain:                  {len(unknown)}")
print(f"\nTop subcategories assigned:")
sc_counts = Counter(
    v["taxonomy_subcategory"] for v in taxonomy_cache.values()
    if v.get("taxonomy_subcategory")
)
for sc, n in sc_counts.most_common(15):
    print(f"  {n:5d}  {sc}")
print(f"\nNext step: run  python scripts/ingest.py")
