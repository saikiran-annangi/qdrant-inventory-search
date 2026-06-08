"""Classify every product into the taxonomy — and GROW the taxonomy when needed.

The taxonomy is an OPEN vocabulary (data/taxonomy_store.py): the curated nodes
in data/taxonomy.py are the seed, but production files bring product types that
fit no seed node. This script:

  Pass 1 (deterministic)  product carries a known ERP category  → CATEGORY_MAP node
  Pass 2 (match)          embed product, cosine top-k vs store, cross-encoder
                          rerank; CE ≥ threshold → assign that existing node
  Pass 3 (mint)           no confident match → create a NEW node:
                            • name it via the ERP category if present, else ask
                              the LLM (shown the existing vocabulary so it stays
                              consistent — models/taxonomy_namer.py)
                            • TaxonomyStore.add_node dedups (exact + semantic) so
                              near-duplicates collapse instead of proliferating
                          Once a node exists, later similar products match it in
                          Pass 2 — so the LLM is called ~once per new concept.

Both the query side and ingestion read the SAME store, so anything minted here
is immediately query-reachable (no drift). Writes taxonomy_cache.json (per-
product labels) and updates taxonomy_store.json / taxonomy_labels.json.

Reads raw source files via data/loaders.py — no Qdrant required. Re-run whenever
sources change.  Usage:  python scripts/build_taxonomy_from_descriptions.py
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
                    TAXONOMY_STORE_PATH, TAXONOMY_LABELS_PATH,
                    TAXONOMY_ASSIGN_THRESHOLD, DENSE_MODEL_NAME, PRODUCT_TAXONOMY)
from data.taxonomy import lookup_category
from data.taxonomy_store import TaxonomyStore, node_text
from data.loaders import load_all
from models import taxonomy_namer

# ms-marco logits: ~[-10, +10]; below this the best existing node is a poor fit,
# so we mint a new node rather than force-assign.
CONFIDENCE_THRESHOLD = -5.5
TOPK = 3

# --- speed controls (env-tunable) ---------------------------------------------
# FAST=1: skip the float64 cross-encoder and assign by COSINE ≥ assign-threshold
#   (orders of magnitude faster — good for a local smoke build). Default off.
# TAXONOMY_LLM_NAMING=0: never call the LLM to name a minted node; fall back to
#   ERP-category naming or domain-only. Removes the per-product LLM latency.
# Use both for a quick local build:  TAXONOMY_FAST=1 TAXONOMY_LLM_NAMING=0
FAST = os.getenv("TAXONOMY_FAST", "0") == "1"
LLM_NAMING = os.getenv("TAXONOMY_LLM_NAMING", "1") != "0"
from config import TAXONOMY_ASSIGN_THRESHOLD

# ---------------------------------------------------------------------------
# 1. Open the taxonomy store (seed it on first run)
# ---------------------------------------------------------------------------
print("Opening taxonomy store...")
store = TaxonomyStore(TAXONOMY_STORE_PATH, TAXONOMY_LABELS_PATH)

print(f"Loading embedding model: {DENSE_MODEL_NAME}")
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer(DENSE_MODEL_NAME)


def _embed(text: str):
    return encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]


if len(store) == 0:
    # Seed from the prebuilt node embeddings if available (fast), else encode.
    if os.path.exists(TAXONOMY_EMBEDDINGS_PATH):
        print("  Seeding store from taxonomy_embeddings.json ...")
        with open(TAXONOMY_EMBEDDINGS_PATH) as f:
            tax_emb = json.load(f)
        for e in tax_emb.values():
            store.add_node(e["domain"], e["category"], e["subcategory"],
                           name_embedding=e["vector"], provenance="seed",
                           dedup_threshold=2.0)  # no dedup while seeding curated set
    else:
        print("  Seeding store from PRODUCT_TAXONOMY (encoding nodes)...")
        store.seed_from(PRODUCT_TAXONOMY, _embed)
    store.save()
print(f"  Store seeded: {len(store)} nodes across domains "
      f"{sorted({n['domain'] for n in store._nodes})}")

# ---------------------------------------------------------------------------
# 2. Domain inference (keyword matching) — covers all four domains
# ---------------------------------------------------------------------------
_ELEC = {
    "electrical", "lighting", "light", "circuit", "breaker", "panel", "wire",
    "cable", "conduit", "transformer", "switchgear", "fuse", "relay", "contactor",
    "outlet", "switch", "receptacle", "luminaire", "fixture", "ballast", "led",
    "motor control", "mcc", "busway", "mcb", "rcbo", "rccb", "isolator",
    "lug", "terminal", "gland", "heatshrink", "earth", "solar", "data", "fibre",
}
_PLUMB = {
    "plumbing", "water", "pipe", "valve", "faucet", "drain", "fitting",
    "coupling", "toilet", "lavatory", "sink", "sewer", "backflow",
    "pex", "cpvc", "shower", "urinal", "flush", "hydrant", "hose",
}
_MECH = {
    "hvac", "duct", "mechanical", "ahu", "chiller", "boiler", "coil",
    "damper", "vav", "vrf", "air conditioning", "aircon", "rangehood",
    "cooker", "oven", "catering", "hand dryer",
}
_TOOLS = {
    "drill", "holesaw", "saw", "blade", "grinding", "plier", "cutter",
    "screwdriver", "spanner", "wrench", "socket set", "hex key", "knife",
    "multimeter", "clamp meter", "tester", "glove", "goggle", "respirator",
    "mask", "lockout", "safety", "ladder", "tool box", "toolbox", "label",
    "marker", "sealant", "adhesive", "aerosol", "lubricant", "fastener",
    "fixing", "washer", "anchor", "screw", "bolt",
}


def _infer_domain(category: str, description: str) -> str:
    text = (category + " " + description).lower()
    e = sum(1 for k in _ELEC  if k in text)
    p = sum(1 for k in _PLUMB if k in text)
    m = sum(1 for k in _MECH  if k in text)
    t = sum(1 for k in _TOOLS if k in text)
    best = max(e, p, m, t)
    if best == 0:
        return "Unknown"
    if e == best:
        return "Electrical"
    if t == best:
        return "Tools & Site"
    if p == best:
        return "Plumbing"
    return "Mechanical"


def _product_text(r: dict) -> str:
    parts = [x for x in [
        r.get("description", "") or "",
        r.get("extended_description", "") or "",
        r.get("manufacturer_name", "") or "",
        r.get("product_category", "") or "",
    ] if x]
    return " | ".join(parts) if parts else r.get("internal_id", "")


# ---------------------------------------------------------------------------
# 3. Load products
# ---------------------------------------------------------------------------
print("\nLoading inventory records...")
records = load_all(verbose=True, attach_caches=False)
print()

taxonomy_cache = {}
det_mapped = 0
to_embed = []

# Pass 1 — deterministic ERP-category mapping (exact, no guessing).
for r in records:
    erp_cat = (r.get("product_category", "") or "").strip()
    domain, category, subcategory = lookup_category(erp_cat)
    if domain is not None:
        node = store.get(domain, category, subcategory)
        if node is not None:
            store.bump(node)
        taxonomy_cache[r["id"]] = {
            "taxonomy_domain": domain, "taxonomy_category": category,
            "taxonomy_subcategory": subcategory, "confidence_score": 1.0,
            "taxonomy_source": "erp_category_map",
        }
        det_mapped += 1
    else:
        to_embed.append(r)

print(f"  Deterministic (ERP category map): {det_mapped}")
print(f"  Need match-or-mint:               {len(to_embed)}")

for r in to_embed:
    r["_domain"] = _infer_domain(r.get("product_category", "") or "",
                                 r.get("description", "") or "")

# ---------------------------------------------------------------------------
# 4. Match-or-mint
# ---------------------------------------------------------------------------
if not FAST:
    print("\nLoading cross-encoder...")
    from models.reranker import score_pairs  # float64 safe
else:
    print("\nFAST mode: cosine-only assignment (cross-encoder skipped).")

print(f"Embedding {len(to_embed)} products...")
texts = [_product_text(r) for r in to_embed]
t0 = time.time()
vecs = encoder.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
print(f"  Embedded in {time.time()-t0:.1f}s")

namer_on = taxonomy_namer.is_enabled() and LLM_NAMING
print(f"LLM node-namer: {'ENABLED' if namer_on else 'DISABLED (ERP/heuristic naming only)'}")

stats = Counter()
new_nodes = []
t0 = time.time()

for idx, (r, vec) in enumerate(zip(to_embed, vecs)):
    domain = r["_domain"]
    if domain == "Unknown":
        taxonomy_cache[r["id"]] = {
            "taxonomy_domain": "", "taxonomy_category": "", "taxonomy_subcategory": "",
            "taxonomy_source": "no_domain",
        }
        stats["no_domain"] += 1
        continue

    # Pass 2 — match an existing node.
    cands = store.topk(vec, domain, k=TOPK)
    assigned = None
    if cands:
        if FAST:
            # Cosine-only: assign the top node if it clears the cosine threshold.
            node, cos = cands[0]
            if cos >= TAXONOMY_ASSIGN_THRESHOLD:
                assigned = node
                src = "store_match"
                conf = round(cos, 4)
        else:
            # Cross-encoder rerank of the top-k (higher accuracy, slower).
            pairs = [(texts[idx], node_text(domain, n["category"], n["subcategory"]))
                     for n, _ in cands]
            ce = score_pairs(pairs)
            bi = int(np.argmax(ce))
            if float(ce[bi]) >= CONFIDENCE_THRESHOLD:
                assigned = cands[bi][0]
                src = "store_match"
                conf = round(float(ce[bi]), 4)

    # Pass 3 — mint a new node.
    if assigned is None:
        erp_cat = (r.get("product_category", "") or "").strip()
        proposed = None
        if namer_on and not FAST:
            proposed = taxonomy_namer.propose_node(
                texts[idx], domain, store.labels_for_domain(domain))
        if proposed is None and erp_cat:
            proposed = (erp_cat, erp_cat)   # ERP-category fallback (no LLM)

        if proposed is not None:
            cat, sub = proposed
            name_emb = _embed(node_text(domain, cat, sub))
            node, created = store.add_node(domain, cat, sub, name_embedding=name_emb)
            if node is not None:
                assigned = node
                src = "auto_created" if created else "store_match"
                conf = 1.0 if created else 0.99
                if created:
                    new_nodes.append((domain, cat, sub))

    if assigned is not None:
        store.bump(assigned)
        taxonomy_cache[r["id"]] = {
            "taxonomy_domain": assigned["domain"],
            "taxonomy_category": assigned["category"],
            "taxonomy_subcategory": assigned["subcategory"],
            "confidence_score": conf,
            "taxonomy_source": src,
        }
        stats[src] += 1
    else:
        # Domain known but no node and no name available → domain-only.
        taxonomy_cache[r["id"]] = {
            "taxonomy_domain": domain, "taxonomy_category": "", "taxonomy_subcategory": "",
            "taxonomy_source": "domain_only",
        }
        stats["domain_only"] += 1

    if (idx + 1) % 1000 == 0:
        rate = (idx + 1) / (time.time() - t0 + 1e-3)
        print(f"    {idx+1}/{len(to_embed)}  ({rate:.0f}/s, {len(new_nodes)} new nodes)", flush=True)

# ---------------------------------------------------------------------------
# 5. Persist
# ---------------------------------------------------------------------------
store.save()
tmp = TAXONOMY_CACHE_PATH + ".tmp"
with open(tmp, "w") as f:
    json.dump(taxonomy_cache, f, indent=2)
os.replace(tmp, TAXONOMY_CACHE_PATH)

print(f"\n{'='*60}")
print(f"taxonomy_cache.json saved: {len(taxonomy_cache)} products")
print(f"  Deterministic (ERP map) : {det_mapped}")
print(f"  Matched existing node   : {stats['store_match']}")
print(f"  Auto-created new node   : {stats['auto_created']}")
print(f"  Domain only (no node)   : {stats['domain_only']}")
print(f"  Unknown domain          : {stats['no_domain']}")
print(f"\nStore now holds {len(store)} nodes "
      f"({sum(1 for n in store._nodes if n['provenance']=='auto')} auto-created).")
if new_nodes:
    print(f"\n{len(new_nodes)} new nodes minted (top by domain):")
    by_dom = Counter((d, c, s) for (d, c, s) in new_nodes)
    for (d, c, s), n in by_dom.most_common(20):
        print(f"  [{d}] {c} > {s}")
print("\nNext: python scripts/ingest.py  (writes labels to Qdrant)")
print("Review/merge auto nodes: python scripts/taxonomy_review.py")
