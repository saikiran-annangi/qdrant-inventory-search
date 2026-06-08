"""Review and curate the open taxonomy — list auto-created nodes, find probable
duplicates, and merge nodes (re-labeling affected products).

The ingest classifier mints new nodes automatically (with dedup), but human
curation keeps the vocabulary clean over time. This is that tool.

Usage:
    python scripts/taxonomy_review.py list            # all nodes, auto first
    python scripts/taxonomy_review.py auto            # only auto-created nodes
    python scripts/taxonomy_review.py dupes [0.80]    # probable duplicate pairs
    python scripts/taxonomy_review.py merge \\
        "Electrical::Cable Accessories::Cable Ties" \\
        "Electrical::Cable Accessories::Cable Ties & Clips"
        # fold the first node into the second; re-label its products
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import TAXONOMY_STORE_PATH, TAXONOMY_LABELS_PATH, TAXONOMY_CACHE_PATH
from data.taxonomy_store import TaxonomyStore, _key


def _store() -> TaxonomyStore:
    s = TaxonomyStore(TAXONOMY_STORE_PATH, TAXONOMY_LABELS_PATH)
    if len(s) == 0:
        print("Taxonomy store is empty — run scripts/build_taxonomy_from_descriptions.py first.")
        sys.exit(1)
    return s


def _fmt(n: dict) -> str:
    tag = "auto" if n["provenance"] == "auto" else "seed"
    return f"[{tag}] ({n.get('product_count', 0):5d})  {n['domain']} :: {n['category']} :: {n['subcategory']}"


def cmd_list(only_auto: bool = False):
    s = _store()
    nodes = [n for n in s._nodes if (n["provenance"] == "auto" or not only_auto)]
    nodes.sort(key=lambda n: (n["provenance"] != "auto", n["domain"], -n.get("product_count", 0)))
    for n in nodes:
        print(_fmt(n))
    autos = sum(1 for n in s._nodes if n["provenance"] == "auto")
    print(f"\n{len(s)} nodes total, {autos} auto-created.")


def cmd_dupes(threshold: float = 0.80):
    """Flag node pairs whose name embeddings are very similar (likely the same
    concept worded differently) — candidates to merge."""
    s = _store()
    by_domain = {}
    for n in s._nodes:
        if n.get("embedding"):
            by_domain.setdefault(n["domain"], []).append(n)
    pairs = []
    for domain, nodes in by_domain.items():
        mat = np.array([n["embedding"] for n in nodes], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True); norms[norms == 0] = 1e-9
        mat = mat / norms
        sims = mat @ mat.T
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if sims[i, j] >= threshold:
                    pairs.append((float(sims[i, j]), nodes[i], nodes[j]))
    pairs.sort(reverse=True, key=lambda x: x[0])
    if not pairs:
        print(f"No node pairs above similarity {threshold:.2f}.")
        return
    print(f"Probable duplicate node pairs (cosine ≥ {threshold:.2f}):\n")
    for score, a, b in pairs:
        keep, drop = (a, b) if a.get("product_count", 0) >= b.get("product_count", 0) else (b, a)
        print(f"  {score:.3f}")
        print(f"     {_fmt(a)}")
        print(f"     {_fmt(b)}")
        print(f"     → suggest: merge \"{drop['domain']}::{drop['category']}::{drop['subcategory']}\" "
              f"\"{keep['domain']}::{keep['category']}::{keep['subcategory']}\"\n")


def _parse_node_arg(arg: str):
    parts = arg.split("::")
    if len(parts) != 3:
        print(f"Bad node spec {arg!r}; expected 'Domain::Category::Subcategory'")
        sys.exit(1)
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def cmd_merge(src_arg: str, dst_arg: str):
    s = _store()
    sd, sc, ss = _parse_node_arg(src_arg)
    dd, dc, dsub = _parse_node_arg(dst_arg)
    src = s.get(sd, sc, ss)
    dst = s.get(dd, dc, dsub)
    if src is None:
        print(f"Source node not found: {src_arg}"); sys.exit(1)
    if dst is None:
        print(f"Destination node not found: {dst_arg}"); sys.exit(1)
    if src is dst:
        print("Source and destination are the same node."); sys.exit(1)

    moved = s.merge(src, dst)
    s.save()

    # Re-label affected products in the cache.
    relabeled = 0
    if os.path.exists(TAXONOMY_CACHE_PATH):
        with open(TAXONOMY_CACHE_PATH) as f:
            cache = json.load(f)
        for entry in cache.values():
            if (entry.get("taxonomy_domain") == sd
                    and entry.get("taxonomy_category") == sc
                    and entry.get("taxonomy_subcategory") == ss):
                entry["taxonomy_category"] = dc
                entry["taxonomy_subcategory"] = dsub
                entry["taxonomy_source"] = "merged"
                relabeled += 1
        tmp = TAXONOMY_CACHE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, TAXONOMY_CACHE_PATH)

    print(f"Merged. Moved {moved} product_count; re-labeled {relabeled} cached products.")
    print("Re-run scripts/ingest.py to push the relabeled products to Qdrant.")


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "list":
        cmd_list(only_auto=False)
    elif cmd == "auto":
        cmd_list(only_auto=True)
    elif cmd == "dupes":
        thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.80
        cmd_dupes(thr)
    elif cmd == "merge":
        if len(sys.argv) != 4:
            print("Usage: taxonomy_review.py merge <src> <dst>"); sys.exit(1)
        cmd_merge(sys.argv[2], sys.argv[3])
    else:
        print(__doc__); sys.exit(1)


if __name__ == "__main__":
    main()
