"""
One-time script: embed all taxonomy nodes and save vectors to taxonomy_embeddings.json.

Each node text is enriched with domain + category context so the embedding model
has enough signal to distinguish short subcategory labels.

Run once after any taxonomy change:
    python scripts/build_taxonomy_embeddings.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # loads .env

from config import PRODUCT_TAXONOMY, TAXONOMY_EMBEDDINGS_PATH, DENSE_MODEL_NAME


def _load_encoder():
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {DENSE_MODEL_NAME}")
    return SentenceTransformer(DENSE_MODEL_NAME)


def main():
    model = _load_encoder()

    nodes = []
    seen_keys = set()

    # The controlled vocabulary in data/taxonomy.py is the ONLY source of nodes.
    # We deliberately do NOT merge any labels back in from taxonomy_cache.json:
    # doing so created a feedback loop where LLM-invented labels accumulated into
    # a 1,473-node mess that the query side (247 config labels) could never match.
    # Embedding only the controlled vocabulary keeps query-side and product-side
    # labels identical by construction.
    for domain, categories in PRODUCT_TAXONOMY.items():
        for category, subcategories in categories.items():
            for subcategory in subcategories:
                key = f"{domain}::{category}::{subcategory}"
                seen_keys.add(key)
                text = f"{domain} | {category} | {subcategory}"
                nodes.append({
                    "key":        key,
                    "domain":     domain,
                    "category":   category,
                    "subcategory": subcategory,
                    "text":       text,
                })

    predefined_count = len(nodes)
    print(f"  {predefined_count} controlled-vocabulary nodes from data/taxonomy.py")

    print(f"Embedding {len(nodes)} controlled-vocabulary taxonomy nodes...")
    texts   = [n["text"] for n in nodes]
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    output = {}
    for node, vec in zip(nodes, vectors):
        output[node["key"]] = {
            "domain":      node["domain"],
            "category":    node["category"],
            "subcategory": node["subcategory"],
            "text":        node["text"],
            "vector":      vec.tolist(),
        }

    with open(TAXONOMY_EMBEDDINGS_PATH, "w") as f:
        json.dump(output, f)

    print(f"Saved {len(output)} node embeddings → {TAXONOMY_EMBEDDINGS_PATH}")

    from collections import Counter
    domains = Counter(v["domain"] for v in output.values())
    for domain, count in domains.most_common():
        print(f"  {domain:<12} {count} nodes")


if __name__ == "__main__":
    main()
