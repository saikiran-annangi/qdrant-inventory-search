"""
Batch-classify all products from attributes_cache.json into the predefined
taxonomy and save results to taxonomy_cache.json.

No API calls — pure local CPU (embedding + cross-encoder).


Usage:
    python scripts/build_taxonomy_cache.py --workers 8
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # loads .env

from config import ATTRIBUTES_CACHE_PATH, TAXONOMY_CACHE_PATH, TAXONOMY_EMBEDDINGS_PATH
from models.taxonomy_mapper import map_to_taxonomy, llm_fallback_taxonomy

WORKERS = 50  # LLM fallback is I/O-bound; high parallelism is beneficial


def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, path)


def _classify(item: tuple) -> tuple:
    pid, raw_attrs, product_fields, skip_cosine = item

    if skip_cosine:
        # Product was previously empty — cosine will fail again, skip straight to LLM
        result = llm_fallback_taxonomy(product_fields, raw_attrs)
        result["taxonomy_source"] = "llm_fallback"
    else:
        result = map_to_taxonomy(raw_attrs)
        if not result.get("taxonomy_category"):
            result = llm_fallback_taxonomy(product_fields, raw_attrs)
            result["taxonomy_source"] = "llm_fallback"
        else:
            result["taxonomy_source"] = "cosine"

    return pid, result


def main():
    parser = argparse.ArgumentParser(description="Build taxonomy classification cache")
    parser.add_argument("--limit",   type=int, help="Max products to process (for testing)")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Thread pool size")
    args = parser.parse_args()

    if not os.path.exists(TAXONOMY_EMBEDDINGS_PATH):
        print(f"[ERROR] taxonomy_embeddings.json not found.")
        print("Run: python scripts/build_taxonomy_embeddings.py")
        sys.exit(1)

    if not os.path.exists(ATTRIBUTES_CACHE_PATH):
        print(f"[ERROR] attributes_cache.json not found.")
        print("Run: python scripts/build_attributes_cache.py")
        sys.exit(1)

    print("Loading attributes cache...")
    attrs_cache = _load_cache(ATTRIBUTES_CACHE_PATH)
    print(f"  {len(attrs_cache)} products in attributes_cache.json")

    taxonomy_cache = _load_cache(TAXONOMY_CACHE_PATH)
    print(f"  {len(taxonomy_cache)} existing entries in taxonomy_cache.json")

    # Load raw product data for LLM fallback context
    print("Loading raw product data for LLM fallback context...")
    from data.loaders import load_all
    raw_records = load_all(verbose=False, attach_caches=False)
    product_text_lookup = {
        r["id"]: {
            "model_number":         r.get("model_number",         "") or "",
            "description":          r.get("description",          "") or "",
            "extended_description": r.get("extended_description") or "",
            "manufacturer_name":    r.get("manufacturer_name",    "") or "",
            "product_category":     r.get("product_category",     "") or "",
        }
        for r in raw_records
    }
    print(f"  {len(product_text_lookup)} product records loaded")

    # Process products that are either:
    #   - not yet in taxonomy_cache at all, OR
    #   - in cache with empty category and not yet through LLM fallback
    #     (identified by absence of taxonomy_source field — pre-existing entries)
    # skip_cosine=True for products known to have empty category — avoids wasting
    # CPU on cosine+cross-encoder that will fail again, goes straight to LLM.
    to_process = [
        (
            pid, raw, product_text_lookup.get(pid, {}),
            pid in taxonomy_cache and not taxonomy_cache[pid].get("taxonomy_category"),
        )
        for pid, raw in attrs_cache.items()
        if pid not in taxonomy_cache
        or (
            not taxonomy_cache[pid].get("taxonomy_category")
            and "taxonomy_source" not in taxonomy_cache[pid]
        )
    ]

    unknown_count = sum(1 for raw in attrs_cache.values() if raw.get("domain") == "Unknown")
    print(f"  {unknown_count} Unknown-domain products (LLM fallback will attempt classification)")

    if args.limit:
        to_process = to_process[: args.limit]
        print(f"  (limited to {args.limit} for testing)")

    if not to_process:
        print("Taxonomy cache is complete — nothing to do.")
        _print_summary(taxonomy_cache)
        return

    # Pre-warm both models in the main thread so workers share the singletons
    # instead of each loading their own copy (avoids 8× model loading overhead).
    print("Pre-warming embedding model and cross-encoder...")
    from models.taxonomy_mapper import _load_taxonomy_nodes, _get_encoder, _get_reranker
    _load_taxonomy_nodes()
    _get_encoder()
    _get_reranker()
    print("Models ready.\n")

    print(f"Classifying {len(to_process)} products with {args.workers} workers...")

    start  = time.perf_counter()
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_classify, item): item for item in to_process}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                pid, result = fut.result()
                taxonomy_cache[pid] = result
            except Exception as exc:
                pid, _, _, _ = futures[fut]
                print(f"  [ERROR] {pid}: {exc}")
                taxonomy_cache[pid] = {"taxonomy_domain": "", "taxonomy_category": "", "taxonomy_subcategory": "", "taxonomy_source": "error"}
                errors += 1

            if i % 500 == 0:
                elapsed = time.perf_counter() - start
                rate    = i / elapsed
                eta     = (len(to_process) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(to_process)}] {rate:.1f} products/s — ETA {eta/60:.1f} min")

            if i % 1000 == 0:
                _save_cache(taxonomy_cache, TAXONOMY_CACHE_PATH)
                print(f"  [{i}/{len(to_process)}] checkpoint saved")

    _save_cache(taxonomy_cache, TAXONOMY_CACHE_PATH)
    elapsed = time.perf_counter() - start
    print(f"\nDone. {len(to_process)} products classified in {elapsed:.1f}s ({errors} errors).")
    print(f"Cache saved to: {TAXONOMY_CACHE_PATH}")

    _print_summary(taxonomy_cache)


def _print_summary(cache: dict) -> None:
    domain_counts    = Counter()
    category_counts  = Counter()
    no_category      = 0
    no_subcategory   = 0

    for v in cache.values():
        d  = v.get("taxonomy_domain",      "") or ""
        c  = v.get("taxonomy_category",    "") or ""
        sc = v.get("taxonomy_subcategory", "") or ""
        domain_counts[d or "Unknown"] += 1
        if c:
            category_counts[c] += 1
        else:
            no_category += 1
        if not sc:
            no_subcategory += 1

    print("\nDomain breakdown:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain:<15} {count}")

    source_counts = Counter(v.get("taxonomy_source", "legacy") for v in cache.values())
    print(f"\nProducts with full classification (category + subcategory): "
          f"{len(cache) - no_subcategory} / {len(cache)}")
    print(f"Products with domain only (low confidence or no attrs):    {no_category}")
    print(f"Products with Unknown domain:                               "
          f"{domain_counts.get('Unknown', 0)}")
    print(f"\nClassification source breakdown:")
    for src, count in source_counts.most_common():
        print(f"  {src:<15} {count}")

    print("\nTop 10 categories assigned:")
    for cat, count in category_counts.most_common(10):
        print(f"  {count:>6}  {cat}")

    sample = next(
        (v for v in cache.values()
         if v.get("taxonomy_subcategory")),
        None
    )
    if sample:
        print("\nSample result:")
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
