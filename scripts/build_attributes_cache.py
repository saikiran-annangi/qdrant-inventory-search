"""
Pre-populate attributes_cache.json for all products using parallel LLM calls.

Run this once before ingestion so ingest.py makes zero LLM calls.

Usage:
    python scripts/build_attributes_cache.py
    python scripts/build_attributes_cache.py --sources guillevin_1 burnaby_dc
    python scripts/build_attributes_cache.py --limit 20   # test run
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # loads .env

from models.extractor import extract_product_attributes

_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_PATH  = os.path.join(_REPO_ROOT, "attributes_cache.json")

WORKERS = 100


def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    tmp = _CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, _CACHE_PATH)


def _extract(rec: dict) -> tuple[str, dict]:
    result = extract_product_attributes(
        model_number=         rec.get("model_number",         "") or "",
        description=          rec.get("description",          "") or "",
        extended_description= rec.get("extended_description", "") or "",
        manufacturer=         rec.get("manufacturer_name",    "") or "",
        product_category=     rec.get("product_category",     "") or "",
        source=               rec.get("source",               "") or "",
    )
    return rec["id"], result


def main():
    parser = argparse.ArgumentParser(description="Pre-populate attributes cache")
    parser.add_argument("--limit", type=int, help="Max products (for testing)")
    args = parser.parse_args()

    print("Loading inventory records...")
    from data.loaders import load_all
    records = load_all(verbose=True, attach_caches=False)

    if args.limit:
        records = records[:args.limit]
        print(f"  (limited to {args.limit} records for testing)")

    cache = _load_cache()
    print(f"\nAttributes cache: {len(cache)} existing entries")

    to_process = [r for r in records if r["id"] not in cache]
    print(f"Products needing extraction: {len(to_process)}")

    if not to_process:
        print("Cache is complete — nothing to do.")
        _print_summary(cache)
        return

    print(f"\nExtracting attributes ({len(to_process)} products, {WORKERS} workers)...")
    start  = time.perf_counter()
    errors = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_extract, rec): rec for rec in to_process}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                pid, result = fut.result()
                cache[pid] = result
            except Exception as exc:
                rec = futures[fut]
                print(f"  [ERROR] {rec['id']}: {exc}")
                cache[rec["id"]] = {"domain": "Unknown", "explicit": {}, "inferred": {}}
                errors += 1

            if i % 500 == 0:
                elapsed = time.perf_counter() - start
                rate    = i / elapsed
                eta     = (len(to_process) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(to_process)}] {rate:.1f} calls/s — ETA {eta/60:.1f} min")

            if i % 1000 == 0:
                _save_cache(cache)
                print(f"  [{i}/{len(to_process)}] checkpoint saved")

    _save_cache(cache)
    elapsed = time.perf_counter() - start
    print(f"\nDone. {len(to_process)} products extracted in {elapsed:.1f}s ({errors} errors).")
    print(f"Cache saved to: {_CACHE_PATH}")

    _print_summary(cache)


def _print_summary(cache: dict) -> None:
    from collections import Counter
    domains = Counter(v.get("domain", "Unknown") for v in cache.values())
    print("\nDomain breakdown:")
    for domain, count in domains.most_common():
        print(f"  {domain:<12} {count}")

    # Show a sample result
    sample = next(
        (v for v in cache.values() if v.get("domain") != "Unknown" and v.get("explicit")),
        None
    )
    if sample:
        print("\nSample result:")
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
