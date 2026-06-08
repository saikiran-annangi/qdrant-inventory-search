"""Guard the taxonomy invariants so drift can never silently return.

The root-cause bug this prevents: the query side and the product side drawing
from different taxonomies. These tests enforce that there is ONE vocabulary and
everything lines up with it. Exits non-zero on failure (CI-friendly).

Run:  python tests/test_taxonomy_consistency.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PRODUCT_TAXONOMY, CATEGORY_MAP

_VALID_DOMAINS = {"Electrical", "Mechanical", "Plumbing", "Tools & Site"}


def _all_nodes() -> set:
    nodes = set()
    for domain, cats in PRODUCT_TAXONOMY.items():
        for cat, subs in cats.items():
            for sub in subs:
                nodes.add((domain, cat, sub))
    return nodes


def test_domains_are_known():
    bad = set(PRODUCT_TAXONOMY) - _VALID_DOMAINS
    assert not bad, f"Unknown domains in PRODUCT_TAXONOMY: {bad}"


def test_category_map_targets_exist():
    """Every non-fallback CATEGORY_MAP target must be a real taxonomy node."""
    nodes = _all_nodes()
    bad = []
    for erp, target in CATEGORY_MAP.items():
        if target[0] is None:
            continue
        if tuple(target) not in nodes:
            bad.append((erp, target))
    assert not bad, (
        f"{len(bad)} CATEGORY_MAP targets not in PRODUCT_TAXONOMY: {bad[:10]}"
    )


def test_no_duplicate_subcategory_within_category():
    dups = []
    for domain, cats in PRODUCT_TAXONOMY.items():
        for cat, subs in cats.items():
            if len(subs) != len(set(subs)):
                dups.append((domain, cat))
    assert not dups, f"Duplicate subcategories within category: {dups}"


def _query_labels() -> set:
    import models.query_taxonomy_llm as q
    q._labels_loaded = False  # force reload
    q._load_labels()
    return {l for labs in q._labels_by_domain.values() for l in labs}


def _store_labels_or_none():
    """Labels the open store currently exposes, or None if no store built yet."""
    from config import TAXONOMY_LABELS_PATH
    if not os.path.exists(TAXONOMY_LABELS_PATH):
        return None
    import json
    with open(TAXONOMY_LABELS_PATH) as f:
        data = json.load(f)
    return {l for labs in data.values() for l in labs}


def _seed_labels() -> set:
    return {f"{c} > {s}"
            for cats in PRODUCT_TAXONOMY.values()
            for c, subs in cats.items() for s in subs}


def test_query_classifier_matches_single_source():
    """The query classifier must offer EXACTLY the labels of the live single
    source — the open store if built, otherwise the curated seed. This is the
    invariant that prevents query/product drift (now that the vocabulary grows)."""
    source = _store_labels_or_none()
    if source is None:
        source = _seed_labels()
    got = _query_labels()
    assert got == source, (
        f"Query labels differ from the single source: "
        f"missing={sorted(source - got)[:10]}, extra={sorted(got - source)[:10]}"
    )


def test_store_superset_of_seed():
    """The open store, once built, must contain every curated seed node (it only
    ever grows from the seed — never drops curated nodes)."""
    store = _store_labels_or_none()
    if store is None:
        return  # no store yet — nothing to check
    missing = _seed_labels() - store
    assert not missing, f"Store is missing seed nodes: {sorted(missing)[:10]}"


def test_query_and_ingest_vocabularies_identical():
    """The decisive no-mismatch guarantee: every label the INGEST side can write
    must be a label the QUERY side can emit.

    Ingest labels = the single source (store or seed) ∪ CATEGORY_MAP targets;
    the match/mint classifier only ever assigns nodes that live in the store, and
    deterministic mappings target seed nodes (⊆ store). Query reads the same
    source. So ingest ⊆ query, with zero unreachable product labels.
    """
    query_labels = _query_labels()
    source = _store_labels_or_none()
    if source is None:
        source = _seed_labels()

    ingest_labels = set(source)
    for d, c, s in CATEGORY_MAP.values():
        if c and s:
            ingest_labels.add(f"{c} > {s}")

    unreachable = ingest_labels - query_labels
    assert not unreachable, f"Product labels unreachable by query: {sorted(unreachable)[:10]}"


def test_category_map_keys_unique_and_stripped():
    for erp in CATEGORY_MAP:
        assert erp == erp.strip(), f"ERP key has surrounding whitespace: {erp!r}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    nodes = _all_nodes()
    mapped = sum(1 for v in CATEGORY_MAP.values() if v[0] is not None)
    print(
        f"\nTaxonomy: {len(nodes)} nodes across {len(PRODUCT_TAXONOMY)} domains; "
        f"CATEGORY_MAP: {len(CATEGORY_MAP)} ERP categories ({mapped} mapped)."
    )
    if failed:
        print(f"\n{failed} test(s) FAILED")
        sys.exit(1)
    print("\nAll taxonomy consistency tests passed.")
