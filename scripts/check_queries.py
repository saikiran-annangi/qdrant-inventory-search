"""Quick smoke test — run the 3 manager-reported queries and print top-5 results.

Usage:  python scripts/check_queries.py

Connects to whatever backend .env points at (local or cloud).
Exits 0 if all 3 expected SKUs appear in the top-5, else 1.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.search import search

TESTS = [
    {
        # "Dry connectors" has no size — any BX/FLEX connector is valid.
        # CI2116 is the manager's example but NES-156 (also 2" BX/FLEX) is
        # equally correct. We check the whole BX/FLEX family surfaces.
        "query":    "Dry connectors - 8",
        "expect":   ["ABB-CI2116", "NES-156", "NES-686-DC2", "ABB-CI2236",
                    "ABB-CI2167", "ABB-CI2172", "ABB-CI2169", "ABB-CI2214"],
        "match":    "any",
        "note":     "BX/FLEX connector — 'dry' trade term (any BX/FLEX is valid)",
    },
    {
        "query":    "Dry connectors 2in - 8",
        "expect":   ["ABB-CI2116"],
        "match":    "any",
        "note":     "BX/FLEX 2\" — with size, CI2116 hits rank 1",
    },
    {
        "query":    "P Clamps for 250 - 20",
        "expect":   ["ABB-CPC150"],
        "match":    "any",
        "note":     "P-clamp for 250 MCM cable",
    },
    {
        "query":    "P Clamps for 250 MCM - 20",
        "expect":   ["ABB-CPC150"],
        "match":    "any",
        "note":     "Same, with explicit MCM unit",
    },
]

LIMIT = 5
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

passed = 0
failed = 0

print(f"\nRunning {len(TESTS)} smoke-test queries (top {LIMIT})...\n")
print("─" * 70)

for t in TESTS:
    raw_query = t["query"]
    expected  = t["expect"]   # list

    results = search(raw_query, limit=LIMIT, use_reranker=True)
    ids = [r.get("internal_id", r.get("id", "")) for r in results]

    hits = [e for e in expected if e in ids]
    hit  = bool(hits)
    rank = min((ids.index(e) + 1 for e in expected if e in ids), default=None)

    status = f"{GREEN}PASS  rank={rank}{RESET}" if hit else f"{RED}FAIL  none of {expected} in top-{LIMIT}{RESET}"
    print(f"  Query  : {raw_query!r}")
    print(f"  Expect : {expected}  ({t['note']})")
    print(f"  Result : {status}")
    print(f"  Top-{LIMIT}  : {ids}")
    print()

    if hit:
        passed += 1
    else:
        failed += 1

print("─" * 70)
print(f"  {passed}/{len(TESTS)} passed", end="")
if failed:
    print(f"  ({RED}{failed} failed{RESET})")
else:
    print(f"  {GREEN}✓ all good{RESET}")
print()

sys.exit(0 if failed == 0 else 1)
