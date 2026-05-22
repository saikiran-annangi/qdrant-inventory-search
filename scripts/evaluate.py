"""
Evaluation against eval_queries.json (90 representative queries).

30 electrical / 30 mechanical / 30 plumbing.
10 model_number + 10 technical + 10 descriptive per domain.

The auto classifier is used by default (not the ground-truth type) for honest metrics.

Usage:
    python scripts/evaluate.py                # auto classifier, no reranker
    python scripts/evaluate.py --rerank       # with cross-encoder reranker
    python scripts/evaluate.py --gt-type      # use ground-truth type (oracle upper bound)
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier import classify_query
from models.embeddings import get_dense_model, get_bm25_model
from models.reranker import get_reranker
from core.search import search

EVAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_queries.json")


# ---------------------------------------------------------------------------
# Hit detection
# ---------------------------------------------------------------------------


def _matches(result: dict, expected: str) -> bool:
    if not expected:
        return False
    exp        = expected.lower().strip()
    p_model    = str(result.get("model_number",  "") or "").strip().lower()
    p_internal = str(result.get("internal_id",   "") or "").strip().lower()

    if p_model == exp or p_internal == exp:
        return True
    # Burnaby DC row-indexed IDs: "99999_0" matches expected "99999"
    if p_internal.startswith(exp + "_"):
        return True
    return False


def is_hit(result: dict, query: dict) -> bool:
    for field in ("expected_model_number", "expected_item_id", "expected_erp_code"):
        val = query.get(field) or ""
        if val and _matches(result, val):
            return True
    return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def reciprocal_rank(results: list, query: dict) -> float:
    for i, r in enumerate(results, 1):
        if is_hit(r, query):
            return 1.0 / i
    return 0.0


def recall_at_k(results: list, query: dict, k: int = 10) -> float:
    for r in results[:k]:
        if is_hit(r, query):
            return 1.0
    return 0.0


def precision_at_k(results: list, query: dict, k: int = 1) -> float:
    """Fraction of the top-k results that are hits.
    For single-answer queries this equals recall_at_k / k."""
    if not results:
        return 0.0
    hits = sum(1 for r in results[:k] if is_hit(r, query))
    return hits / k


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    use_reranker: bool = False,
    use_auto_classifier: bool = True,
    verbose: bool = True,
) -> dict:
    with open(EVAL_PATH) as f:
        data = json.load(f)

    queries = data["queries"]

    if verbose:
        print("Loading models...")
    get_dense_model()
    get_bm25_model()
    if use_reranker:
        get_reranker()

    classifier_mode = "auto" if use_auto_classifier else "ground-truth"
    if verbose:
        print(f"Evaluating {len(queries)} queries  |  reranker={'on' if use_reranker else 'off'}  |  classifier={classifier_mode}\n")

    acc: dict = {}
    detail_rows = []

    for q in queries:
        query_str = q["query"]
        domain    = q["domain"]
        gt_type   = q["type"]

        qtype = classify_query(query_str) if use_auto_classifier else gt_type

        hits = search(
            query_str,
            limit=10,
            query_type=qtype,
            use_reranker=use_reranker,
            rerank_top_k=50 if use_reranker else 10,
        )

        rr   = reciprocal_rank(hits, q)
        p1   = precision_at_k(hits, q, k=1)
        p5   = precision_at_k(hits, q, k=5)
        rec5 = recall_at_k(hits, q, k=5)
        rec  = recall_at_k(hits, q, k=10)

        key = (domain, gt_type)
        acc.setdefault(key, []).append((rr, p1, p5, rec5, rec))

        top1 = hits[0] if hits else {}
        detail_rows.append({
            "id":            q["id"],
            "domain":        domain,
            "gt_type":       gt_type,
            "classified_as": qtype,
            "correct_type":  qtype == gt_type,
            "query":         query_str,
            "rr":            round(rr, 4),
            "precision_at_1": round(p1, 4),
            "precision_at_5": round(p5, 4),
            "recall_at_5":   round(rec5, 4),
            "recall_at_10":  round(rec, 4),
            "hit":           rr > 0,
            "top1_model":    top1.get("model_number", ""),
            "top1_desc":     str(top1.get("description", ""))[:60],
            "expected":      (
                q.get("expected_model_number")
                or q.get("expected_item_id")
                or q.get("expected_erp_code")
                or ""
            ),
        })

        if verbose:
            marker     = "+" if rr > 0 else "x"
            cls_marker = "" if qtype == gt_type else f" [mis->{qtype}]"
            print(
                f"  {marker} [{domain:11s}|{gt_type:13s}]{cls_marker:15s} "
                f"{query_str[:52]:52s}  P@1={p1:.2f}  P@5={p5:.2f}  R@5={rec5:.1f}  R@10={rec:.1f}"
            )

    # Aggregation helper -- tuple layout: (rr, p1, p5, rec5, rec10)
    def avg(pairs, idx):
        return round(sum(x[idx] for x in pairs) / len(pairs), 4) if pairs else 0.0

    def agg(pairs):
        return {
            "MRR@10":      avg(pairs, 0),
            "Precision@1": avg(pairs, 1),
            "Precision@5": avg(pairs, 2),
            "Recall@5":    avg(pairs, 3),
            "Recall@10":   avg(pairs, 4),
            "n":           len(pairs),
        }

    summary: dict = {}

    all_pairs = [p for v in acc.values() for p in v]
    summary["overall"] = agg(all_pairs)

    for domain in ("electrical", "mechanical", "plumbing"):
        domain_pairs = [p for (d, _), v in acc.items() if d == domain for p in v]
        summary[domain] = agg(domain_pairs)

    for qtype in ("model_number", "technical", "descriptive"):
        type_pairs = [p for (_, t), v in acc.items() if t == qtype for p in v]
        summary[qtype] = agg(type_pairs)

    for (domain, qtype), pairs in sorted(acc.items()):
        summary[f"{domain}/{qtype}"] = agg(pairs)

    total_q     = len(detail_rows)
    correct_cls = sum(1 for r in detail_rows if r["correct_type"])
    summary["classifier_accuracy"] = round(correct_cls / total_q, 4)

    if verbose:
        print("\n" + "=" * 96)
        print(f"{'BREAKDOWN':<30} {'MRR@10':>8} {'P@1':>7} {'P@5':>7} {'R@5':>7} {'R@10':>7} {'N':>5}")
        print("-" * 96)
        for label in ["overall", "electrical", "mechanical", "plumbing",
                      "model_number", "technical", "descriptive"]:
            v = summary[label]
            print(f"  {label:<28} {v['MRR@10']:>8.4f} {v['Precision@1']:>7.4f} {v['Precision@5']:>7.4f} {v['Recall@5']:>7.4f} {v['Recall@10']:>7.4f} {v['n']:>5}")
        print("-" * 96)
        print(f"  {'domain x type':28}")
        for label in sorted(k for k in summary if "/" in k):
            v = summary[label]
            print(f"    {label:<26} {v['MRR@10']:>8.4f} {v['Precision@1']:>7.4f} {v['Precision@5']:>7.4f} {v['Recall@5']:>7.4f} {v['Recall@10']:>7.4f} {v['n']:>5}")
        print("=" * 96)
        print(f"  Query classifier accuracy: {summary['classifier_accuracy']:.1%}  ({correct_cls}/{total_q} correct)")

        misclassed = [r for r in detail_rows if not r["correct_type"]]
        if misclassed:
            print(f"\n  Misclassified queries ({len(misclassed)}):")
            for r in misclassed:
                print(f"    {r['id']} gt={r['gt_type']} -> got={r['classified_as']}  '{r['query'][:55]}'")

    return {"summary": summary, "details": detail_rows}


if __name__ == "__main__":
    use_reranker = "--rerank"  in sys.argv
    use_gt_type  = "--gt-type" in sys.argv

    results = run_evaluation(
        use_reranker=use_reranker,
        use_auto_classifier=not use_gt_type,
        verbose=True,
    )

    tag = ("_reranked" if use_reranker else "") + ("_gttype" if use_gt_type else "")
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f"eval_results{tag}.json",
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out}")
