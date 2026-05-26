"""
Evaluation against eval_queries.json (90 representative queries).

30 electrical / 30 mechanical / 30 plumbing.
10 model_number + 10 technical + 10 descriptive per domain.

Metrics computed at k=3, k=10, and k=50 in a single pass.

Usage:
    python scripts/evaluate.py                # auto classifier, with reranker
    python scripts/evaluate.py --no-rerank    # without cross-encoder reranker
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
K_VALUES  = [3, 10, 50]   # evaluate at all three depths in one pass
FETCH_K   = max(K_VALUES)  # fetch this many results per query


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


def mrr_at_k(results: list, query: dict, k: int) -> float:
    """Reciprocal rank of the first hit within the top k. 0 if not found."""
    for i, r in enumerate(results[:k], 1):
        if is_hit(r, query):
            return 1.0 / i
    return 0.0


def recall_at_k(results: list, query: dict, k: int) -> float:
    """1.0 if the correct answer appears anywhere in the top k, else 0."""
    for r in results[:k]:
        if is_hit(r, query):
            return 1.0
    return 0.0


def miss_at_k(results: list, query: dict, k: int) -> float:
    """1.0 if the correct answer is absent from the top k (complete failure)."""
    return 1.0 - recall_at_k(results, query, k)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    use_reranker: bool = True,
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
        print(
            f"Evaluating {len(queries)} queries  |  "
            f"reranker={'on' if use_reranker else 'off'}  |  "
            f"classifier={classifier_mode}  |  "
            f"k={K_VALUES}\n"
        )

    acc: dict = {}
    detail_rows = []

    for q in queries:
        query_str = q["query"]
        domain    = q["domain"]
        gt_type   = q["type"]

        qtype = classify_query(query_str) if use_auto_classifier else gt_type

        # Fetch FETCH_K (50) results once — evaluate at all depths from this list
        hits = search(
            query_str,
            limit=FETCH_K,
            query_type=qtype,
            use_reranker=use_reranker,
            rerank_top_k=50 if use_reranker else FETCH_K,
        )

        # Compute metrics at every depth
        metrics = {}
        for k in K_VALUES:
            metrics[k] = {
                "mrr":  mrr_at_k(hits, q, k),
                "rec":  recall_at_k(hits, q, k),
                "miss": miss_at_k(hits, q, k),
            }

        # Accumulator stores one tuple per query: metrics at each k in order
        # layout: (mrr3, rec3, miss3, mrr10, rec10, miss10, mrr50, rec50, miss50)
        key = (domain, gt_type)
        acc.setdefault(key, []).append(tuple(
            metrics[k][m] for k in K_VALUES for m in ("mrr", "rec", "miss")
        ))

        top1 = hits[0] if hits else {}
        detail_rows.append({
            "id":            q["id"],
            "domain":        domain,
            "gt_type":       gt_type,
            "classified_as": qtype,
            "correct_type":  qtype == gt_type,
            "query":         query_str,
            **{f"mrr_at_{k}":  round(metrics[k]["mrr"],  4) for k in K_VALUES},
            **{f"recall_at_{k}": round(metrics[k]["rec"], 4) for k in K_VALUES},
            **{f"miss_at_{k}": round(metrics[k]["miss"],  4) for k in K_VALUES},
            "hit_at_3":   metrics[3]["rec"]  > 0,
            "hit_at_10":  metrics[10]["rec"] > 0,
            "hit_at_50":  metrics[50]["rec"] > 0,
            "top1_model": top1.get("model_number", ""),
            "top1_desc":  str(top1.get("description", ""))[:60],
            "expected": (
                q.get("expected_model_number")
                or q.get("expected_item_id")
                or q.get("expected_erp_code")
                or ""
            ),
        })

        if verbose:
            marker     = "+" if metrics[3]["rec"] > 0 else "x"
            cls_marker = "" if qtype == gt_type else f" [mis->{qtype}]"
            print(
                f"  {marker} [{domain:11s}|{gt_type:13s}]{cls_marker:15s} "
                f"{query_str[:50]:50s}  "
                f"MRR@3={metrics[3]['mrr']:.2f}  "
                f"R@3={metrics[3]['rec']:.0f}  "
                f"R@10={metrics[10]['rec']:.0f}  "
                f"R@50={metrics[50]['rec']:.0f}"
            )

    # Aggregation
    # tuple indices: 0=mrr3, 1=rec3, 2=miss3, 3=mrr10, 4=rec10, 5=miss10, 6=mrr50, 7=rec50, 8=miss50
    IDX = {k: {m: i for i, m in enumerate(("mrr","rec","miss"), base)}
           for k, base in zip(K_VALUES, [0, 3, 6])}

    def avg(pairs, idx):
        return round(sum(x[idx] for x in pairs) / len(pairs), 4) if pairs else 0.0

    def agg(pairs):
        out = {"n": len(pairs)}
        for k in K_VALUES:
            out[f"MRR@{k}"]    = avg(pairs, IDX[k]["mrr"])
            out[f"Recall@{k}"]  = avg(pairs, IDX[k]["rec"])
            out[f"Miss@{k}"]    = avg(pairs, IDX[k]["miss"])
        return out

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
        w = 100
        print("\n" + "=" * w)
        hdr = f"{'BREAKDOWN':<30}"
        for k in K_VALUES:
            hdr += f"  {'MRR@'+str(k):>7} {'R@'+str(k):>6} {'Miss@'+str(k):>7}"
        hdr += f"  {'N':>4}"
        print(hdr)
        print("-" * w)
        for label in ["overall", "electrical", "mechanical", "plumbing",
                      "model_number", "technical", "descriptive"]:
            v = summary[label]
            row = f"  {label:<28}"
            for k in K_VALUES:
                row += f"  {v[f'MRR@{k}']:>7.4f} {v[f'Recall@{k}']:>6.4f} {v[f'Miss@{k}']:>7.4f}"
            row += f"  {v['n']:>4}"
            print(row)
        print("-" * w)
        print(f"  {'domain x type':<28}")
        for label in sorted(lbl for lbl in summary if "/" in lbl):
            v = summary[label]
            row = f"    {label:<26}"
            for k in K_VALUES:
                row += f"  {v[f'MRR@{k}']:>7.4f} {v[f'Recall@{k}']:>6.4f} {v[f'Miss@{k}']:>7.4f}"
            row += f"  {v['n']:>4}"
            print(row)
        print("=" * w)
        print(f"  Classifier accuracy: {summary['classifier_accuracy']:.1%}  ({correct_cls}/{total_q} correct)")

        misclassed = [r for r in detail_rows if not r["correct_type"]]
        if misclassed:
            print(f"\n  Misclassified queries ({len(misclassed)}):")
            for r in misclassed:
                print(f"    {r['id']} gt={r['gt_type']} -> got={r['classified_as']}  '{r['query'][:55]}'")

    return {"summary": summary, "details": detail_rows}


if __name__ == "__main__":
    use_reranker = "--no-rerank" not in sys.argv
    use_gt_type  = "--gt-type"   in sys.argv

    results = run_evaluation(
        use_reranker=use_reranker,
        use_auto_classifier=not use_gt_type,
        verbose=True,
    )

    tag = ("" if use_reranker else "_no_rerank") + ("_gttype" if use_gt_type else "")
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f"eval_results{tag}.json",
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out}")
