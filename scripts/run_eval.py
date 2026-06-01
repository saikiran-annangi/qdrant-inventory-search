"""Run an eval CSV through the production search pipeline.

Pipeline used: OpenRouter Gemini classifier -> 3-retriever RRF -> cross-encoder
rerank -> size-aware sort. Scoring is pure string match between the retrieved
`internal_id` / `model_number` and the eval row's `expected_erp_id`.

Usage:
    python scripts/run_eval.py <eval.csv>             # defaults to mep_eval_300_v3.csv

The eval CSV must have columns: query_id, query_text, expected_erp_id,
domain, query_type, expected_description (last two are reported but
not used for scoring; classifier is run live)."""
import os, sys, csv, time, json, warnings, collections
warnings.filterwarnings("ignore"); os.environ["TOKENIZERS_PARALLELISM"] = "false"
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)

# load .env so OPENROUTER_API_KEY / QDRANT_URL+KEY (or QDRANT_LOCAL_PATH) are available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_R, ".env"))
except ImportError:
    pass

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "mep_eval_300_v3.csv")
OUT_LOG  = os.path.join(os.path.dirname(__file__), "eval_results.json")
KS = [5, 10, 50]

from scripts.evaluate import is_hit, mrr_at_k, recall_at_k, miss_at_k
from core.client import get_client
from core.search import search
from models.embeddings import get_dense_model, get_bm25_model
from models.reranker import get_reranker
from models.classifier import classify_query

# 1. Load CSV
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
queries = [{
    "id":     f"q{int(r['query_id']):04d}",
    "domain": r["domain"].strip(),
    "type":   r["query_type"].strip(),           # eval's ground-truth (for comparison only)
    "query":  r["query_text"],
    "expected_erp_code": (r.get("expected_erp_id") or "").strip(),
} for r in rows]
print(f"Loaded {len(queries)} queries from {os.path.basename(CSV_PATH)}")
print(f"Classifier: OpenRouter Gemini 2.5 Flash  (size-aware sort always on)")

# 2. ID compatibility (works for either local-embedded or cloud client)
client = get_client()
iid_set = set()
off = None
while True:
    pts, off = client.scroll("inventory", limit=2000, offset=off,
                             with_payload=["internal_id"], with_vectors=False)
    for p in pts: iid_set.add((p.payload.get("internal_id") or "").lower())
    if off is None: break
match_iid = sum(1 for q in queries if q["expected_erp_code"].lower() in iid_set)
print(f"ID compatibility: {match_iid}/{len(queries)} match internal_id verbatim")

# 3. Warm models
get_dense_model(); get_bm25_model(); get_reranker()
if not os.getenv("OPENROUTER_API_KEY"):
    raise SystemExit("OPENROUTER_API_KEY not set — aborting before issuing calls.")

# 4. Classify ALL queries first (so we get a single batch of API calls,
#    and progress is visible), then retrieve.
print("\nClassifying via OpenRouter Gemini 2.5 Flash...")
predicted_types = []
t0 = time.time()
for i, q in enumerate(queries, 1):
    try:
        pred = classify_query(q["query"])
    except Exception as e:
        print(f"  [warn] q{i} classify failed: {e}")
        pred = "descriptive"  # neutral fallback
    predicted_types.append(pred)
    if i % 30 == 0 or i == len(queries):
        rate = i / (time.time() - t0)
        print(f"  {i}/{len(queries)} classified  ({rate:.1f} q/s, {time.time()-t0:.0f}s)", flush=True)

# Classifier agreement (vs eval's ground-truth)
agreement = sum(1 for q, p in zip(queries, predicted_types) if q["type"] == p)
print(f"\nClassifier agreement with eval CSV's query_type: {agreement}/{len(queries)} ({100*agreement/len(queries):.1f}%)")
confusion = collections.Counter()
for q, p in zip(queries, predicted_types):
    confusion[(q["type"], p)] += 1
print("Confusion (truth -> pred):")
for (t, p), n in sorted(confusion.items()):
    mark = " <- mismatch" if t != p else ""
    print(f"  {t:>12} -> {p:<12} {n:>4}{mark}")

# 5. Retrieve with the classifier-predicted type
print("\nRetrieving...")
results = []
t0 = time.time()
for i, (q, pred_type) in enumerate(zip(queries, predicted_types), 1):
    try:
        hits = search(q["query"], limit=50, query_type=pred_type,
                      use_reranker=True, rerank_top_k=50)
    except Exception as e:
        hits = []
        if i <= 3: print(f"  [warn] q{i} search failed: {e}")
    row = {"domain": q["domain"], "type": q["type"], "pred_type": pred_type}
    for k in KS:
        row[f"mrr{k}"]  = mrr_at_k(hits, q, k)
        row[f"rec{k}"]  = recall_at_k(hits, q, k)
        row[f"miss{k}"] = miss_at_k(hits, q, k)
    results.append(row)
    if i % 30 == 0 or i == len(queries):
        rate = i / (time.time() - t0)
        print(f"  {i}/{len(queries)} ({rate:.1f} q/s, {time.time()-t0:.0f}s)", flush=True)

# 6. Aggregate
def agg(subset, label):
    n = len(subset)
    if not n: return
    f = lambda k: sum(r[k] for r in subset) / n
    cells = []
    for k in KS:
        cells.append(f"{f(f'mrr{k}'):6.4f} {f(f'rec{k}'):6.4f} {f(f'miss{k}'):6.4f}")
    print(f"  {label:<24}" + "  ".join(cells) + f"  {n:>5}")

hdr = f"  {'BREAKDOWN':<24}" + "  ".join(f"{'MRR@'+str(k):>6} {'R@'+str(k):>6} {'Miss'+str(k):>6}" for k in KS) + f"  {'N':>5}"
print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
agg(results, "overall")
for d in ("electrical", "mechanical", "plumbing"): agg([r for r in results if r["domain"] == d], d)
for t in ("model_number", "technical", "descriptive"): agg([r for r in results if r["type"] == t], t)
print("-" * len(hdr))
for d in ("electrical", "mechanical", "plumbing"):
    for t in ("model_number", "technical", "descriptive"):
        agg([r for r in results if r["domain"] == d and r["type"] == t], f"{d}/{t}")
print("=" * len(hdr))

# Persist per-query results for later inspection
json.dump({"results": results, "predicted_types": predicted_types,
           "queries":  queries, "agreement": agreement},
          open(OUT_LOG, "w"), indent=2)
print(f"\nPer-query log -> {OUT_LOG}")
