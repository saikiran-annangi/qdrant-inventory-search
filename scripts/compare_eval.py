"""
Parallel A/B eval:
  A (Baseline) — cloud Qdrant, current 450-token domain-aware enrichments
  B (PrefPO)   — local Qdrant, re-enriched with PrefPO 366-token prompt

Timeline (parallel):
  T=0   cloud eval starts  +  PrefPO re-enrich starts
  T+5   re-enrich done     →  local ingest starts
  T+15  cloud eval done
  T+25  local ingest done  →  local eval starts
  T+40  local eval done    →  comparison table printed

Usage:
    python scripts/compare_eval.py
"""

import json, math, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

EVAL_CSV   = os.path.join(REPO, "scripts", "mep_eval_300_v3.csv")
CLOUD_JSON = os.path.join(REPO, "scripts", "eval_results_cloud_baseline.json")
LOCAL_JSON = os.path.join(REPO, "scripts", "eval_results_local_prefpo.json")
LOG_DIR    = "/tmp"
KS         = [5, 10, 50]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_bg(cmd, log_file, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    f = open(log_file, "w")
    return subprocess.Popen(cmd, shell=True, cwd=REPO, env=env,
                            stdout=f, stderr=subprocess.STDOUT)


def wait(proc, label, log_file, poll_secs=15):
    log(f"  waiting for {label}...")
    t0 = time.time()
    while proc.poll() is None:
        time.sleep(poll_secs)
        elapsed = int(time.time() - t0)
        # show last progress line from log
        try:
            lines = open(log_file).readlines()
            last = next((l.strip() for l in reversed(lines) if l.strip() and
                         not any(x in l for x in ["WARNING","INFO","TOKENIZER","deprecated"])), "")
            if last:
                print(f"    [{elapsed:3d}s] {last[:90]}", flush=True)
        except Exception:
            pass
    rc = proc.returncode
    elapsed = int(time.time() - t0)
    status = "✓" if rc == 0 else f"✗ (exit {rc})"
    log(f"  {label} done in {elapsed}s {status}")
    return rc == 0


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def is_hit(result_ids, expected):
    exp_l = expected.lower()
    for rid in result_ids:
        if rid.lower() == exp_l:
            return True
    return False


def compute_metrics(results, k):
    mrr_sum = recall_sum = miss_sum = 0
    n = len(results)
    for r in results:
        ids = r["retrieved_ids"][:k]
        if is_hit(ids, r["expected"]):
            rank = next(i+1 for i,rid in enumerate(ids)
                        if rid.lower() == r["expected"].lower())
            mrr_sum += 1.0 / rank
            recall_sum += 1
        else:
            miss_sum += 1
    return mrr_sum/n, recall_sum/n, miss_sum/n


def load_results(path):
    data = json.load(open(path))
    out = []
    for item in data:
        out.append({
            "domain":       item.get("domain", ""),
            "type":         item.get("type", ""),
            "expected":     item.get("expected_erp_code", ""),
            "retrieved_ids": [h.get("internal_id","") for h in item.get("hits",[])]
        })
    return out


def print_table(label_a, label_b, results_a, results_b):
    domains = ["overall", "electrical", "mechanical", "plumbing"]

    def get_subset(results, domain):
        if domain == "overall":
            return results
        return [r for r in results if r["domain"].lower() == domain]

    # Header
    col_w = max(len(label_a), len(label_b), 12)
    metric_cols = "  ".join(f"{'MRR@'+str(k):>7} {'R@'+str(k):>7} {'Miss@'+str(k):>7}" for k in KS)
    sep = "─" * (col_w + 4 + len(metric_cols) + 10)

    print(f"\n{'='*len(sep)}")
    print(f"  ENRICHMENT PROMPT A/B EVALUATION — {len(results_a)} queries")
    print(f"  A (Baseline): {label_a}")
    print(f"  B (PrefPO)  : {label_b}")
    print(f"{'='*len(sep)}")
    print(f"\n  {'Domain':<14}  {'Prompt':<{col_w}}  {metric_cols}")
    print(f"  {sep}")

    for domain in domains:
        sub_a = get_subset(results_a, domain)
        sub_b = get_subset(results_b, domain)
        if not sub_a or not sub_b:
            continue
        for label, sub in [(label_a, sub_a), (label_b, sub_b)]:
            cells = "  ".join(
                f"{compute_metrics(sub,k)[0]:>7.3f} {compute_metrics(sub,k)[1]:>7.3f} {compute_metrics(sub,k)[2]:>7.3f}"
                for k in KS
            )
            dom_label = domain if label == label_a else ""
            marker = " ←" if label == label_b else ""
            print(f"  {dom_label:<14}  {label:<{col_w}}  {cells}{marker}")
        print(f"  {sep}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("=== PARALLEL A/B EVAL ===")
    log("A: cloud Qdrant (current 450-token domain-aware enrichment)")
    log("B: local Qdrant (PrefPO 366-token — re-enrich + re-ingest)")
    print()

    # ── PHASE 1: cloud eval + PrefPO re-enrich in parallel ────────────────
    log("[PHASE 1] Starting cloud eval + PrefPO re-enrich in parallel")

    cloud_eval = run_bg(
        f"python scripts/run_eval.py {EVAL_CSV}",
        f"{LOG_DIR}/eval_cloud.log",
    )
    log("  cloud eval started (PID {})".format(cloud_eval.pid))

    # Delete old cache so PrefPO prompt re-enriches everything fresh
    cache_path = os.path.join(REPO, "enrichment_cache.json")
    if os.path.exists(cache_path):
        os.rename(cache_path, cache_path + ".bak_compare")
        log("  enrichment_cache.json moved to .bak_compare")

    enrich = run_bg(
        "python scripts/enrich_descriptions.py",
        f"{LOG_DIR}/enrich_prefpo.log",
    )
    log(f"  PrefPO re-enrich started (PID {enrich.pid})")

    # Wait for enrichment (usually ~5 min at 133/s)
    ok = wait(enrich, "PrefPO re-enrich", f"{LOG_DIR}/enrich_prefpo.log")
    if not ok:
        log("ERROR: enrichment failed — check /tmp/enrich_prefpo.log")
        sys.exit(1)

    # ── PHASE 2: local ingest (while cloud eval still running) ─────────────
    log("[PHASE 2] Starting local ingest with PrefPO enrichments")

    ingest = run_bg(
        "python scripts/ingest.py",
        f"{LOG_DIR}/ingest_prefpo.log",
        extra_env={"QDRANT_LOCAL_PATH": "./qdrant_local"},
    )
    log(f"  local ingest started (PID {ingest.pid})")

    # Wait for ingest
    ok = wait(ingest, "local ingest", f"{LOG_DIR}/ingest_prefpo.log", poll_secs=20)
    if not ok:
        log("ERROR: local ingest failed — check /tmp/ingest_prefpo.log")
        sys.exit(1)

    # ── PHASE 3: local eval ────────────────────────────────────────────────
    log("[PHASE 3] Starting local eval (PrefPO)")

    local_eval = run_bg(
        f"python scripts/run_eval.py {EVAL_CSV}",
        f"{LOG_DIR}/eval_local.log",
        extra_env={"QDRANT_LOCAL_PATH": "./qdrant_local"},
    )
    log(f"  local eval started (PID {local_eval.pid})")

    # Wait for cloud eval (should be done by now, or nearly so)
    if cloud_eval.poll() is None:
        log("[waiting] cloud eval still running...")
        wait(cloud_eval, "cloud eval", f"{LOG_DIR}/eval_cloud.log")
    else:
        log("  cloud eval already finished")

    # Save cloud results
    src = os.path.join(REPO, "scripts", "eval_results.json")
    if os.path.exists(src):
        import shutil
        shutil.copy(src, CLOUD_JSON)
        log(f"  cloud results saved → {os.path.basename(CLOUD_JSON)}")

    # Wait for local eval
    ok = wait(local_eval, "local eval", f"{LOG_DIR}/eval_local.log")
    if not ok:
        log("ERROR: local eval failed — check /tmp/eval_local.log")
        sys.exit(1)

    # Save local results
    if os.path.exists(src):
        import shutil
        shutil.copy(src, LOCAL_JSON)
        log(f"  local results saved → {os.path.basename(LOCAL_JSON)}")

    # ── PHASE 4: print comparison table ───────────────────────────────────
    log("[PHASE 4] Printing comparison table")
    print()

    results_a = load_results(CLOUD_JSON)
    results_b = load_results(LOCAL_JSON)

    print_table(
        label_a="Cloud 450-tok (baseline)",
        label_b="Local PrefPO 366-tok",
        results_a=results_a,
        results_b=results_b,
    )

    log("Done. Log files: /tmp/eval_cloud.log  /tmp/eval_local.log")
    log("Results: scripts/eval_results_cloud_baseline.json  scripts/eval_results_local_prefpo.json")


if __name__ == "__main__":
    main()
