"""
Train and save the query-type logistic regression classifier.

Usage:
    python scripts/build_classifier.py           # train and save
    python scripts/build_classifier.py --eval    # also run on the 90 eval queries

Training data:
  model_number  -- real SKUs scrolled from the Qdrant index
  technical     -- eval queries labelled technical + spec-pattern descriptions from the catalog
  descriptive   -- eval queries labelled descriptive + plain descriptions from the catalog

Features:
  TF-IDF char n-grams (2-4) + TF-IDF word uni/bigrams + 10 handcrafted numeric features

Saved to: query_classifier.joblib (repo root)
"""

import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score

from config import CLASSIFIER_PATH, COLLECTION_NAME
from core.client import get_client

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH  = os.path.join(REPO_ROOT, "eval_queries.json")

MAX_MODEL_SAMPLES = 3000
MAX_DESC_SAMPLES  = 800

_SPEC_RE = re.compile(
    r"\b\d+\.?\d*\s*(A|KA|V|W|KW|MA|HP|VOLT|AMP)\b|"
    r"\b(MCB|RCD|RCBO|MCCB|VFD|GPF|GPM|IP\d{2}|NEMA|TXV|BTU|PSI)\b|"
    r"\b(SINGLE|DOUBLE|TRIPLE)\s+(POLE|PHASE)\b|"
    r"\b\d+[-x]\d+\b",
    re.IGNORECASE,
)

_MN_RE = re.compile(
    r"^[A-Z0-9]{2,}[-./+][A-Z0-9]|"
    r"^[A-Z]{1,4}[0-9]{3,}|"
    r"^[0-9]{3,}[A-Z]|"
    r"^[A-Z]{3,6}[0-9]{1,3}[A-Z]?$|"
    r"^\d{4,6}$|"
    r"^[A-Z]+-\d|"
    r"^[A-Z]+\s+\d+[A-Z]?$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Handcrafted numeric features
# ---------------------------------------------------------------------------


def _handcrafted(queries):
    rows = []
    for q in queries:
        n      = max(len(q), 1)
        words  = q.split()
        nw     = max(len(words), 1)
        rows.append([
            len(words),
            n,
            sum(c.isdigit() for c in q) / n,
            sum(c.isalpha() for c in q) / n,
            int(bool(re.search(r"[A-Z0-9][-./+][A-Z0-9]", q, re.I))),
            int(bool(re.search(r"\b\d+\.?\d*\s*(A|KA|V|W|KW|MA|HP|GPF|GPM)\b", q, re.I))),
            int(bool(re.search(r"\b(MCB|RCD|RCBO|IP\d{2}|NEMA|TXV|VFD|MCCB)\b", q, re.I))),
            max(len(w) for w in words) if words else 0,
            sum(1 for w in words if w.isupper()) / nw,
            int(len(words) <= 2),
        ])
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _load_eval_queries():
    with open(EVAL_PATH) as f:
        return json.load(f)["queries"]


def _collect_from_qdrant():
    client = get_client()
    print("Scrolling Qdrant for training data...")

    model_numbers: set = set()
    tech_descs:    list = []
    desc_descs:    list = []
    offset = None
    total  = 0

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            break

        for point in results:
            p    = point.payload
            mn   = str(p.get("model_number", "") or "").strip()
            desc = str(p.get("description", "") or "").strip()

            if len(mn) >= 3 and mn.lower() not in ("none", "null", "n/a"):
                model_numbers.add(mn)

            if len(desc) >= 10:
                lowered = desc.lower().strip()
                if _SPEC_RE.search(desc) and len(tech_descs) < MAX_DESC_SAMPLES:
                    tech_descs.append(lowered)
                elif not _SPEC_RE.search(desc) and len(desc_descs) < MAX_DESC_SAMPLES:
                    desc_descs.append(lowered)

        total  += len(results)
        offset  = next_offset
        if offset is None:
            break
        if (len(model_numbers) >= MAX_MODEL_SAMPLES
                and len(tech_descs) >= MAX_DESC_SAMPLES
                and len(desc_descs) >= MAX_DESC_SAMPLES):
            break

    print(f"  Scrolled {total} points -> "
          f"{len(model_numbers)} model numbers, "
          f"{len(tech_descs)} technical descs, "
          f"{len(desc_descs)} descriptive descs")

    mn_list = list(model_numbers)
    if len(mn_list) > MAX_MODEL_SAMPLES:
        rng     = np.random.default_rng(42)
        mn_list = list(rng.choice(mn_list, size=MAX_MODEL_SAMPLES, replace=False))

    return mn_list, tech_descs[:MAX_DESC_SAMPLES], desc_descs[:MAX_DESC_SAMPLES]


def _build_training_data():
    eval_queries               = _load_eval_queries()
    mn_list, tech_descs, desc_descs = _collect_from_qdrant()

    X, y = [], []

    for q in eval_queries:
        if q["type"] == "model_number":
            X.append(q["query"]); y.append("model_number")
    for mn in mn_list:
        X.append(mn); y.append("model_number")

    for q in eval_queries:
        if q["type"] == "technical":
            X.append(q["query"]); y.append("technical")
    for d in tech_descs:
        X.append(d); y.append("technical")

    for q in eval_queries:
        if q["type"] == "descriptive":
            X.append(q["query"]); y.append("descriptive")
    for d in desc_descs:
        X.append(d); y.append("descriptive")

    print(
        f"Training set: {y.count('model_number')} model_number | "
        f"{y.count('technical')} technical | "
        f"{y.count('descriptive')} descriptive | "
        f"total={len(X)}"
    )
    return X, y


# ---------------------------------------------------------------------------
# Sklearn pipeline
# ---------------------------------------------------------------------------


def _build_pipeline():
    features = FeatureUnion([
        ("char_ngrams", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=8000, sublinear_tf=True)),
        ("word_ngrams", TfidfVectorizer(analyzer="word",    ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("numeric",     FunctionTransformer(_handcrafted, validate=False)),
    ])
    clf = LogisticRegression(
        C=1.0, max_iter=2000, multi_class="multinomial",
        solver="lbfgs", class_weight="balanced", random_state=42,
    )
    return Pipeline([("features", features), ("clf", clf)])


def _evaluate_on_eval_set(pipeline):
    eval_queries = _load_eval_queries()
    correct, misses = 0, []
    for q in eval_queries:
        got = pipeline.predict([q["query"]])[0]
        if got == q["type"]:
            correct += 1
        else:
            misses.append((q["id"], q["type"], got, q["query"]))

    acc = correct / len(eval_queries)
    print(f"\nEval-set accuracy: {correct}/{len(eval_queries)} = {acc:.1%}")
    if misses:
        print(f"Misclassified ({len(misses)}):")
        for mid, gt, got, qtext in misses:
            print(f"  {mid}  gt={gt:13s} -> got={got:13s}  '{qtext[:55]}'")
    return acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    do_eval = "--eval" in sys.argv

    X, y = _build_training_data()

    print("\nTraining classifier...")
    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    if do_eval:
        _evaluate_on_eval_set(pipeline)

    joblib.dump(pipeline, CLASSIFIER_PATH)
    print(f"\nSaved -> {CLASSIFIER_PATH}")
    return pipeline


if __name__ == "__main__":
    main()
