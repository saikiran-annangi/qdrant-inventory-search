"""
Taxonomy mapper: classifies a product's raw Phase-1 attributes into the
predefined domain → category → subcategory taxonomy.

Two-stage pipeline:
  1. Per-attribute RRF voting   — each extracted attribute independently ranks
                                  domain-filtered taxonomy nodes by cosine
                                  similarity; ranks are fused via RRF.
  2. Cross-encoder rerank       — top-3 RRF candidates are re-scored by the
                                  ms-marco cross-encoder already used for search.

Public API:
    map_to_taxonomy(raw_attrs: dict) -> dict
        raw_attrs: Phase-1 output {"domain":..., "explicit":{...}, "inferred":{...}}
        returns:   {"taxonomy_domain":..., "taxonomy_category":..., "taxonomy_subcategory":...}
"""

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Cross-encoder score below this threshold → store empty subcategory rather
# than force a wrong match. ms-marco logits: ~[-10, +10]; -3 is low confidence.
_CONFIDENCE_THRESHOLD = -5.5

_EMPTY = {"taxonomy_domain": "", "taxonomy_category": "", "taxonomy_subcategory": ""}

# Singleton state
_nodes:   list  = []   # all taxonomy nodes with vectors
_by_domain: dict = {}  # domain → list of node indices into _nodes

# LLM fallback — OpenRouter client singleton
_openrouter_client = None


def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        import os
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _openrouter_client


# Cached formatted string of all predefined taxonomy nodes (built once)
_EXISTING_NODES_TEXT: str = ""


def _get_existing_nodes_text() -> str:
    global _EXISTING_NODES_TEXT
    if _EXISTING_NODES_TEXT:
        return _EXISTING_NODES_TEXT
    from config import PRODUCT_TAXONOMY
    lines = []
    for domain, categories in PRODUCT_TAXONOMY.items():
        for category, subcategories in categories.items():
            for subcategory in subcategories:
                lines.append(f"  {domain} > {category} > {subcategory}")
    _EXISTING_NODES_TEXT = "\n".join(lines)
    return _EXISTING_NODES_TEXT


_LLM_FALLBACK_PROMPT = """\
You are classifying an industrial product into a taxonomy.

These taxonomy nodes already exist (shown for style reference only — \
this product could NOT be confidently mapped to any of them):
{existing_nodes}

Product details:
  Model Number: {model_number}
  Description: {description}
{extended_line}\
  Manufacturer: {manufacturer}
  Category: {product_category}

Extracted attributes:
  Domain: {domain}
{attrs_lines}
Create a NEW category and subcategory for this product.
Requirements:
  - Domain must be one of: Electrical, Mechanical, Plumbing
  - Category: 2-5 words, title case
  - Subcategory: 2-5 words, title case, more specific than category
  - Match the naming style of the existing nodes listed above

Return JSON only: {{"taxonomy_domain": "...", "taxonomy_category": "...", "taxonomy_subcategory": "..."}}"""


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _load_taxonomy_nodes() -> None:
    global _nodes, _by_domain
    if _nodes:
        return

    from config import TAXONOMY_EMBEDDINGS_PATH
    if not os.path.exists(TAXONOMY_EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"taxonomy_embeddings.json not found at {TAXONOMY_EMBEDDINGS_PATH}. "
            "Run scripts/build_taxonomy_embeddings.py first."
        )

    with open(TAXONOMY_EMBEDDINGS_PATH) as f:
        data = json.load(f)

    for key, entry in data.items():
        idx = len(_nodes)
        _nodes.append({
            "key":        key,
            "domain":     entry["domain"],
            "category":   entry["category"],
            "subcategory": entry["subcategory"],
            "text":       entry["text"],
            "vector":     np.array(entry["vector"], dtype=np.float32),
        })
        _by_domain.setdefault(entry["domain"], []).append(idx)

    logger.info("Taxonomy nodes loaded: %d total", len(_nodes))


def _get_encoder():
    from config import DENSE_MODEL_NAME
    from sentence_transformers import SentenceTransformer
    # Reuse a module-level singleton to avoid reloading the model per call.
    if not hasattr(_get_encoder, "_model"):
        _get_encoder._model = SentenceTransformer(DENSE_MODEL_NAME)
    return _get_encoder._model


def _get_reranker():
    from models.reranker import get_reranker
    return get_reranker()


# ---------------------------------------------------------------------------
# Stage 1 — per-attribute RRF voting
# ---------------------------------------------------------------------------

def _cosine_sim(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """vec: (D,), matrix: (N, D) — both already L2-normalised."""
    return matrix @ vec  # dot product == cosine sim for unit vectors


def _rrf_top_k(attr_texts: list[str], domain_indices: list[int], k: int = 3) -> list[int]:
    """
    For each attribute text, rank domain nodes by cosine similarity.
    Combine all per-attribute rankings with RRF (k=60).
    Return indices (into _nodes) of the top-k candidates.
    """
    if not attr_texts or not domain_indices:
        return []

    encoder = _get_encoder()
    domain_vectors = np.stack([_nodes[i]["vector"] for i in domain_indices])  # (N, D)

    rrf_scores: dict[int, float] = {}
    RRF_K = 60

    # Batch-encode all attribute texts in one forward pass — much faster than
    # individual encode() calls per attribute.
    vecs = encoder.encode(attr_texts, normalize_embeddings=True, batch_size=len(attr_texts))
    for vec in vecs:
        sims = _cosine_sim(vec, domain_vectors)           # (N,)
        ranked = np.argsort(sims)[::-1]                   # best first
        for rank, local_idx in enumerate(ranked, 1):
            global_idx = domain_indices[local_idx]
            rrf_scores[global_idx] = rrf_scores.get(global_idx, 0.0) + 1.0 / (RRF_K + rank)

    top = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return top


# ---------------------------------------------------------------------------
# Stage 2 — cross-encoder rerank
# ---------------------------------------------------------------------------

def _rerank_candidates(product_text: str, candidate_indices: list[int]) -> tuple[int, float]:
    """
    Score (product_text, taxonomy_node_text) pairs with the cross-encoder.
    Returns (best_index_into_nodes, best_score).
    """
    reranker = _get_reranker()
    pairs = [(product_text, _nodes[i]["text"]) for i in candidate_indices]
    scores = reranker.predict(pairs)
    best_pos = int(np.argmax(scores))
    return candidate_indices[best_pos], float(scores[best_pos])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_to_taxonomy(raw_attrs: dict) -> dict:
    """
    Map Phase-1 raw attributes to the predefined taxonomy.

    Args:
        raw_attrs: {"domain": str, "explicit": dict, "inferred": dict}

    Returns:
        {"taxonomy_domain": str, "taxonomy_category": str, "taxonomy_subcategory": str}
        Any field may be empty string if classification is not possible/confident.
    """
    _load_taxonomy_nodes()

    domain = raw_attrs.get("domain", "Unknown")

    # Gate 1: unknown domain — can't filter nodes safely
    if domain not in ("Electrical", "Mechanical", "Plumbing"):
        return {"taxonomy_domain": "Unknown", "taxonomy_category": "", "taxonomy_subcategory": "", "confidence_score": None}

    domain_indices = _by_domain.get(domain, [])
    if not domain_indices:
        return {"taxonomy_domain": domain, "taxonomy_category": "", "taxonomy_subcategory": "", "confidence_score": None}

    # Merge attributes — explicit wins over inferred for same key
    merged = {**raw_attrs.get("inferred", {}), **raw_attrs.get("explicit", {})}
    merged = {k: v for k, v in merged.items() if v}  # drop null/empty values

    # Gate 2: no attributes to vote with
    if not merged:
        return {"taxonomy_domain": domain, "taxonomy_category": "", "taxonomy_subcategory": "", "confidence_score": None}

    # Build per-attribute text strings for RRF voting
    attr_texts = [f"{k}: {v}" for k, v in merged.items()]

    # Stage 1: RRF → top 3 candidates
    top3 = _rrf_top_k(attr_texts, domain_indices, k=3)
    if not top3:
        return {"taxonomy_domain": domain, "taxonomy_category": "", "taxonomy_subcategory": "", "confidence_score": None}

    # Stage 2: cross-encoder rerank
    product_text = " | ".join(attr_texts)
    best_idx, best_score = _rerank_candidates(product_text, top3)

    # Gate 3: low confidence → store domain only, not a forced wrong subcategory
    if best_score < _CONFIDENCE_THRESHOLD:
        logger.debug(
            "Low confidence (%.2f) for %s — storing domain only", best_score, product_text[:80]
        )
        return {"taxonomy_domain": domain, "taxonomy_category": "", "taxonomy_subcategory": "", "confidence_score": round(best_score, 4)}

    node = _nodes[best_idx]
    return {
        "taxonomy_domain":      node["domain"],
        "taxonomy_category":    node["category"],
        "taxonomy_subcategory": node["subcategory"],
        "confidence_score":     round(best_score, 4),
    }


def llm_fallback_taxonomy(product_fields: dict, attrs: dict) -> dict:
    """
    LLM fallback for products that couldn't be mapped via cosine similarity.

    Called only when map_to_taxonomy() returns empty category/subcategory.
    Uses Gemini 2.5 Flash to invent a new category and subcategory that doesn't
    exist in the predefined taxonomy but is stylistically consistent with it.

    Args:
        product_fields: dict with model_number, description, extended_description,
                        manufacturer_name, product_category (from raw product data)
        attrs:          Phase-1 output dict with domain, explicit, inferred
                        (from attributes_cache.json)

    Returns:
        {taxonomy_domain, taxonomy_category, taxonomy_subcategory}
        Falls back to domain-only on any failure.
    """
    import re
    import json as _json
    import time

    client = _get_openrouter_client()
    fallback_domain = attrs.get("domain", "") if attrs.get("domain") in ("Electrical", "Mechanical", "Plumbing") else ""

    if client is None:
        return {"taxonomy_domain": fallback_domain, "taxonomy_category": "", "taxonomy_subcategory": ""}

    merged = {**attrs.get("inferred", {}), **attrs.get("explicit", {})}
    merged = {k: v for k, v in merged.items() if v}
    attrs_lines = "".join(f"  {k}: {v}\n" for k, v in merged.items()) if merged else ""

    ext = (product_fields.get("extended_description") or "")[:300]
    extended_line = f"  Extended Description: {ext}\n" if ext else ""

    prompt = _LLM_FALLBACK_PROMPT.format(
        existing_nodes=   _get_existing_nodes_text(),
        model_number=     (product_fields.get("model_number")     or "")[:80],
        description=      (product_fields.get("description")      or "")[:300],
        extended_line=    extended_line,
        manufacturer=     (product_fields.get("manufacturer_name") or "")[:80],
        product_category= (product_fields.get("product_category") or "")[:80],
        domain=           attrs.get("domain", "Unknown"),
        attrs_lines=      attrs_lines,
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = resp.choices[0].message.content or ""
            match = re.search(r'\{.*\}', raw.strip(), re.DOTALL)
            if not match:
                break
            parsed = _json.loads(match.group())
            domain = parsed.get("taxonomy_domain", fallback_domain)
            if domain not in ("Electrical", "Mechanical", "Plumbing"):
                domain = fallback_domain
            return {
                "taxonomy_domain":      domain,
                "taxonomy_category":    parsed.get("taxonomy_category",    "") or "",
                "taxonomy_subcategory": parsed.get("taxonomy_subcategory", "") or "",
            }
        except Exception as exc:
            logger.warning("llm_fallback_taxonomy attempt %d/3 failed: %s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(2 ** attempt)

    return {"taxonomy_domain": fallback_domain, "taxonomy_category": "", "taxonomy_subcategory": ""}
