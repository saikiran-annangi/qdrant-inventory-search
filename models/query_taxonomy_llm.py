"""
Query-side two-stage LLM taxonomy classifier.

Replaces the embedding-based classify_query_taxonomy() with two fast
Gemini Flash calls that always return a taxonomy result and always map
into labels that actually exist in the collection.

Stage 1: classify domain  (Electrical / Mechanical / Plumbing / Unknown)
         ~50 token prompt — 3 choices.

Stage 2: classify category/subcategory within the identified domain.
         Domain-filtered label list (~30-50 nodes instead of 500+).
         LLM picks from labels that exist in taxonomy_cache.json — never
         invents new ones.

Public API:
    classify_query_taxonomy_llm(query, query_type) -> dict
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_openrouter_client = None
_labels_by_domain: dict = {}
_labels_loaded: bool = False


def _get_client():
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _openrouter_client


def _load_labels() -> None:
    """
    Build domain → sorted list of "Category > Subcategory" strings from
    taxonomy_cache.json.  Falls back to PRODUCT_TAXONOMY from config when
    the cache file is absent (e.g. first run before Phase 1 is complete).
    Loaded once at process start; subsequent calls are no-ops.
    """
    global _labels_by_domain, _labels_loaded
    if _labels_loaded:
        return

    from config import TAXONOMY_CACHE_PATH, PRODUCT_TAXONOMY

    labels: dict = {"Electrical": set(), "Mechanical": set(), "Plumbing": set()}

    # Build set of predefined label strings for filtering
    predefined_labels: set = set()
    for domain, categories in PRODUCT_TAXONOMY.items():
        for category, subcategories in categories.items():
            for subcategory in subcategories:
                predefined_labels.add(f"{category} > {subcategory}")

    if os.path.exists(TAXONOMY_CACHE_PATH):
        with open(TAXONOMY_CACHE_PATH) as f:
            cache = json.load(f)
        for entry in cache.values():
            domain      = entry.get("taxonomy_domain",      "") or ""
            category    = entry.get("taxonomy_category",    "") or ""
            subcategory = entry.get("taxonomy_subcategory", "") or ""
            label       = f"{category} > {subcategory}"
            # Only include labels from the predefined MEP vocabulary.
            # LLM-invented labels (from non-MEP items like tools, stationery)
            # would produce a taxonomy_subcategory that no product in the collection
            # matches, making the soft boost a no-op at best or wrong at worst.
            if domain in labels and category and subcategory and label in predefined_labels:
                labels[domain].add(label)

    # Always include predefined nodes so the list is never empty
    for domain, categories in PRODUCT_TAXONOMY.items():
        if domain not in labels:
            labels[domain] = set()
        for category, subcategories in categories.items():
            for subcategory in subcategories:
                labels[domain].add(f"{category} > {subcategory}")

    _labels_by_domain = {d: sorted(v) for d, v in labels.items() if v}
    total = sum(len(v) for v in _labels_by_domain.values())
    logger.info(
        "Query taxonomy labels loaded: %d total (%s)",
        total,
        ", ".join(f"{d}: {len(v)}" for d, v in _labels_by_domain.items()),
    )
    _labels_loaded = True


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_STAGE1_PROMPT = """\
Classify this industrial product search query into one domain.
Domains: Electrical, Mechanical, Plumbing, Unknown

Query: {query}

Return JSON only: {{"domain": "..."}}"""


_STAGE2_PROMPT = """\
Classify this industrial product search query into a taxonomy node.

Available nodes for domain "{domain}":
{node_list}

Query: {query}

Pick the single best matching node from the list above.
Use the EXACT text from the list — do not invent new labels.

Return JSON only: {{"taxonomy_category": "...", "taxonomy_subcategory": "..."}}"""


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _llm_call(prompt: str, timeout: float = 5.0) -> str:
    client = _get_client()
    if client is None:
        return ""
    resp = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
        timeout=timeout,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Patterns that indicate a model-number lookup even when the classifier says otherwise.
# These queries have explicit part/stock/catalog number prefixes mixed with description.
_MODEL_NUMBER_PREFIXES = (
    "pn:", "p/n:", "p/n ", "stk no.", "stk no ", "cat#", "cat #",
    "part#", "part #", "part no.", "ref#", "ref #", "item#", "item #",
    "(model ", "model no.", "model no ", "model# ", "model #",
)


def _looks_like_model_number(query: str) -> bool:
    q = query.lower().strip()
    return any(q.startswith(p) or f" {p}" in q for p in _MODEL_NUMBER_PREFIXES)


def classify_query_taxonomy_llm(query: str, query_type: str) -> dict:
    """
    Two-stage LLM query taxonomy classifier.

    Args:
        query:      Raw query string.
        query_type: "model_number" | "technical" | "descriptive"

    Returns:
        dict with taxonomy_domain, taxonomy_category, taxonomy_subcategory,
        confidence_score (0-1). Empty dict {} when taxonomy is skipped.
    """
    if query_type == "model_number":
        return {}
    # Skip taxonomy for queries that look like model-number lookups even if
    # the classifier predicted technical/descriptive (common misclassification).
    if _looks_like_model_number(query):
        return {}

    if _get_client() is None:
        return {}

    _load_labels()

    # ------------------------------------------------------------------
    # Stage 1 — domain
    # ------------------------------------------------------------------
    try:
        raw1 = _llm_call(_STAGE1_PROMPT.format(query=query))
        match1 = re.search(r'\{.*\}', raw1.strip(), re.DOTALL)
        if not match1:
            return {}
        domain = json.loads(match1.group()).get("domain", "Unknown")
        if domain not in ("Electrical", "Mechanical", "Plumbing"):
            return {}
    except Exception as exc:
        logger.warning("Taxonomy Stage 1 failed for %r: %s", query, exc)
        return {}

    # ------------------------------------------------------------------
    # Stage 2 — category / subcategory within domain
    # ------------------------------------------------------------------
    domain_labels = _labels_by_domain.get(domain, [])
    if not domain_labels:
        return {"taxonomy_domain": domain, "taxonomy_category": "", "taxonomy_subcategory": ""}

    node_list = "\n".join(f"  {label}" for label in domain_labels)

    try:
        raw2 = _llm_call(_STAGE2_PROMPT.format(
            domain=domain,
            node_list=node_list,
            query=query,
        ))
        match2 = re.search(r'\{.*\}', raw2.strip(), re.DOTALL)
        if not match2:
            return {"taxonomy_domain": domain, "taxonomy_category": "", "taxonomy_subcategory": ""}
        parsed = json.loads(match2.group())
        return {
            "taxonomy_domain":      domain,
            "taxonomy_category":    parsed.get("taxonomy_category",    "") or "",
            "taxonomy_subcategory": parsed.get("taxonomy_subcategory", "") or "",
        }
    except Exception as exc:
        logger.warning("Taxonomy Stage 2 failed for %r: %s", query, exc)
        return {"taxonomy_domain": domain, "taxonomy_category": "", "taxonomy_subcategory": ""}
