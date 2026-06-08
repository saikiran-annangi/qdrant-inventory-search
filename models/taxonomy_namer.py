"""Name a novel product's taxonomy node — consistently with the existing tree.

Used by the ingest classifier when a product matches no existing node well
enough. The whole point: the LLM is SHOWN the current vocabulary for the domain
and told to PREFER an existing "Category > Subcategory", or extend it by adding a
subcategory under an existing category, and only mint a new category as a last
resort — always following the established naming style. That keeps auto-created
names consistent ("LED Downlights", not "recessed downlight fixtures (LED)") so
the dedup in TaxonomyStore actually collapses near-duplicates.

Graceful degradation: if OPENROUTER_API_KEY is absent the caller falls back to
an ERP-category or heuristic name (see build_taxonomy_from_descriptions.py).
"""

from __future__ import annotations

import json
import os
import re

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        from openai import OpenAI
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _client


def is_enabled() -> bool:
    return _get_client() is not None


def _format_vocabulary(labels_by_category: dict, max_subs: int = 12) -> str:
    """Render existing 'Category: sub, sub, ...' lines for the prompt."""
    lines = []
    for category in sorted(labels_by_category):
        subs = sorted(labels_by_category[category])
        shown = ", ".join(subs[:max_subs])
        if len(subs) > max_subs:
            shown += ", …"
        lines.append(f"- {category}: {shown}")
    return "\n".join(lines)


_PROMPT = """\
You maintain a controlled product taxonomy for an electrical/MEP distributor.
A product needs a (category, subcategory) within the domain "{domain}".

EXISTING categories and subcategories in this domain:
{vocabulary}

Product:
{product}

Rules, in priority order:
1. If an existing "category / subcategory" above fits, return it EXACTLY as written.
2. Otherwise, if an existing CATEGORY fits, keep that category and add a new
   subcategory named in the SAME style (Title Case, concise, "X & Y" form,
   singular product-type wording like the examples above).
3. Only invent a NEW category if none above is appropriate — name it in the same
   style as the existing categories.
Never output punctuation-heavy or verbose names. Match the existing style closely.

Return JSON only: {{"category": "...", "subcategory": "..."}}"""


def propose_node(product_text: str, domain: str, labels_by_category: dict,
                 timeout: float = 8.0):
    """Return (category, subcategory) for a novel product, or None on failure.

    labels_by_category: {category: [subcategory, ...]} — the domain's current
    vocabulary, shown to the model so it stays consistent.
    """
    client = _get_client()
    if client is None:
        return None
    prompt = _PROMPT.format(
        domain=domain,
        vocabulary=_format_vocabulary(labels_by_category) or "(none yet)",
        product=product_text[:400],
    )
    try:
        resp = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
            timeout=timeout,
        )
        raw = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        parsed = json.loads(m.group())
        category = " ".join(str(parsed.get("category", "")).split())
        subcategory = " ".join(str(parsed.get("subcategory", "")).split())
        if category and subcategory:
            return category, subcategory
    except Exception:
        return None
    return None
