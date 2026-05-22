"""
Query classifier: model_number / technical / descriptive.

Uses Gemini 2.5 Flash via OpenRouter exclusively.
Set OPENROUTER_API_KEY in the environment (or .env file).

Results are cached in-process so repeated calls for the same query are free.
"""

import os
import warnings

warnings.filterwarnings("ignore")

_openrouter_client = None
_classify_cache: dict = {}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """\
Classify this product search query into exactly one category.

Categories:
  model_number  - The query IS a part number, SKU, or model code. Usually short,
                  alphanumeric, may contain separators (- / .) or a brand prefix
                  separated by a space. No sentence structure.
                  Examples: "K-2084"  "LC1D18B7"  "12110"  "REGAL 111"  "BDD6R"
                            "CL10-10BM"  "ADA940T"  "RDVG5"  "30-0101"

  technical     - The query describes a product using specifications, numbers with
                  units, or technical abbreviations. Has some natural language but
                  the core meaning is spec-driven.
                  Examples: "16A single pole MCB"  "500W 240V baseboard heater"
                            "IP54 weatherproof outlet 10A"  "6-ton R22 slab coil TXV"

  descriptive   - Natural language describing a product by function, appearance, or
                  application. No model numbers or spec codes.
                  Examples: "weatherproof outdoor power outlet grey"
                            "mop sink for commercial kitchen"
                            "window exhaust fan with pull cord"

Query: "{query}"

Reply with exactly one word (model_number / technical / descriptive):"""

# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------


def _get_openrouter_client():
    """Return an OpenAI-compatible client pointed at OpenRouter."""
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file to enable the Gemini classifier."
            )
        from openai import OpenAI
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _openrouter_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_query(query: str) -> str:
    """
    Classify a query as 'model_number', 'technical', or 'descriptive'
    using Gemini 2.5 Flash via OpenRouter.

    Results are cached in-process to avoid redundant API calls.
    """
    q = query.strip()

    if q in _classify_cache:
        return _classify_cache[q]

    client = _get_openrouter_client()
    resp = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(query=q)}],
        max_tokens=8,
        temperature=0,
    )
    token = resp.choices[0].message.content.strip().lower().split()[0]

    if token not in ("model_number", "technical", "descriptive"):
        raise ValueError(
            f"Gemini returned unexpected classification '{token}' for query: {q!r}"
        )

    _classify_cache[q] = token
    return token
