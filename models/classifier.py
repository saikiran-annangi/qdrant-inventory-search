"""
Three-tier query classifier: model_number / technical / descriptive.

Priority order:
  1. Trained logistic regression (query_classifier.joblib) -- fast, local, ~85-90% accuracy
  2. OpenRouter Gemini 2.5 Flash (OPENROUTER_API_KEY env var) -- highest accuracy, ~99%
  3. Regex fallback (always available, no dependencies) -- ~55% accuracy

Results are cached in-process so repeated calls are free.
"""

import os
import re
import warnings

warnings.filterwarnings("ignore")

from config import CLASSIFIER_PATH

_lr_classifier = None
_openrouter_client = None
_classify_cache: dict = {}

# ---------------------------------------------------------------------------
# Prompt used for the OpenRouter LLM classifier
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
# Regex patterns (tier 3 fallback)
# ---------------------------------------------------------------------------

_MODEL_NUMBER_RE = re.compile(
    r"^[A-Z0-9]{2,}[-./][A-Z0-9]|"   # alphanum + separator
    r"^[A-Z]{1,4}[0-9]{3,}|"          # letter prefix + 3+ digits
    r"^[0-9]{4,}[A-Z]",               # digit-heavy with trailing letter
    re.IGNORECASE,
)

_TECH_KEYWORDS = re.compile(
    r"\b(\d+\.?\d*\s*(A|KA|V|W|MA|HP|KW|POLE|P|AMP|VOLT|OHM))\b|"
    r"\b(MCB|RCD|GFCI|MCCB|VFD|UPS|DOL|Y[-/]D|NPN|PNP)\b|"
    r"\b(SINGLE|DOUBLE|TRIPLE|3[-\s]?PHASE|1[-\s]?PHASE)\b|"
    r"\b(DIN|IP\d{2}|NEMA|CSA|UL|IEC|AS/NZS)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Singleton loaders
# ---------------------------------------------------------------------------


def _get_lr_classifier():
    """Load the trained logistic regression classifier from disk if available."""
    global _lr_classifier
    if _lr_classifier is None:
        if os.path.exists(CLASSIFIER_PATH):
            try:
                import joblib
                _lr_classifier = joblib.load(CLASSIFIER_PATH)
            except Exception:
                pass
    return _lr_classifier


def _get_openrouter_client():
    """Return an OpenAI-compatible client pointed at OpenRouter."""
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
            _openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        except Exception:
            return None
    return _openrouter_client


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------


def classify_query_regex(query: str) -> str:
    """Regex-based fallback classifier (~55% accuracy on the eval set)."""
    q = query.strip()
    if " " not in q and _MODEL_NUMBER_RE.search(q):
        return "model_number"
    words = q.split()
    if len(words) <= 3 and _MODEL_NUMBER_RE.search(q):
        return "model_number"
    if _TECH_KEYWORDS.search(q):
        return "technical"
    return "descriptive"


def classify_query(query: str) -> str:
    """
    Classify a query as 'model_number', 'technical', or 'descriptive'.

    Tries each tier in order and returns the first successful result.
    Caches results in-process to avoid redundant API calls.
    """
    q = query.strip()

    if q in _classify_cache:
        return _classify_cache[q]

    result = None

    # Tier 1: trained LR model (fastest, no network required)
    lr = _get_lr_classifier()
    if lr is not None:
        try:
            result = lr.predict([q])[0]
        except Exception:
            result = None

    # Tier 2: OpenRouter Gemini 2.5 Flash (most accurate, requires API key)
    if result is None:
        client = _get_openrouter_client()
        if client is not None:
            try:
                resp = client.chat.completions.create(
                    model="google/gemini-2.5-flash",
                    messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(query=q)}],
                    max_tokens=8,
                    temperature=0,
                )
                token = resp.choices[0].message.content.strip().lower().split()[0]
                if token in ("model_number", "technical", "descriptive"):
                    result = token
            except Exception:
                result = None

    # Tier 3: regex fallback (always available)
    if result is None:
        result = classify_query_regex(q)

    _classify_cache[q] = result
    return result
