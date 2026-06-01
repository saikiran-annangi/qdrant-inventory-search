"""
Product attribute extractor.

Calls Gemini 2.5 Flash via OpenRouter to extract structured attributes
(explicit + inferred) from product fields. Results are meant to be cached
in attributes_cache.json — never call this on every ingest.

Public API:
    extract_product_attributes(model_number, description, extended_description,
                               manufacturer, product_category, source) -> dict
"""

import json
import logging
import os
import re
import time
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

_openrouter_client = None

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT = """\
Extract product attributes. Respond with JSON only.

Product:
Model: {model_number}
Description: {description}
{extended_line}Manufacturer: {manufacturer}
Category: {product_category}

Detect domain (Electrical/Plumbing/Mechanical/Unknown), then extract only \
attributes you are confident about, split into explicit (stated) and \
inferred (deduced from model number or manufacturer knowledge).

Electrical: voltage, amperage, poles, phase, product_type, brand, mounting_type, \
enclosure_type, wire_gauge, conduit_size, interrupting_rating, application
Plumbing: pipe_size, material, connection_type, product_type, pressure_rating, \
end_type, application, temperature_rating
Mechanical: duct_size, product_type, material, airflow_rating, application, operation

{{"domain":"...","explicit":{{...}},"inferred":{{...}}}}"""

_EMPTY_RESULT = {"domain": "Unknown", "explicit": {}, "inferred": {}}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client():
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set.")
        from openai import OpenAI
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _openrouter_client


def _call_llm(prompt: str) -> str:
    """Call Gemini 2.5 Flash with 3 retries. Returns raw response string."""
    client = _get_client()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning("LLM call failed (attempt %d/3): %s. Retrying in %ds.", attempt + 1, exc, wait)
            if attempt < 2:
                time.sleep(wait)
    logger.error("LLM call failed after 3 attempts.")
    return ""


def _parse_json(raw: str) -> dict:
    """Extract JSON object from LLM response. Returns empty result on failure."""
    # Find the outermost {...} block regardless of surrounding code fences or text
    match = re.search(r'\{.*\}', raw.strip(), re.DOTALL)
    if not match:
        logger.warning("No JSON object found in LLM response: %r", raw[:200])
        return _EMPTY_RESULT
    try:
        parsed = json.loads(match.group())
        domain = parsed.get("domain", "Unknown")
        if domain not in ("Electrical", "Plumbing", "Mechanical", "Unknown"):
            domain = "Unknown"
        return {
            "domain":   domain,
            "explicit": parsed.get("explicit", {}) or {},
            "inferred": parsed.get("inferred", {}) or {},
        }
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse LLM JSON response: %r", raw[:200])
        return _EMPTY_RESULT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_product_attributes(
    model_number:         str,
    description:          str,
    extended_description: str,
    manufacturer:         str,
    product_category:     str,
    source:               str,
) -> dict:
    """
    Extract structured attributes from a product using Gemini 2.5 Flash.

    Returns:
        {
            "domain":   "Electrical" | "Plumbing" | "Mechanical" | "Unknown",
            "explicit": { "amperage": "200A", ... },
            "inferred": { "voltage": "120/240V", ... },
        }

    Never raises — returns _EMPTY_RESULT on any failure.
    """
    extended_line = (
        f"Extended description: {(extended_description or '')[:400]}\n"
        if extended_description
        else ""
    )

    prompt = _PROMPT.format(
        model_number=     (model_number     or "")[:80],
        description=      (description      or "")[:300],
        extended_line=    extended_line,
        manufacturer=     (manufacturer     or "")[:80],
        product_category= (product_category or "")[:80],
    )

    try:
        raw = _call_llm(prompt)
        return _parse_json(raw)
    except Exception as exc:
        logger.error("extract_product_attributes failed for %r: %s", model_number, exc)
        return _EMPTY_RESULT
