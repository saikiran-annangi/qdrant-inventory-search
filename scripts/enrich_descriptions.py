"""
Generate rich natural-language extended_description for products with terse
or heavily-abbreviated descriptions.

Uses Gemini 2.5 Flash via OpenRouter (same key as the query classifier).
Results are saved to enrichment_cache.json as {point_id: extended_description}.
The ingest pipeline (step 2) will read this cache to populate the field.

Usage:
    python scripts/enrich_descriptions.py            # enrich all eligible products
    python scripts/enrich_descriptions.py --dry-run  # preview counts, no API calls
    python scripts/enrich_descriptions.py --sample 5 # enrich 5 products and print output
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable the Gemini client inside load_all() -- we generate descriptions here.
# Must be set before importing data.loader (which reads env at call time).
os.environ["GEMINI_API_KEY"] = ""

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()  # loads OPENROUTER_API_KEY; GEMINI_API_KEY already set to "" above

from openai import OpenAI
from data.loader import load_all

CACHE_PATH  = Path(__file__).parent.parent / "enrichment_cache.json"
MODEL       = "google/gemini-2.5-flash"
MAX_WORKERS = 20
SAVE_EVERY  = 100


# ---------------------------------------------------------------------------
# OpenRouter client (same pattern as models/classifier.py)
# ---------------------------------------------------------------------------

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set — check your .env file")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


# ---------------------------------------------------------------------------
# Which products need enrichment?
# ---------------------------------------------------------------------------

def _needs_enrichment(desc: str) -> bool:
    """
    True for descriptions that are too terse or abbreviated for semantic search.

    Catches two patterns:
      - Very short (< 5 meaningful words): "ELBOW 90 DEGREE 10 IN-28GA"
      - Short + mostly-uppercase abbreviations (5-15 words):
        "POWER POINT 10A SGL WEATHERSHIELD HRZ GREY IP54 FLUSH MNT"
    """
    if not desc or str(desc).strip() in ("", "nan"):
        return True
    words = [w for w in str(desc).split() if len(w) > 1]
    if len(words) < 5:
        return True
    if len(words) <= 15:
        upper_ratio = sum(1 for w in words if w.isupper()) / len(words)
        if upper_ratio > 0.6:
            return True
    return False


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

# System prompt sent once per session (cached by the API — not re-charged per call).
# User message carries only the compact product line, keeping per-call tokens minimal.
_SYSTEM_PROMPT = (
    "You write product descriptions for an industrial/electrical/plumbing distributor catalog. "
    "Input format: abbreviated label | manufacturer | model | category. "
    "Output: 1-2 sentences in plain English. "
    "Rules: expand all abbreviations into full words, use lowercase, factual only — "
    "no marketing language, no bullet points. "
    "Output ONLY the description — no preamble, no quotes, no 'Here is...'.\n\n"
    "Examples:\n"
    "RCBO 1P+N 40A 6KA C 30MA | Schneider Electric | A9D31640 | Circuit Breakers\n"
    "→ single-pole plus neutral residual current circuit breaker with overcurrent protection, "
    "rated 40 amp, 6kA breaking capacity, C-curve, 30mA sensitivity.\n\n"
    "EXHAUST FAN 190MM P/CORD LOUV | Whisper | EC190WP | Ventilation\n"
    "→ window-mounted exhaust fan with a 190mm blade, pull cord operation, and a louvred grille."
)

# Per-field input length caps — prevents oversized prompts for edge-case long descriptions.
_MAX_DESC  = 150
_MAX_MFR   = 40
_MAX_MODEL = 30
_MAX_CAT   = 40


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s[:n] if len(s) > n else s


def _clean_output(text: str) -> str | None:
    """Strip LLM preambles and validate output. Returns None if unusable."""
    text = text.strip().strip("\"'")
    for prefix in ("here is ", "here's ", "this is ", "description: ", "→ ", "- "):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].lstrip()
    text = text.strip()
    if len(text.split()) < 4 or len(text) > 300:
        return None
    return text


def _generate(client: OpenAI, record: dict) -> tuple[str, str]:
    """Return (point_id, extended_description). Raises on API error."""
    user_msg = (
        f"{_truncate(record['description'], _MAX_DESC)} | "
        f"{_truncate(record['manufacturer_name'], _MAX_MFR)} | "
        f"{_truncate(record['model_number'], _MAX_MODEL)} | "
        f"{_truncate(record['product_category'], _MAX_CAT)}"
    )
    for _ in range(2):  # retry once on bad output
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=80,
            temperature=0,
            timeout=30,
        )
        result = _clean_output(resp.choices[0].message.content)
        if result:
            return record["id"], result

    # Fallback: cleaned original description or manufacturer + model
    fallback = _truncate(record["description"], 200) or (
        f"{record['manufacturer_name']} {record['model_number']}".strip()
    )
    return record["id"], fallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true", help="count only, no API calls")
    parser.add_argument("--sample",   type=int, default=0, help="enrich N products and print output")
    args = parser.parse_args()

    # Load existing cache
    cache: dict = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    print(f"Cache: {len(cache)} existing entries\n")

    # Load all products (no Gemini calls — disabled above)
    print("Loading all products from all sources...")
    records = load_all(verbose=True)

    # Classify each product
    to_enrich    = [r for r in records if _needs_enrichment(r["description"]) and r["id"] not in cache]
    cached_count = sum(1 for r in records if r["id"] in cache)
    rich_count   = len(records) - len(to_enrich) - cached_count

    print(f"\n{'─'*50}")
    print(f"  Total products  : {len(records)}")
    print(f"  Already in cache: {cached_count}")
    print(f"  Rich description: {rich_count}  (no enrichment needed)")
    print(f"  Needs enrichment: {len(to_enrich)}")
    print(f"{'─'*50}\n")

    if args.dry_run:
        print("Dry-run complete — no API calls made.")
        return

    if not to_enrich:
        print("Nothing to enrich.")
        return

    if args.sample:
        to_enrich = to_enrich[:args.sample]
        print(f"Sample mode: enriching {len(to_enrich)} products\n")

    client = _get_client()
    done   = 0
    errors = 0
    start  = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_generate, client, r): r for r in to_enrich}

        for future in as_completed(futures):
            record = futures[future]
            try:
                point_id, ext_desc = future.result()
                cache[point_id] = ext_desc
                done += 1

                if args.sample:
                    print(f"  [{done}] {record['model_number']}")
                    print(f"        original : {record['description']}")
                    print(f"        enriched : {ext_desc}\n")

            except Exception as e:
                errors += 1
                print(f"  [ERR] {record.get('model_number', '?')}: {e}")

            # Periodic checkpoint save
            if not args.sample and done % SAVE_EVERY == 0 and done > 0:
                with open(CACHE_PATH, "w") as f:
                    json.dump(cache, f)
                elapsed = time.time() - start
                rate    = done / elapsed
                eta     = (len(to_enrich) - done) / rate if rate > 0 else 0
                print(
                    f"  [{done:5d}/{len(to_enrich)}  "
                    f"{done/len(to_enrich)*100:5.1f}%]  "
                    f"errors={errors}  "
                    f"rate={rate:.1f}/s  "
                    f"ETA={eta/60:.1f}min"
                )

    # Final save
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"  Enriched   : {done}")
    print(f"  Errors     : {errors}")
    print(f"  Cache size : {len(cache)} total entries")
    print(f"  Time       : {elapsed:.0f}s")
    print(f"  Saved to   : {CACHE_PATH}")
    print(f"{'='*50}")
    print("\nNext step: run  python scripts/ingest.py  to re-ingest with enriched descriptions.")


if __name__ == "__main__":
    main()
