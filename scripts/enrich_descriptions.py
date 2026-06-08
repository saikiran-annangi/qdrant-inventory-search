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

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from data.loaders import load_all

CACHE_PATH  = Path(__file__).parent.parent / "enrichment_cache.json"
MODEL       = "google/gemini-2.5-flash"
MAX_WORKERS = 150
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

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# System prompt — PrefPO-optimised, 366 tokens, 87% quality on curated test.
# ---------------------------------------------------------------------------
# Optimised by PrefPO (champion-challenger, 25 rounds) starting from the
# 450-token domain-aware prompt. Result: 366 tokens, 87% quality (vs 83%
# at 450 tokens) — cheaper AND better.
#
# Architecture: general instruction + correction map.
#   Layer 1: "Include field trade names in single quotes" — handles ANY
#            product type using the model's built-in MEP knowledge.
#   Layer 2: Inline map of ~20 terms the model gets wrong without help
#            (regional jargon: GPO/power point; brand-as-generic: Fernco;
#             specialist HVAC: canvas connection; AU-specific: TPS/flat cable).
#   1 example: teaches the output format (quoting style, size bridge).
#
# For Phase 2 (fine-tuning the embedding model), simplify this prompt
# significantly — the fine-tuned encoder handles vocabulary natively.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Write 2-3 sentence, lowercase, factual, no-marketing MEP product descriptions. "
    "Expand all abbreviations. Include field trade names in single quotes "
    "(e.g. BX/FLEX=\'dry connector\'/\'flex connector\'/\'not liquidtight\', "
    "P-trap=\'bottle trap\'/\'u-bend\'/\'drain trap\', "
    "no-hub coupling=\'Fernco\'/\'mission coupling\'/\'rubber band coupler\', "
    "ball valve=\'isolation valve\'/\'stopcock\'/\'quarter-turn\', "
    "GPO=\'power point\'/\'socket\'/\'general purpose outlet\', "
    "VAV terminal=\'VAV box\'/\'zone box\'/\'VAV terminal\', "
    "fan coil=\'FCU\'/\'chilled water FCU\', EMT=\'thinwall\', "
    "RCBO=\'safety switch\'/\'GFCI\', "
    "flexible duct connector=\'canvas connection\'/\'vibration isolator\'/\'flexible connector\', "
    "Y-strainer=\'line strainer\'/\'basket strainer\'/\'y-strainer\', "
    "conduit saddle=\'conduit strap\'/\'one-hole strap\'/\'pipe strap\', "
    "liquidtight connector=\'sealtight\'/\'watertight\', "
    "lug=\'cable lug\'/\'crimp lug\', "
    "push fitting=\'push fit\'/\'push to connect\'/\'sharkbite\', "
    "angle stop=\'under-sink valve\'/\'supply stop\', "
    "MCB 3P=\'triple pole MCB\'/\'three pole MCB\', "
    "AHU=\'rooftop unit\'/\'RTU\', cable=\'flat cable\'/\'TPS\', "
    "switch=\'rocker switch\'/\'light switch\'). "
    "Note compatible wire gauge (AWG/MCM/kcmil), pipe size (NPS/CTS), or airflow (CFM). "
    "Output description only.\n\n"
    "label | manufacturer | model | category\n"
    "CLAMP 1.69-1.98IN CABLE 1-1/2IN EMT | Cooper B-Line | B2000 | Conduit Fittings\n"
    "\u2192 1-1/2-inch electrical metallic tubing (EMT) clamp (\'one-hole strap\'/\'p-clamp\') "
    "for securing 1.69-1.98-inch diameter cables (e.g. 250 kcmil, 4/0 AWG)."
)

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
    if len(text.split()) < 4 or len(text) > 500:
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
            max_tokens=120,
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
