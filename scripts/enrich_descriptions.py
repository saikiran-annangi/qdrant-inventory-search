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
# Domain-aware system prompts
# ---------------------------------------------------------------------------
# One prompt per MEP domain, each with domain-specific few-shot examples.
# The LLM follows examples far more reliably than instructions alone — so an
# electrical product shown electrical examples generates electrical trade terms,
# a plumbing product shown plumbing examples generates plumbing trade terms, etc.
# This is what makes enrichment work for ALL MEP domains, not just the two
# queries that originally failed.
# ---------------------------------------------------------------------------

_PROMPT_BASE = (
    "You write product descriptions for an MEP distributor catalog. "
    "Input format: abbreviated label | manufacturer | model | category. "
    "Output: 2-3 sentences in plain English covering:\n"
    "  1. What the product is and what it does — expand all abbreviations.\n"
    "  2. Common trade names or synonyms that {trade} use for it in the field.\n"
    "  3. Compatible {compat} where applicable.\n"
    "Rules: lowercase, factual only — no marketing language, no bullet points. "
    "Output ONLY the description — no preamble, no quotes.\n\n"
    "Examples:\n{examples}"
)

_ELEC_EXAMPLES = (
    "CONNECTOR BX/FLEX 2IN 2-SCREW ALUMINUM | ABB | CI2116 | Conduit Fittings\n"
    "→ two-screw aluminum connector for attaching 2-inch flexible metal conduit "
    "(BX or armored cable) to an electrical box knockout; field trade name is "
    "'dry connector' because it is not liquidtight; also called flex connector or BX connector.\n\n"
    "CLAMP 1.69-1.98IN CABLE 1-1/2IN EMT-RIG | ABB | CPC150 | Conduit Supports\n"
    "→ p-type cable clamp (P-clamp) securing large power cable to 1-1/2-inch EMT "
    "or rigid conduit; jaw 1.69–1.98 inches fits 250 MCM through 350 MCM (kcmil) "
    "including 4/0 AWG.\n\n"
    "RCBO 1P+N 40A 6KA C 30MA | Schneider Electric | A9D31640 | Circuit Breakers\n"
    "→ single-pole plus neutral residual current circuit breaker with overcurrent "
    "protection, 40 amp, 6kA, C-curve, 30mA; also called GFCI breaker or safety switch.\n\n"
    "LUG AL 1-HOLE 250MCM | Burndy | YA250 | Lugs & Links\n"
    "→ single-hole aluminum compression lug (crimp lug, cable lug) for terminating "
    "250 MCM (kcmil) aluminum conductor to a bus bar or equipment terminal."
)

_PLUMB_EXAMPLES = (
    "COUPLING FLEX 4IN NO-HUB | Fernco | P1056-44 | Pipe Fittings\n"
    "→ flexible rubber no-hub coupling (Fernco coupling) joining two 4-inch "
    "cast-iron or PVC drain pipes; used by plumbers where rigid fittings cannot "
    "be rotated; also called a fernco, mission coupling, or rubber band coupler.\n\n"
    "TRAP 1-1/2IN P ABS | Genova | 73715 | Drain Fittings\n"
    "→ 1-1/2-inch ABS P-trap (sink trap, drain trap) that holds a water seal "
    "to block sewer gas; fits standard lavatory and kitchen sink waste outlets; "
    "also called a U-bend or bottle trap.\n\n"
    "VALVE BALL 3/4IN CPVC | Spears | 2522-007 | Valves\n"
    "→ 3/4-inch CPVC full-port ball valve (isolation valve, shutoff valve) for "
    "hot and cold domestic water; quarter-turn operation; compatible with 3/4-inch "
    "NPS (nominal pipe size) CPVC solvent-weld fittings.\n\n"
    "FITTING PUSH 1/2IN ELBOW | SharkBite | U256LFA | Push Fittings\n"
    "→ 1/2-inch push-to-connect elbow (SharkBite fitting, push-fit fitting) "
    "joining copper, PEX, or CPVC pipe without soldering or crimping; "
    "works on 1/2-inch CTS (copper tube size) pipe."
)

_MECH_EXAMPLES = (
    "VAV BOX SDZ 12IN 0-2000CFM | Trane | VCVL12 | HVAC Terminal Units\n"
    "→ single-duct variable air volume terminal unit (VAV box, VAV terminal) "
    "controlling airflow from 0 to 2000 CFM through a 12-inch round duct inlet; "
    "used by mechanical contractors to zone heating and cooling in commercial buildings.\n\n"
    "FAN COIL 2-PIPE 800CFM 240V | Daikin | FWD08ATN | Fan Coil Units\n"
    "→ horizontal ceiling-concealed fan coil unit (FCU) for 2-pipe chilled-water "
    "or hot-water HVAC systems; 800 CFM capacity; also called a chilled beam "
    "terminal or hydronic fan coil; connects to 240V supply.\n\n"
    "FLEX DUCT CONN 14IN FABRIC | Vibro-Acoustics | FDC-14 | Ductwork\n"
    "→ 14-inch fabric flexible duct connector (flex connection, duct isolation "
    "connector) isolating air-handling equipment vibration from the duct system; "
    "also called a canvas connection or vibration isolator collar.\n\n"
    "STRAINER Y-TYPE 2IN 150LB | Watts | LF777 | Hydronic Specialties\n"
    "→ 2-inch Y-type strainer (Y-strainer, line strainer) removing debris from "
    "hydronic heating and cooling pipework; 150 lb flanged ends; compatible with "
    "2-inch NPS carbon steel or copper pipe."
)

_DOMAIN_PROMPTS = {
    "Electrical": _PROMPT_BASE.format(
        trade="electricians and electrical contractors",
        compat="wire sizes (AWG, MCM/kcmil), conduit trade sizes, or voltage ratings",
        examples=_ELEC_EXAMPLES,
    ),
    "Plumbing": _PROMPT_BASE.format(
        trade="plumbers and plumbing contractors",
        compat="pipe sizes (NPS, CTS, OD), pressure ratings, or material compatibility",
        examples=_PLUMB_EXAMPLES,
    ),
    "Mechanical": _PROMPT_BASE.format(
        trade="mechanical contractors and HVAC technicians",
        compat="duct sizes, airflow (CFM), pipe sizes, or voltage/refrigerant ratings",
        examples=_MECH_EXAMPLES,
    ),
}
# Default for unknown / Tools & Site domains
_SYSTEM_PROMPT_DEFAULT = _PROMPT_BASE.format(
    trade="electrical, mechanical, and plumbing contractors",
    compat="sizes, ratings, or compatible materials",
    examples=_ELEC_EXAMPLES + "\n\n" + _PLUMB_EXAMPLES[:_PLUMB_EXAMPLES.index("\n\nVALVE")],
)

# Re-export as _SYSTEM_PROMPT for backwards compatibility with callers that
# reference it directly (e.g. tests / scripts).
_SYSTEM_PROMPT = _SYSTEM_PROMPT_DEFAULT


def _domain_from_record(record: dict) -> str:
    """Infer MEP domain from product_category + description for prompt selection."""
    text = (
        (record.get("product_category") or "") + " " +
        (record.get("description") or "")
    ).lower()
    elec = sum(1 for k in (
        "electrical","lighting","circuit","breaker","cable","conduit","wire",
        "switch","outlet","relay","contactor","transformer","fuse","led","lamp",
        "mcb","rcbo","lug","terminal","gland","earth","solar","fan","enclosure",
    ) if k in text)
    plumb = sum(1 for k in (
        "plumbing","water","pipe","valve","drain","fitting","trap","coupling",
        "toilet","sink","shower","faucet","sewer","pvc","cpvc","pex","copper",
    ) if k in text)
    mech = sum(1 for k in (
        "hvac","duct","ahu","coil","damper","vav","vrf","chiller","boiler",
        "air handling","fan coil","fcu","ventilation","refrigerant","hydronic",
    ) if k in text)
    best = max(elec, plumb, mech)
    if best == 0:
        return "Default"
    if elec == best:
        return "Electrical"
    if plumb == best:
        return "Plumbing"
    return "Mechanical"


# System prompt sent once per session (cached by the API — not re-charged per call).
# User message carries only the compact product line, keeping per-call tokens minimal.
# (Legacy variable kept for any callers that imported _SYSTEM_PROMPT directly.)
_EXHAUST_FAN_EXAMPLE = (
    "EXHAUST FAN 190MM P/CORD LOUV | Whisper | EC190WP | Ventilation\n"
    "→ window-mounted exhaust fan with a 190mm blade, pull cord operation, and a "
    "louvred grille; suitable for bathrooms and utility rooms."
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
    # Pick the domain-specific prompt so examples match the product type.
    # An electrical product gets electrical trade-term examples; a plumbing
    # product gets plumbing examples — the LLM follows examples, not just
    # instructions, so this is what makes enrichment work across all MEP domains.
    domain = _domain_from_record(record)
    system_prompt = _DOMAIN_PROMPTS.get(domain, _SYSTEM_PROMPT_DEFAULT)
    for _ in range(2):  # retry once on bad output
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
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
