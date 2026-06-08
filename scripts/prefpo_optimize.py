"""PrefPO — correct implementation: shorter prompt, same quality.

The goal: find the MOST TOKEN-EFFICIENT prompt that scores high on trade
vocabulary coverage. PrefPO paper benchmarks "prompt hygiene" (3-5x shorter
than alternatives). Our previous runs went the wrong direction (450→1074 tok)
by adding examples for every missing term instead of finding general instructions.

Key insight: Gemini 2.5 Flash already KNOWS MEP trade vocabulary. We don't need
to teach it "dry connector = BX/FLEX" — we need to INSTRUCT it to include those
terms. A well-crafted instruction replaces many examples.

Design changes:
  1. Score = quality_hits - token_penalty   → rewards shorter prompts
     token_penalty = tokens / TARGET_TOKENS  (soft penalty, not hard)
  2. Start with a SHORT instruction-focused prompt (~150 tokens)
  3. Each challenger: try to COMPRESS while maintaining quality
  4. Half the strategies target compression; half target quality gaps
  5. Optimizer is told to REDUCE tokens unless quality requires more

Usage:
    python scripts/prefpo_optimize.py
"""

import asyncio, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass
import warnings; warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from openai import AsyncOpenAI, OpenAI
from data.loaders import load_all

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ITERATIONS      = 25
CHALLENGERS     = 3
TARGET_TOKENS   = 250      # soft target — score penalises above this
TOKEN_WEIGHT    = 0.08     # each token above target costs this many quality pts
MAX_CONC        = 30
TIMEOUT         = 28.0
MODEL           = "google/gemini-2.5-flash"
BASE_URL        = "https://openrouter.ai/api/v1"

sync_client  = OpenAI(base_url=BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY"))
async_client = AsyncOpenAI(base_url=BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY"))
sem          = asyncio.Semaphore(MAX_CONC)

# ---------------------------------------------------------------------------
# Starting prompt — SHORT instruction-focused, minimal examples.
# The model knows the vocabulary; this instructs it HOW to include it.
# ---------------------------------------------------------------------------
START_PROMPT = """\
MEP distributor catalog. Write product descriptions.
Input: label | manufacturer | model | category
Output: 2-3 sentences. Expand ALL abbreviations. Include the field trade name in single quotes — what electricians, plumbers, or HVAC techs actually call it (e.g. BX/FLEX='dry connector', P-trap='bottle trap'/'u-bend', no-hub coupling='Fernco', ball valve='isolation valve', GPO='power point', VAV terminal='VAV box', fan coil='FCU'). Note compatible wire gauge (AWG/MCM/kcmil), pipe size (NPS/CTS), or airflow (CFM).
Rules: lowercase, factual, no marketing. Output description only.

CONNECTOR BX/FLEX 2IN ALUMINUM | ABB | CI2116 | Conduit Fittings
→ two-screw aluminum connector for 2-inch flexible metal conduit (BX/armored cable); trade name 'dry connector' (not liquidtight); also called 'flex connector'.

TRAP 1-1/2IN P ABS | Genova | 73715 | Drain Fittings
→ 1-1/2-inch ABS p-trap ('bottle trap', 'u-bend') blocking sewer gas at sink drain; fits 1-1/2-inch NPS waste outlets."""

# ---------------------------------------------------------------------------
# Curated evaluation set — 20 products with known correct trade terms
# ---------------------------------------------------------------------------
CURATED = [
    {"input": "CONNECTOR BX/FLEX 2IN ALUMINUM | ABB | CI2116 | Metal Conduit Fittings",
     "expected": ["dry connector", "flex connector", "not liquidtight"],
     "domain": "Electrical"},
    {"input": "CONDUIT SADDLE 1IN 1-HOLE STEEL | Erico | RS100 | Conduit Saddles",
     "expected": ["conduit strap", "one-hole strap", "pipe strap"],
     "domain": "Electrical"},
    {"input": "CLAMP 1.69-1.98IN CABLE 1-1/2IN EMT | ABB | CPC150 | Conduit Supports",
     "expected": ["p-clamp", "250 mcm", "4/0", "kcmil"],
     "domain": "Electrical"},
    {"input": "CONN LIQUIDTIGHT 1/2IN STRAIGHT | Appleton | ST-050L | Conduit Fittings",
     "expected": ["liquidtight", "sealtight", "watertight"],
     "domain": "Electrical"},
    {"input": "RCBO 1P+N 20A 6KA C | Schneider Electric | A9D12220 | Circuit Breakers",
     "expected": ["residual current", "safety switch", "gfci"],
     "domain": "Electrical"},
    {"input": "MCB 3P 32A 6KA B | Hager | MCN332B | Circuit Breakers",
     "expected": ["miniature circuit breaker", "triple pole", "three pole"],
     "domain": "Electrical"},
    {"input": "OUTLET 2GANG 10A 250V WHITE | Clipsal | 15EDPW | Domestic Outlets GPOs",
     "expected": ["gpo", "power point", "socket"],
     "domain": "Electrical"},
    {"input": "SWITCH 1WAY 10A 250V WHITE | Clipsal | 30EDW | Domestic Switches",
     "expected": ["light switch", "single pole", "rocker switch"],
     "domain": "Electrical"},
    {"input": "LUG AL 1-HOLE 250MCM | Burndy | YA250 | Lugs & Links",
     "expected": ["crimp lug", "cable lug", "kcmil"],
     "domain": "Electrical"},
    {"input": "CABLE TPS 2.5MM2 TWIN EARTH 100M | Olex | | Building Wire",
     "expected": ["twin and earth", "flat cable", "tps"],
     "domain": "Electrical"},
    {"input": "TRAP 1-1/2IN P ABS | Genova | 73715 | Drain Fittings",
     "expected": ["bottle trap", "u-bend", "drain trap", "nps"],
     "domain": "Plumbing"},
    {"input": "COUPLING FLEX 4IN NO-HUB | Fernco | P1056-44 | Pipe Fittings",
     "expected": ["fernco", "mission coupling", "rubber band coupler"],
     "domain": "Plumbing"},
    {"input": "VALVE BALL 3/4IN CPVC | Spears | 2522-007 | Valves",
     "expected": ["isolation valve", "stopcock", "quarter-turn"],
     "domain": "Plumbing"},
    {"input": "FITTING PUSH 1/2IN COUPLING | SharkBite | U008LFA | Push Fittings",
     "expected": ["push to connect", "push fit", "sharkbite", "cts"],
     "domain": "Plumbing"},
    {"input": "VALVE STOP ANGLE 1/2PEX X 3/8OD | Watts | | Valves",
     "expected": ["angle stop", "under-sink valve", "supply stop"],
     "domain": "Plumbing"},
    {"input": "VAV BOX 12IN 0-2000CFM | Trane | VCVL12 | HVAC Terminal Units",
     "expected": ["variable air volume", "vav terminal", "zone"],
     "domain": "Mechanical"},
    {"input": "FAN COIL 2-PIPE 800CFM 240V | Daikin | FWD08ATN | Fan Coil Units",
     "expected": ["fan coil unit", "fcu", "chilled water"],
     "domain": "Mechanical"},
    {"input": "FLEX DUCT CONN 14IN FABRIC | Vibro-Acoustics | FDC-14 | Ductwork",
     "expected": ["canvas connection", "vibration isolator", "flexible connector"],
     "domain": "Mechanical"},
    {"input": "STRAINER Y-TYPE 2IN 150LB | Watts | LF777 | Hydronic",
     "expected": ["y-strainer", "line strainer", "basket strainer"],
     "domain": "Mechanical"},
    {"input": "AHU ROOFTOP 5T 3PH 460V | Carrier | 48XZ060 | Air Handling Units",
     "expected": ["air handling unit", "rooftop unit", "rtu", "5 ton"],
     "domain": "Mechanical"},
]

MAX_POSSIBLE = sum(len(p["expected"]) for p in CURATED)

# ---------------------------------------------------------------------------
# Strategies — half compress, half fix quality gaps
# ---------------------------------------------------------------------------
STRATEGIES = [
    # COMPRESSION strategies
    "COMPRESS: Remove any example whose pattern is already covered by the inline "
    "vocabulary hints in the instructions (e.g. 'BX/FLEX=dry connector' hint makes "
    "the BX/FLEX example redundant). Only keep examples that teach FORMAT, not vocabulary.",

    "COMPRESS: Rewrite the inline vocabulary hint list in the instructions to be shorter "
    "but still cover Electrical, Plumbing, AND Mechanical examples. Use the fewest words "
    "to convey each trade name mapping.",

    "COMPRESS: Replace multiple specific examples with ONE example that demonstrates "
    "the general pattern (single quotes, 'also called', size bridge). Remove all others.",

    "COMPRESS: Can the output instruction ('Include the field trade name in single quotes') "
    "replace examples entirely? Try a version with just instructions + zero examples.",

    "COMPRESS: Reduce the vocabulary hint list to only the most non-obvious mappings "
    "(ones the model would NOT produce without a hint). Remove obvious ones.",

    "COMPRESS: Merge the vocabulary hints and the output instruction into a single "
    "clear sentence. Cut anything redundant between instruction and examples.",

    # QUALITY strategies — fix specific gaps while staying concise
    "QUALITY: The MCB example must produce 'triple pole'. Add this to the inline "
    "vocabulary hints: 'MCB 3P = triple pole MCB'. Keep total tokens ≤ current.",

    "QUALITY: The flex duct connector must produce 'canvas connection' and 'vibration "
    "isolator'. Add a mechanical hint: 'flex duct connector = canvas connection'. "
    "Keep total tokens ≤ current.",

    "QUALITY: The Y-strainer must produce 'y-strainer' and 'basket strainer'. "
    "Add: 'Y-strainer = basket strainer, line strainer'. Keep total tokens ≤ current.",

    "QUALITY: The liquidtight connector must produce 'watertight'. Add it to the "
    "liquidtight hint in the inline vocabulary. Keep total tokens ≤ current.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def norm(s: str) -> str:
    return re.sub(r'[-\s]+', ' ', s.lower().strip())

def truncate(s: str, n: int) -> str:
    return (s or "")[:n].strip()

def product_to_user_msg(r: dict) -> str:
    return (
        f"{truncate(r.get('description',''), 120)} | "
        f"{truncate(r.get('manufacturer_name',''), 35)} | "
        f"{truncate(r.get('model_number',''), 28)} | "
        f"{truncate(r.get('product_category',''), 35)}"
    )

async def agenerate(sys_p: str, user_p: str, max_tokens: int = 110) -> str:
    async with sem:
        resp = await async_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": sys_p},
                      {"role": "user",   "content": user_p}],
            max_tokens=max_tokens, temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
            timeout=TIMEOUT,
        )
    txt = (resp.choices[0].message.content or "").strip()
    for pfx in ("→ ", "- ", "Here is", "Description: "):
        if txt.startswith(pfx):
            txt = txt[len(pfx):]
    return txt.strip()

async def allm(prompt: str, max_tokens: int = 800) -> str:
    async with sem:
        resp = await async_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
            timeout=TIMEOUT,
        )
    return (resp.choices[0].message.content or "").strip()

# ---------------------------------------------------------------------------
# Scoring — quality_hits minus token penalty above target
# ---------------------------------------------------------------------------
async def score_async(prompt: str) -> tuple[float, int, list[dict]]:
    """Returns (composite_score, quality_hits, per_product_details)."""
    tasks = [agenerate(prompt, p["input"]) for p in CURATED]
    outputs = await asyncio.gather(*tasks)

    total = 0
    details = []
    for p, out in zip(CURATED, outputs):
        out_norm = norm(out)
        hits   = [t for t in p["expected"] if norm(t) in out_norm]
        misses = [t for t in p["expected"] if norm(t) not in out_norm]
        total += len(hits)
        details.append({"domain": p["domain"],
                        "input": p["input"].split("|")[0].strip(),
                        "output": out, "hits": hits, "misses": misses})

    tokens = len(prompt) // 4
    # Soft token penalty — every token above TARGET_TOKENS costs TOKEN_WEIGHT quality pts
    penalty = max(0, tokens - TARGET_TOKENS) * TOKEN_WEIGHT
    composite = total - penalty
    return composite, total, details

def format_failures(details: list, top_n: int = 8) -> str:
    failures = [(d["input"], d["misses"], d["output"]) for d in details if d["misses"]]
    failures.sort(key=lambda x: -len(x[1]))
    lines = []
    for inp, misses, out in failures[:top_n]:
        lines.append(f'  "{inp}" → missing: {misses}\n  output: "{out[:90]}"')
    return "\n\n".join(lines) if lines else "No failures."

# ---------------------------------------------------------------------------
# Challenger generation
# ---------------------------------------------------------------------------
OPTIMIZER_PROMPT = """\
You are optimising a few-shot prompt for MEP product descriptions.

OBJECTIVE: Find a prompt that achieves HIGH quality AND uses FEW TOKENS.
A prompt scoring 60/65 at 200 tokens beats one scoring 63/65 at 800 tokens.

CURRENT CHAMPION ({champ_tok} tokens, quality {quality}/{max_possible}):
{champion}

CURRENT FAILURES (products where expected trade terms were missing):
{failures}

STRATEGY FOR THIS ITERATION:
{strategy}

RULES:
- The prompt must remain self-contained and complete
- If you use examples, they must teach FORMAT (quoting style, size bridge) not vocabulary
- The inline instruction can carry vocabulary hints more efficiently than examples
- Output ONLY the improved prompt — no explanation, no preamble"""

async def generate_challenger(champion: str, failures: str, strategy: str,
                               quality: int) -> str:
    tok = len(champion) // 4
    prompt = OPTIMIZER_PROMPT.format(
        champ_tok=tok, quality=quality, max_possible=MAX_POSSIBLE,
        champion=champion, failures=failures, strategy=strategy,
    )
    result = await allm(prompt, max_tokens=700)
    # Strip preamble
    for pfx in ("Here is", "Improved:", "```\n", "---\n", "Sure,", "Certainly"):
        if result.startswith(pfx):
            idx = result.find("\n")
            if idx >= 0:
                result = result[idx + 1:]
    result = result.strip().strip("`").strip()
    if len(result) < 80:
        return ""
    return result

# ---------------------------------------------------------------------------
# Phase 1 — Champion-challenger
# ---------------------------------------------------------------------------
async def run_prefpo() -> str:
    champion = START_PROMPT
    print(f"\nScoring starting prompt...")
    champ_composite, champ_quality, champ_details = await score_async(champion)
    champ_tok = len(champion) // 4
    print(f"  Start: quality={champ_quality}/{MAX_POSSIBLE}, "
          f"tokens={champ_tok}, composite={champ_composite:.1f}")

    print(f"\n{'='*65}")
    print(f"PrefPO — compression-first | {ITERATIONS} rounds | {CHALLENGERS} challengers/round")
    print(f"Score = quality - token_penalty (TARGET={TARGET_TOKENS} tok, weight={TOKEN_WEIGHT})")
    print(f"{'='*65}")

    improvements = 0

    for i in range(ITERATIONS):
        failures = format_failures(champ_details)
        strats = [STRATEGIES[(i * CHALLENGERS + k) % len(STRATEGIES)]
                  for k in range(CHALLENGERS)]

        # Generate challengers concurrently
        chal_tasks = [generate_challenger(champion, failures, s, champ_quality)
                      for s in strats]
        challengers = await asyncio.gather(*chal_tasks)
        challengers = [c for c in challengers if c]

        if not challengers:
            print(f"  round {i+1:2d}: all challengers empty")
            continue

        # Score all concurrently
        score_tasks = [score_async(c) for c in challengers]
        results = await asyncio.gather(*score_tasks)

        best_comp, best_q, best_det, best_idx = champ_composite, champ_quality, champ_details, -1
        for j, (comp, q, det) in enumerate(results):
            if comp > best_comp:
                best_comp, best_q, best_det, best_idx = comp, q, det, j

        if best_idx >= 0:
            champion       = challengers[best_idx]
            champ_composite = best_comp
            champ_quality   = best_q
            champ_details   = best_det
            improvements   += 1
            tok = len(champion) // 4
            print(f"  round {i+1:2d}: NEW CHAMPION ✓  "
                  f"quality={champ_quality}/{MAX_POSSIBLE}  "
                  f"tokens={tok}  composite={champ_composite:.1f}")
        else:
            scores_str = " | ".join(
                f"q{q}t{len(c)//4}" for (comp, q, _), c in zip(results, challengers)
            )
            print(f"  round {i+1:2d}: held  "
                  f"(q={champ_quality},t={len(champion)//4},c={champ_composite:.1f})  "
                  f"challengers: [{scores_str}]")

    print(f"\nFinal champion:")
    print(f"  Quality:   {champ_quality}/{MAX_POSSIBLE} ({100*champ_quality//MAX_POSSIBLE}%)")
    print(f"  Tokens:    {len(champion)//4}")
    print(f"  Composite: {champ_composite:.1f}")
    print(f"  Improved:  {improvements} times")
    print(f"\nRemaining failures:")
    for d in champ_details:
        if d["misses"]:
            print(f"  [{d['domain']}] {d['input'][:40]} → {d['misses']}")
    return champion

# ---------------------------------------------------------------------------
# Phase 2 — 50/50/50 comparison
# ---------------------------------------------------------------------------
def infer_domain(r: dict):
    t = ((r.get("description","") or "") + " " + (r.get("product_category","") or "")).lower()
    if any(k in t for k in ["connector","cable","conduit","breaker","switch","lug","terminal",
        "led","lamp","outlet","gpo","fuse","rcbo","relay","contactor","enclosure","circuit",
        "bushing","wiring","socket","plug","gland"]):
        return "Electrical"
    if any(k in t for k in ["plumb","water","pipe","valve","drain","trap","coupling",
        "toilet","sink","shower","pex","cpvc","faucet","fitting"]):
        return "Plumbing"
    if any(k in t for k in ["hvac","ahu","vav","fcu","duct","chiller","boiler",
        "coil","damper","vrf","fan coil","refrigerant"]):
        return "Mechanical"
    return None

async def run_comparison(champion: str, all_records: list, cache: dict) -> list:
    print(f"\n{'='*65}")
    print("50/50/50 comparison: current Qdrant descriptions vs PrefPO...")

    enriched = [r for r in all_records
                if cache.get(r["id"]) and r.get("extended_description")
                and len((r.get("extended_description") or "").split()) >= 6]
    by_domain: dict[str, list] = {"Electrical": [], "Plumbing": [], "Mechanical": []}
    for r in enriched:
        d = infer_domain(r)
        if d:
            by_domain[d].append(r)

    random.seed(42)
    lines = [
        "# Enrichment: Current Qdrant (450 tok, domain-aware) vs PrefPO Champion\n\n",
        f"**Current (Qdrant cloud):** ~450 tokens  \n",
        f"**PrefPO champion:** ~{len(champion)//4} tokens  \n\n---\n",
    ]

    for domain in ("Electrical", "Plumbing", "Mechanical"):
        prods = random.sample(by_domain[domain], min(50, len(by_domain[domain])))
        outs = await asyncio.gather(
            *[agenerate(champion, product_to_user_msg(r)) for r in prods],
            return_exceptions=True)
        lines.append(f"\n## {domain} ({len(prods)} products)\n\n")
        for idx, (r, pout) in enumerate(zip(prods, outs), 1):
            iid  = r.get("internal_id", r.get("id", "?"))
            cur  = (r.get("extended_description") or "").strip()
            out  = str(pout) if not isinstance(pout, Exception) else "[err]"
            lines.append(f"### {idx}. `{iid}` — {r.get('description','')[:65]}\n")
            if r.get("product_category"):
                lines.append(f"*{r['product_category']}*  \n")
            lines.append(f"\n**CURRENT (Qdrant):**  \n{cur}\n\n")
            lines.append(f"**PREFPO:**  \n{out}\n\n---\n")
        print(f"  {domain}: done")
    return lines

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    print("Loading products and cache...")
    all_records = load_all(verbose=False, attach_caches=True)
    cache = json.load(open("enrichment_cache.json"))
    print(f"  {len(all_records)} products, {len(cache)} cached")

    champion = await run_prefpo()

    with open("optimized_enrichment_prompt.txt", "w") as f:
        f.write(champion)
    print("\nSaved → optimized_enrichment_prompt.txt")

    lines = await run_comparison(champion, all_records, cache)
    with open("enrichment_comparison.md", "w") as f:
        f.writelines(lines)
    print("Saved → enrichment_comparison.md\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
