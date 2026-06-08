"""Query-side trade-jargon synonym expansion.

The complement to description enrichment. Enrichment (index-side) bakes synonyms
into each product's extended_description; this layer (query-side) expands the
QUERY with canonical/alternate terms before encoding. It's more general because
it doesn't depend on every product happening to contain every synonym — and it's
the durable feedback hook:

    *** To permanently fix a reported "query X doesn't find product Y" miss,
        add ONE entry below mapping the user's wording → the wording products
        use. That closes the gap for that query AND all similar ones, forever. ***

Applied in models/embeddings.encode_query() to the dense and BM25-description
channels (NOT the model-number channel — synonyms must not pollute part-number
matching). It is purely additive: a query with no trigger is returned unchanged,
so it can never hurt recall.

Triggers are matched case-insensitively on word boundaries; multi-word triggers
are preferred over single words. Expansions are appended (deduped, and only
terms not already in the query are added).

Coverage is electrical/MEP trade jargon (AU + North American), since the catalog
is an electrical distributor's. Pairs are bidirectional where both wordings are
common in queries (e.g. outlet ↔ gpo, earth ↔ ground).
"""

from __future__ import annotations

import re

# trigger phrase  ->  extra terms to append when it appears in a query
SYNONYM_MAP: dict[str, str] = {

    # =========================================================================
    # ELECTRICAL
    # =========================================================================

    # --- conduit fittings & support ---
    "dry connector":   "bx flex armored connector non liquidtight",
    "dry connectors":  "bx flex armored connector non liquidtight",
    "p clamp":         "cable clamp conduit clamp saddle strap p-clip",
    "p clamps":        "cable clamp conduit clamp saddle strap p-clip",
    "pclamp":          "cable clamp conduit clamp saddle p-clip",
    "saddle":          "clamp clip strap mount",
    "conduit saddle":  "conduit clamp strap clip",
    "catenary":        "support wire suspension",
    "loom":            "corrugated conduit flexible split tube",
    "tek screw":       "self drilling screw self tapping",

    # --- outlets / sockets (AU "GPO"/"power point" ↔ US "outlet"/"receptacle") ---
    "gpo":             "outlet socket receptacle power point",
    "power point":     "gpo outlet socket receptacle",
    "powerpoint":      "gpo outlet socket receptacle",
    "outlet":          "gpo socket receptacle power point",
    "receptacle":      "gpo outlet socket power point",
    "double adapter":  "plug adaptor double socket",

    # --- circuit protection ---
    "rcbo":            "residual current circuit breaker overcurrent",
    "rcd":             "residual current device safety switch",
    "elcb":            "earth leakage circuit breaker residual current",
    "safety switch":   "rcd residual current device",
    "mcb":             "miniature circuit breaker",
    "breaker":         "circuit breaker mcb",
    "fuse link":       "fuse carrier cartridge",

    # --- cable & wire ---
    "tps":             "twin earth flat sheathed building wire",
    "twin and earth":  "tps flat building wire",
    "building wire":   "tps single insulated conductor",
    "figure 8":        "speaker cable parallel",
    "earth wire":      "ground grounding conductor",

    # --- terminations ---
    "lug":             "terminal connector crimp cable lug",
    "choc block":      "terminal strip connector block",
    "chocolate block": "terminal strip connector block",
    "gland":           "cable gland connector cord grip",

    # --- lighting ---
    "downlight":       "recessed luminaire downlighter",
    "batten":          "linear luminaire fluorescent led batten",
    "oyster":          "ceiling light surface luminaire",
    "bunker":          "wall light exterior luminaire",
    "ballast":         "driver control gear",
    "starter":         "fluorescent starter ignitor",
    "globe":           "lamp bulb led",
    "bulb":            "lamp globe led",

    # --- earthing / grounding (US ↔ AU) ---
    "earth":           "ground grounding earthing",
    "ground":          "earth earthing grounding",
    "earthing":        "ground grounding bonding",

    # --- detection / automation ---
    "smoke alarm":     "smoke detector",
    "sensor light":    "pir motion sensor security light",
    "weatherproof":    "ip rated outdoor exterior",
    "din rail":        "din mount rail terminal",

    # =========================================================================
    # PLUMBING
    # =========================================================================

    # --- flexible / rubber couplings (one of the most common plumbing misses) ---
    "fernco":          "flexible coupling no-hub rubber coupling mission coupling drain pipe",
    "fernco coupling": "flexible rubber coupling no-hub mission drain connector",
    "no hub coupling": "fernco flexible rubber coupling drain pipe connector",
    "mission coupling":"fernco no-hub flexible rubber coupling drain",
    "rubber coupling": "fernco no-hub flexible coupling drain connector",

    # --- push-fit / quick connect fittings ---
    "shark bite":      "push fit push to connect speedfit quick connect plumbing",
    "sharkbite":       "push fit push to connect speedfit quick connect plumbing",
    "push fit":        "push to connect sharkbite speedfit quick connect",
    "speedfit":        "push fit push to connect quick connect plumbing fitting",

    # --- drain & waste fittings ---
    "p trap":          "bottle trap u bend drain trap waste fitting sink trap",
    "bottle trap":     "p trap u bend drain trap waste fitting",
    "u bend":          "p trap bottle trap drain trap waste pipe",
    "floor waste":     "floor drain floor trap floor gully",
    "floor gully":     "floor drain floor waste trap",

    # --- pipe joining ---
    "sweat fitting":   "solder fitting soldered joint capillary copper fitting",
    "solder fitting":  "sweat fitting capillary copper soldered joint",
    "compression fitting": "olives ferrule compression joint water fitting",
    "olive":           "compression ring ferrule fitting",

    # --- valves ---
    "isolation valve": "ball valve shutoff valve stop valve gate valve",
    "stopcock":        "isolation valve ball valve shutoff gate valve",
    "stop valve":      "ball valve isolation valve shutoff",
    "gate valve":      "isolation valve shutoff stop valve",

    # --- pipe sizing (NPS / CTS naming confusion) ---
    "nps":             "nominal pipe size ips iron pipe size thread",
    "cts":             "copper tube size copper pipe od outside diameter",
    "ips":             "iron pipe size nps nominal thread",

    # --- water heating ---
    "hot water system": "water heater storage tank element",
    "hws":             "hot water system water heater storage",
    "tempering valve": "thermostatic mixing valve tmv",
    "tmv":             "tempering valve thermostatic mixing valve",

    # =========================================================================
    # MECHANICAL / HVAC
    # =========================================================================

    # --- terminal units ---
    "vav":             "variable air volume terminal unit hvac",
    "vav box":         "variable air volume terminal unit reheat box hvac",
    "fcu":             "fan coil unit hydronic terminal chilled water",
    "fan coil":        "fcu fan coil unit hydronic air conditioning",
    "ahu":             "air handling unit air handler hvac",
    "air handler":     "ahu air handling unit hvac",
    "doas":            "dedicated outdoor air system ventilation unit hvac",

    # --- refrigerant / cooling systems ---
    "vrf":             "variable refrigerant flow vrf system multi-split hvac",
    "vrv":             "variable refrigerant volume vrf system multi-split hvac",
    "mini split":      "split system ductless ac vrf hvac",
    "ductless":        "mini split split system vrf hvac air conditioning",
    "split system":    "mini split ductless air conditioning hvac",

    # --- ductwork ---
    "flex connection": "flexible duct connector vibration isolator canvas connection",
    "canvas connection":"flexible duct connector vibration isolator flex connection",
    "vcd":             "volume control damper balancing damper hvac",
    "balancing damper":"volume control damper vcd hvac",
    "diffuser":        "air diffuser supply grille ceiling register hvac",

    # --- hydronic / piping ---
    "circuit setter":  "balancing valve flow setter hydronic",
    "air separator":   "air eliminator air purger hvac hydronic",
    "expansion tank":  "expansion vessel pressure vessel hydronic",
    "strainer":        "y strainer line strainer basket strainer pipe filter",
    "y strainer":      "strainer line strainer filter pipe",
    "actuator":        "valve actuator motorised valve hvac control",
}

# Build a match order: longest (most specific) triggers first so "power point"
# wins before "point", etc.
_TRIGGERS = sorted(SYNONYM_MAP, key=len, reverse=True)
_TRIGGER_RES = [(t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)) for t in _TRIGGERS]


def expand_synonyms(query: str) -> str:
    """Append canonical/alternate terms for any trade jargon found in `query`.

    Additive and order-preserving: returns the original query followed by any new
    expansion tokens (deduped, lowercased, excluding terms already present). A
    query with no recognized jargon is returned unchanged.
    """
    if not query:
        return query
    present = set(re.findall(r"[a-z0-9]+", query.lower()))
    extra: list[str] = []
    seen = set(present)
    for trigger, rx in _TRIGGER_RES:
        if rx.search(query):
            for tok in SYNONYM_MAP[trigger].split():
                t = tok.lower()
                if t not in seen:
                    seen.add(t)
                    extra.append(tok)
    if not extra:
        return query
    return f"{query} {' '.join(extra)}"
