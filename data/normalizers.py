"""
Text normalization utilities shared by the data loader and the search pipeline.

Functions:
  normalize_manufacturer  -- canonicalize manufacturer names via alias table
  normalize_specs         -- expand unit abbreviations for BM25 matching
  model_number_variants   -- produce casing/separator variants of a model number
  make_id                 -- deterministic MD5 UUID from (source, internal_id)
  is_sparse_description   -- detect null or very short descriptions
"""

import re
import hashlib
import logging

import pandas as pd
import pint

_log = logging.getLogger(__name__)

# Single shared unit registry — pint is for the unit math; the regex layer below
# is only for finding number-with-unit spans inside free text (which pint itself
# doesn't do). Inch sign (") and the metric `mm` alias don't need extra config —
# they're in pint's default registry.
_UREG = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
_INCH = _UREG.inch
_MM   = _UREG.millimeter

# ---------------------------------------------------------------------------
# Manufacturer alias table
# ---------------------------------------------------------------------------

MFR_ALIASES: dict[str, str] = {
    "SQUARE D":                         "SCHNEIDER ELECTRIC",
    "SCHNEIDER":                        "SCHNEIDER ELECTRIC",
    "CUTLER-HAMMER":                    "EATON",
    "CUTLER HAMMER":                    "EATON",
    "MOELLER":                          "EATON",
    "KLOCKNER MOELLER":                 "EATON",
    "MERLIN GERIN":                     "SCHNEIDER ELECTRIC",
    "TELEMECANIQUE":                    "SCHNEIDER ELECTRIC",
    "GENERAL ELECTRIC":                 "GE",
    "GE INDUSTRIAL":                    "GE",
    "3M CANADA":                        "3M",
    "3M INDUSTRIAL ADHESIVE & TAPES":   "3M",
}

# ---------------------------------------------------------------------------
# Spec expansion patterns for the sparse_desc BM25 field
# ---------------------------------------------------------------------------

SPEC_PATTERNS = [
    # Amperage: 16A -> 16a 16amp 16ampere
    (r"\b(\d+\.?\d*)\s*A\b",    lambda m: f"{m.group(1)}a {m.group(1)}amp {m.group(1)}ampere"),
    # Kiloamp: 6KA -> 6ka 6kiloamp
    (r"\b(\d+\.?\d*)\s*KA\b",   lambda m: f"{m.group(1)}ka {m.group(1)}kiloamp"),
    # Poles
    (r"\b1\s*POLE\b",            lambda _: "1p single pole"),
    (r"\b2\s*POLE\b",            lambda _: "2p two pole double pole"),
    (r"\b3\s*POLE\b",            lambda _: "3p three pole triple pole"),
    # Trip curves
    (r"\bC\s*CURVE\b",           lambda _: "c curve type c"),
    (r"\bB\s*CURVE\b",           lambda _: "b curve type b"),
    (r"\bD\s*CURVE\b",           lambda _: "d curve type d"),
    # DIN rail
    (r"\bDIN\s*MOUNT\b",         lambda _: "din rail din mount"),
    (r"\bDIN\s*RAIL\b",          lambda _: "din rail din mount"),
    # Voltage: 230V -> 230v 230volt
    (r"\b(\d+\.?\d*)\s*V\b",    lambda m: f"{m.group(1)}v {m.group(1)}volt"),
    # Watts: 500W -> 500w 500watt
    (r"\b(\d+\.?\d*)\s*W\b",    lambda m: f"{m.group(1)}w {m.group(1)}watt"),
    # Milliamps: 30MA -> 30ma 30milliamp
    (r"\b(\d+\.?\d*)\s*MA\b",   lambda m: f"{m.group(1)}ma {m.group(1)}milliamp"),
]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def normalize_manufacturer(name: str) -> str:
    """Return the canonical manufacturer name using the alias table."""
    if not name or pd.isna(name):
        return ""
    name = str(name).strip().upper()
    return MFR_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Dimension normalization (inch / fraction / mixed / mm)
# ---------------------------------------------------------------------------
# Sizes appear in many inconsistent surface forms across the catalog: glued
# `4IN`, fractions `3/4IN`, mixed `1-1/4IN`, double-quote `2"`, metric `50MM`.
# BM25 splits on punctuation, so `3/4in` shatters and `1/2IN` collides with
# `2IN` (both yield a `2in` token). We collapse every size to a punctuation-free,
# high-IDF anchor applied identically to documents and queries:
#
#     sizeN   N = round(inches * 100)   e.g. 2" -> size200, 3/4" -> size75
#     mmN     N = round(millimetres)    e.g. 50MM -> mm50
#
# bridge_metric=True additionally cross-emits the other unit system on the
# QUERY side (inch->mm with a +/-1 window, mm->inch) so an imperial query can
# reach the metric catalog. It is noisy (nominal sizes), hence opt-in.

MM_PER_IN = 25.4

# One scanner finds any "<number-form><unit>" span; pint then does the unit math.
# The number form may be: int, decimal, simple fraction (3/4), or mixed (1-1/2).
# The unit may be: in / inch / inches (case-insensitive) / " / mm (case-insensitive),
# with optional space- or hyphen-separator between number and unit. This single
# pattern replaces the previous trio of handcrafted dimension regexes.
_NUM_FORM   = r"\d+(?:\.\d+)?(?:-\d+/\d+|/\d+)?"
_UNIT_FORM  = r"inches|inch|in|\"|mm"
_DIM_SCAN   = re.compile(
    # Lookbehind: don't match mid-word (e.g. "P12in" -> "12in").
    # Lookahead:  must NOT be followed by a letter or digit. Excluding digits
    #             is what stops "2.5MM2" (wire cross-section mm^2) and "MM22"
    #             from being read as a 2.5 mm length. Lets through spaces,
    #             punctuation, end-of-string.
    rf"(?<![A-Za-z0-9_])({_NUM_FORM})[\s\-]*({_UNIT_FORM})(?![A-Za-z0-9²])",
    re.IGNORECASE,
)
_ANCHOR_RE  = re.compile(r"\b(?:size|mm)\d+\b")


def _parse_number(s: str):  # -> Optional[float]; py3.9 compat (no PEP 604 union)
    """Turn '2', '2.5', '3/4', '1-1/2' into a float. None on unparseable."""
    s = s.strip()
    m = re.fullmatch(r"(\d+)-(\d+)/(\d+)", s)        # 1-1/2
    if m:
        a, b, c = (int(g) for g in m.groups())
        return a + (b / c) if c else None
    m = re.fullmatch(r"(\d+)/(\d+)", s)              # 3/4
    if m:
        a, b = (int(g) for g in m.groups())
        return a / b if b else None
    try:
        return float(s)
    except ValueError:
        return None


def _quantity(num_str: str, unit_str: str):
    """Build a pint Quantity from messy MEP forms. None if anything fails."""
    val = _parse_number(num_str)
    if val is None:
        return None
    u = unit_str.lower()
    if u in ('"', "in", "inch", "inches"):
        return val * _INCH
    if u == "mm":
        return val * _MM
    return None


def _inch_tokens(inches: float, bridge_metric: bool) -> list:
    toks = [f"size{int(round(inches * 100))}"]
    if float(inches).is_integer():
        toks.append(f"{int(inches)}in")
    if bridge_metric:
        base = int(round(inches * MM_PER_IN))
        toks += [f"mm{base + d}" for d in (-1, 0, 1)]
    return toks


def _mm_tokens(mm: float, bridge_metric: bool) -> list:
    toks = [f"mm{int(round(mm))}"]
    if bridge_metric:
        toks.append(f"size{int(round((mm / MM_PER_IN) * 100))}")
    return toks


def _dim_replacement(match: re.Match, bridge_metric: bool) -> str:
    q = _quantity(match.group(1), match.group(2))
    if q is None:
        return match.group(0)
    if q.units == _INCH:
        toks = _inch_tokens(q.to("inch").magnitude, bridge_metric)
    elif q.units == _MM:
        toks = _mm_tokens(q.to("mm").magnitude, bridge_metric)
    else:
        return match.group(0)
    return " " + " ".join(toks) + " "


def normalize_specs(text: str, bridge_metric: bool = False) -> str:
    """
    Expand unit abbreviations for BM25 index coverage, including size/dimension
    attributes collapsed to canonical anchor tokens. Safe to apply identically
    to documents (ingest) and queries.

    Examples:
        "16A 1 POLE" -> "16a 16amp 16ampere 1p single pole"
        "2 inch pipe" / "2 inches pipe" / "2-inch pipe" / '2" pipe' / "2IN pipe"
            -> "size200 2in pipe"
        '3/4" coupling' / "3/4 inch coupling" -> "size75 coupling"
        "1-1/2 inches conduit" -> "size150 conduit"
    """
    if not text:
        return ""
    result = text.upper()
    for pattern, replacement in SPEC_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = result.lower()

    result = _DIM_SCAN.sub(lambda m: _dim_replacement(m, bridge_metric), result)

    return re.sub(r"\s+", " ", result).strip()


def size_anchor_tokens(query: str, bridge_metric: bool = False) -> set:
    """Canonical sizeNNN/mmNNN anchors a query is asking for (size-aware rerank)."""
    return set(_ANCHOR_RE.findall(normalize_specs(query, bridge_metric=bridge_metric)))


def doc_size_anchors(text: str) -> set:
    """Canonical size anchors present in a document's text (no metric bridging)."""
    return set(_ANCHOR_RE.findall(normalize_specs(text or "")))


def model_number_variants(model: str) -> str:
    """
    Generate multiple surface forms of a model number for BM25 matching.

    Produces: original, lowercase, slash-to-space, slash-to-hyphen,
    alphanum-only, alphanum-only-lowercase.
    """
    if not model or pd.isna(model):
        return ""
    m = str(model).strip()
    variants = {
        m,
        m.lower(),
        m.replace("/", " "),
        m.replace("/", "-"),
        m.replace("+", ""),       # ABH120-4042EV  -- full model as one BM25 token
        m.replace("+", "-"),      # ABH120-4042E-V -- hyphen-safe variant
        re.sub(r"[^a-zA-Z0-9]", "", m),
        re.sub(r"[^a-zA-Z0-9]", "", m).lower(),
    }
    return " ".join(variants)


def make_id(source: str, internal_id: str) -> str:
    """
    Deterministic UUID from (source, internal_id).

    Using MD5 ensures the same product always maps to the same Qdrant point ID,
    making ingestion idempotent via upsert.
    """
    raw = f"{source}:{internal_id}"
    return hashlib.md5(raw.encode()).hexdigest()


def is_sparse_description(desc) -> bool:
    """Return True if the description is missing or fewer than 5 meaningful words."""
    if not desc or pd.isna(desc):
        return True
    words = [w for w in str(desc).split() if len(w) > 1]
    return len(words) < 5
