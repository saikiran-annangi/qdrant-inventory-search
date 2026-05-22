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

import pandas as pd

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


def normalize_specs(text: str) -> str:
    """
    Expand unit abbreviations in text for BM25 index coverage.

    Example: "16A 1 POLE" -> "16a 16amp 16ampere 1p single pole"
    """
    if not text:
        return ""
    result = text.upper()
    for pattern, replacement in SPEC_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result.lower()


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
