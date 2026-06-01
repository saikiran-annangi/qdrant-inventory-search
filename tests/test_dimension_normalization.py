"""Black-box assertion suite for the pint-backed normalize_specs.

Each case lists an input string and the set of size/mm anchor tokens that
MUST appear in the output. We don't pin the exact output (extra electrical
expansions, ordering, whitespace can vary) — we just check the canonical
anchors are present.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.normalizers import normalize_specs, size_anchor_tokens, doc_size_anchors

# Each case: (input, required_anchor_tokens, optional_forbidden_tokens)
CASES = [
    # --- 2 inches: every surface form must produce size200 ---
    ("2 inches locknut",        {"size200"}, None),
    ("2 INCHES LOCKNUT",        {"size200"}, None),
    ("2 inch locknut",          {"size200"}, None),
    ("2 INCH LOCKNUT",          {"size200"}, None),
    ("2 in locknut",            {"size200"}, None),
    ("2IN locknut",             {"size200"}, None),
    ("2in locknut",             {"size200"}, None),
    ('2" locknut',              {"size200"}, None),
    ("2-inch locknut",          {"size200"}, None),
    ("2-INCH LOCKNUT",          {"size200"}, None),

    # --- 1/2 inch ---
    ("1/2 inch coupling",       {"size50"},  None),
    ("1/2 INCHES COUPLING",     {"size50"},  None),
    ("1/2in coupling",          {"size50"},  None),
    ('1/2" coupling',           {"size50"},  None),
    ("1/2-inch coupling",       {"size50"},  None),

    # --- 3/4 inch ---
    ("3/4 inch pipe",           {"size75"},  None),
    ('3/4" pipe',               {"size75"},  None),

    # --- mixed: 1-1/2 ---
    ("1-1/2 inches conduit",    {"size150"}, None),
    ("1-1/2 IN CONDUIT",        {"size150"}, None),
    ('1-1/2" conduit',          {"size150"}, None),

    # --- mixed: 2-1/2 ---
    ("STEEL LOCKNUT 2-1/2IN",   {"size250"}, None),
    ("2-1/2 inch fitting",      {"size250"}, None),

    # --- decimals ---
    ("0.5 inch fitting",        {"size50"},  None),
    ("2.5IN fitting",           {"size250"}, None),

    # --- millimeters ---
    ("50mm conduit",            {"mm50"},    None),
    ("50 MM CONDUIT",           {"mm50"},    None),
    ("25mm fitting",            {"mm25"},    None),

    # --- conflict-detection precondition: different sizes must produce different anchors
    #     (so apply_size_sort can tell "2in query" from "2-1/2in document") ---
    ("2 inches",                {"size200"}, {"size250"}),
    ("2-1/2 inch",              {"size250"}, {"size200"}),

    # --- electrical expansions still work (no regression in unrelated path) ---
    ("16A 1 POLE",              set(),       None),  # no size; should not crash

    # --- mm² (wire cross-section) MUST NOT be read as a length ---
    ("CABLE 2.5MM2 BLUE",       set(),       {"mm2", "mm3"}),
    ("CABLE 1.5MM2 BLACK",      set(),       {"mm1", "mm2"}),
    ("LUG 70MM2 10MM HOLE",     {"mm10"},    {"mm70"}),  # mm² ignored, "10MM HOLE" kept
    ("TERMINAL 25MM2 RED",      set(),       {"mm25"}),
]


def main():
    fails = []
    for text, required, forbidden in CASES:
        out = normalize_specs(text)
        tokens = doc_size_anchors(text)
        missing = required - tokens
        wrong   = (forbidden or set()) & tokens
        status  = "OK" if not (missing or wrong) else "FAIL"
        if status == "FAIL":
            fails.append((text, required, forbidden, tokens, out))
        print(f"  [{status}]  {text!r:<32}  tokens={sorted(tokens)}  -> {out!r}")

    print()
    if fails:
        print(f"FAILURES: {len(fails)}/{len(CASES)}")
        for text, req, forb, got, out in fails:
            print(f"  {text!r}")
            print(f"    required {sorted(req)}, forbidden {sorted(forb or [])}, got {sorted(got)}")
        sys.exit(1)
    print(f"All {len(CASES)} cases passed.")


if __name__ == "__main__":
    main()
